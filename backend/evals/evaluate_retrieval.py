import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

from app.rag.embeddings import get_embeddings
from app.rag.retriever import PgVectorRetriever


DEFAULT_CASES_PATH = Path(__file__).with_name("retrieval_cases.json")


@dataclass(frozen=True)
class RetrievalCase:
    query: str
    expected_files: frozenset[str]


@dataclass(frozen=True)
class CaseResult:
    query: str
    rank: int | None
    elapsed_seconds: float
    retrieved_files: list[str]


def load_cases(path: Path = DEFAULT_CASES_PATH) -> list[RetrievalCase]:
    raw_cases = json.loads(path.read_text(encoding="utf-8"))
    return [
        RetrievalCase(
            query=case["query"],
            expected_files=frozenset(case["expected_files"]),
        )
        for case in raw_cases
    ]


def find_expected_rank(
    retrieved_files: list[str],
    expected_files: frozenset[str],
) -> int | None:
    for index, file_name in enumerate(retrieved_files, start=1):
        if file_name in expected_files:
            return index
    return None


def calculate_metrics(results: list[CaseResult]) -> dict[str, float]:
    if not results:
        return {"hit_at_1": 0.0, "hit_at_5": 0.0, "mrr": 0.0, "avg_seconds": 0.0}

    total = len(results)
    return {
        "hit_at_1": sum(result.rank == 1 for result in results) / total,
        "hit_at_5": sum(
            result.rank is not None and result.rank <= 5
            for result in results
        )
        / total,
        "mrr": sum(1 / result.rank if result.rank else 0 for result in results)
        / total,
        "avg_seconds": sum(result.elapsed_seconds for result in results) / total,
    }


def run_evaluation(limit: int) -> tuple[list[CaseResult], dict[str, float]]:
    embeddings = get_embeddings()
    retriever = PgVectorRetriever(embeddings=embeddings, search_limit=limit)
    results = []

    for case in load_cases():
        started_at = time.perf_counter()
        documents = retriever.invoke(case.query)
        elapsed_seconds = time.perf_counter() - started_at
        retrieved_files = [
            str(document.metadata["file_name"])
            for document in documents
        ]
        results.append(
            CaseResult(
                query=case.query,
                rank=find_expected_rank(retrieved_files, case.expected_files),
                elapsed_seconds=elapsed_seconds,
                retrieved_files=retrieved_files,
            )
        )

    return results, calculate_metrics(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pgvector retrieval quality.")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--fail-under-hit-at-5", type=float, default=0.0)
    args = parser.parse_args()

    results, metrics = run_evaluation(args.limit)
    for result in results:
        rank = result.rank if result.rank is not None else "MISS"
        print(f"[{rank}] {result.query}")
        print(f"    top: {result.retrieved_files[:3]}")

    print()
    print(f"cases={len(results)}")
    print(f"hit@1={metrics['hit_at_1']:.3f}")
    print(f"hit@5={metrics['hit_at_5']:.3f}")
    print(f"mrr={metrics['mrr']:.3f}")
    print(f"avg_seconds={metrics['avg_seconds']:.3f}")

    if metrics["hit_at_5"] < args.fail_under_hit_at_5:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
