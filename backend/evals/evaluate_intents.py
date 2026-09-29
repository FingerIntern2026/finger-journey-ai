import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from app.chat.intent_classifier import get_intent_classifier
from app.chat.schemas import ChatIntent


DEFAULT_CASES_PATH = Path(__file__).with_name("intent_cases.json")


@dataclass(frozen=True)
class IntentCase:
    message: str
    expected_intent: ChatIntent


@dataclass(frozen=True)
class IntentCaseResult:
    message: str
    expected_intent: ChatIntent
    actual_intent: ChatIntent
    elapsed_seconds: float

    @property
    def passed(self) -> bool:
        return self.actual_intent == self.expected_intent


def load_cases(path: Path = DEFAULT_CASES_PATH) -> list[IntentCase]:
    raw_cases = json.loads(path.read_text(encoding="utf-8"))
    return [
        IntentCase(
            message=case["message"],
            expected_intent=ChatIntent(case["expected_intent"]),
        )
        for case in raw_cases
    ]


def calculate_accuracy(results: list[IntentCaseResult]) -> float:
    if not results:
        return 0.0
    return sum(result.passed for result in results) / len(results)


def run_evaluation() -> list[IntentCaseResult]:
    classifier = get_intent_classifier()
    results = []

    for case in load_cases():
        started_at = time.perf_counter()
        result = classifier.classify(case.message)
        elapsed_seconds = time.perf_counter() - started_at
        results.append(
            IntentCaseResult(
                message=case.message,
                expected_intent=case.expected_intent,
                actual_intent=result.intent,
                elapsed_seconds=elapsed_seconds,
            )
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Gemini chat intent classification."
    )
    parser.add_argument("--fail-under-accuracy", type=float, default=0.0)
    args = parser.parse_args()

    load_dotenv()
    results = run_evaluation()
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(
            f"[{status}] {result.message} "
            f"expected={result.expected_intent.value} "
            f"actual={result.actual_intent.value} "
            f"seconds={result.elapsed_seconds:.3f}"
        )

    accuracy = calculate_accuracy(results)
    average_seconds = (
        sum(result.elapsed_seconds for result in results) / len(results)
        if results
        else 0.0
    )
    print()
    print(f"cases={len(results)}")
    print(f"accuracy={accuracy:.3f}")
    print(f"avg_seconds={average_seconds:.3f}")

    if accuracy < args.fail_under_accuracy:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
