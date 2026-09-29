import unittest

from evals.evaluate_retrieval import (
    CaseResult,
    calculate_metrics,
    find_expected_rank,
)


class RetrievalEvaluationTest(unittest.TestCase):
    def test_finds_first_matching_expected_file(self) -> None:
        rank = find_expected_rank(
            ["다른문서.md", "정답문서.md", "정답문서.md"],
            frozenset({"정답문서.md"}),
        )

        self.assertEqual(rank, 2)

    def test_calculates_retrieval_metrics(self) -> None:
        results = [
            CaseResult("질문1", 1, 0.1, ["a.md"]),
            CaseResult("질문2", 3, 0.3, ["b.md"]),
            CaseResult("질문3", None, 0.2, ["c.md"]),
        ]

        metrics = calculate_metrics(results)

        self.assertAlmostEqual(metrics["hit_at_1"], 1 / 3)
        self.assertAlmostEqual(metrics["hit_at_5"], 2 / 3)
        self.assertAlmostEqual(metrics["mrr"], (1 + 1 / 3) / 3)
        self.assertAlmostEqual(metrics["avg_seconds"], 0.2)


if __name__ == "__main__":
    unittest.main()
