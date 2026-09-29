import unittest

from app.chat.schemas import ChatIntent
from evals.evaluate_intents import (
    IntentCaseResult,
    calculate_accuracy,
    load_cases,
)


class IntentEvaluationTest(unittest.TestCase):
    def test_loads_cases_with_supported_intents(self) -> None:
        cases = load_cases()

        self.assertGreater(len(cases), 0)
        self.assertTrue(
            all(isinstance(case.expected_intent, ChatIntent) for case in cases)
        )

    def test_calculates_accuracy(self) -> None:
        results = [
            IntentCaseResult(
                message="정책 질문",
                expected_intent=ChatIntent.POLICY_QUESTION,
                actual_intent=ChatIntent.POLICY_QUESTION,
                elapsed_seconds=0.1,
            ),
            IntentCaseResult(
                message="잡담",
                expected_intent=ChatIntent.CASUAL,
                actual_intent=ChatIntent.AMBIGUOUS,
                elapsed_seconds=0.1,
            ),
        ]

        self.assertEqual(calculate_accuracy(results), 0.5)
        self.assertEqual(calculate_accuracy([]), 0.0)


if __name__ == "__main__":
    unittest.main()
