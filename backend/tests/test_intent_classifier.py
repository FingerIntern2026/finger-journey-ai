import unittest
from unittest.mock import MagicMock

from langchain_core.runnables import RunnableLambda

from app.chat.intent_classifier import (
    MAX_INTENT_HISTORY_MESSAGES,
    IntentClassifier,
)
from app.chat.schemas import ChatIntent, IntentResult


class IntentClassifierTest(unittest.TestCase):
    def test_returns_structured_intent_result(self) -> None:
        prompt_values = []
        expected = IntentResult(
            intent=ChatIntent.CASUAL,
            reply="오늘 많이 지치셨나 봐요.",
        )
        chat_model = MagicMock()
        chat_model.with_structured_output.return_value = RunnableLambda(
            lambda prompt_value: prompt_values.append(prompt_value) or expected
        )
        classifier = IntentClassifier(chat_model=chat_model)

        with self.assertLogs("uvicorn.error", level="INFO") as logs:
            result = classifier.classify("집에 가고 싶다")

        self.assertEqual(result, expected)
        chat_model.with_structured_output.assert_called_once_with(IntentResult)
        self.assertIn("집에 가고 싶다", prompt_values[0].messages[-1].content)
        self.assertTrue(any("CASUAL" in log for log in logs.output))

    def test_only_includes_recent_history_in_prompt(self) -> None:
        prompt_values = []
        chat_model = MagicMock()
        chat_model.with_structured_output.return_value = RunnableLambda(
            lambda prompt_value: prompt_values.append(prompt_value)
            or IntentResult(intent=ChatIntent.POLICY_QUESTION)
        )
        classifier = IntentClassifier(chat_model=chat_model)
        history = [
            ("user", f"메시지 {index}")
            for index in range(MAX_INTENT_HISTORY_MESSAGES + 2)
        ]

        classifier.classify("신청 방법 알려줘", history)

        user_prompt = prompt_values[0].messages[-1].content
        self.assertNotIn("메시지 0\n", user_prompt)
        self.assertNotIn("메시지 1\n", user_prompt)
        self.assertIn("메시지 2", user_prompt)
        self.assertIn("메시지 7", user_prompt)
        self.assertIn("신청 방법 알려줘", user_prompt)

    def test_formats_empty_history_explicitly(self) -> None:
        self.assertEqual(
            IntentClassifier._format_recent_history([]),
            "이전 대화 없음",
        )


if __name__ == "__main__":
    unittest.main()
