import unittest

from pydantic import ValidationError

from app.chat.schemas import ChatIntent, IntentResult


class IntentResultTest(unittest.TestCase):
    def test_accepts_policy_intent_without_reply(self) -> None:
        result = IntentResult(intent=ChatIntent.POLICY_QUESTION)

        self.assertEqual(result.intent, ChatIntent.POLICY_QUESTION)
        self.assertIsNone(result.reply)

    def test_accepts_reply_for_each_non_policy_intent(self) -> None:
        for intent in (
            ChatIntent.CASUAL,
            ChatIntent.AMBIGUOUS,
            ChatIntent.OUT_OF_SCOPE,
        ):
            with self.subTest(intent=intent):
                result = IntentResult(intent=intent, reply="  안내 응답  ")

                self.assertEqual(result.intent, intent)
                self.assertEqual(result.reply, "안내 응답")

    def test_accepts_reply_for_non_policy_route(self) -> None:
        result = IntentResult(
            intent=ChatIntent.CASUAL,
            reply="오늘 많이 지치셨나 봐요.",
        )

        self.assertEqual(result.intent, ChatIntent.CASUAL)
        self.assertEqual(result.reply, "오늘 많이 지치셨나 봐요.")

    def test_rejects_unknown_intent(self) -> None:
        with self.assertRaises(ValidationError):
            IntentResult(intent="UNKNOWN")

    def test_rejects_policy_intent_with_reply(self) -> None:
        with self.assertRaisesRegex(
            ValidationError,
            "POLICY_QUESTION must not include a reply",
        ):
            IntentResult(
                intent=ChatIntent.POLICY_QUESTION,
                reply="분류기가 만든 정책 답변",
            )

    def test_rejects_non_policy_intent_without_reply(self) -> None:
        for intent in (
            ChatIntent.CASUAL,
            ChatIntent.AMBIGUOUS,
            ChatIntent.OUT_OF_SCOPE,
        ):
            with self.subTest(intent=intent):
                with self.assertRaisesRegex(
                    ValidationError,
                    "must include a reply",
                ):
                    IntentResult(intent=intent)

    def test_rejects_blank_non_policy_reply(self) -> None:
        with self.assertRaisesRegex(ValidationError, "must include a reply"):
            IntentResult(intent=ChatIntent.CASUAL, reply="   ")


if __name__ == "__main__":
    unittest.main()
