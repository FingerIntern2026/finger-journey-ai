import unittest

from pydantic import ValidationError

from app.chat.schemas import ChatIntent, IntentResult


class IntentResultTest(unittest.TestCase):
    def test_accepts_each_supported_intent(self) -> None:
        for intent in ChatIntent:
            with self.subTest(intent=intent):
                result = IntentResult(intent=intent)

                self.assertEqual(result.intent, intent)
                self.assertIsNone(result.reply)

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


if __name__ == "__main__":
    unittest.main()
