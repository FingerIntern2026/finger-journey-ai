import unittest

from app.chat.prompts import INTENT_SYSTEM_PROMPT, INTENT_USER_PROMPT


class IntentPromptTest(unittest.TestCase):
    def test_defines_all_supported_intents(self) -> None:
        for intent in (
            "POLICY_QUESTION",
            "CASUAL",
            "AMBIGUOUS",
            "OUT_OF_SCOPE",
        ):
            with self.subTest(intent=intent):
                self.assertIn(intent, INTENT_SYSTEM_PROMPT)

    def test_classifies_meaning_instead_of_similar_keywords(self) -> None:
        self.assertIn("문장 전체의 의미", INTENT_SYSTEM_PROMPT)
        self.assertIn("집에 가고 싶다", INTENT_SYSTEM_PROMPT)
        self.assertIn("퇴직 절차 질문으로 해석하면 안 된다", INTENT_SYSTEM_PROMPT)

    def test_uses_history_only_for_omitted_context(self) -> None:
        self.assertIn("현재 메시지를 우선", INTENT_SYSTEM_PROMPT)
        self.assertIn("생략된 대상이나 표현", INTENT_SYSTEM_PROMPT)

    def test_defines_reply_behavior_for_each_route(self) -> None:
        self.assertIn("POLICY_QUESTION이면 reply를 비워", INTENT_SYSTEM_PROMPT)
        self.assertIn("CASUAL이면", INTENT_SYSTEM_PROMPT)
        self.assertIn("간결한 확인 질문", INTENT_SYSTEM_PROMPT)
        self.assertIn("지원 범위", INTENT_SYSTEM_PROMPT)
        self.assertIn("분류명이나 내부 처리 방식을 노출하면 안 된다", INTENT_SYSTEM_PROMPT)

    def test_user_prompt_contains_history_and_current_message(self) -> None:
        self.assertIn("{history}", INTENT_USER_PROMPT)
        self.assertIn("{message}", INTENT_USER_PROMPT)


if __name__ == "__main__":
    unittest.main()
