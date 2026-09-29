import unittest

from app.rag.prompts import RAG_SYSTEM_PROMPT


class RagPromptTest(unittest.TestCase):
    def test_requires_progressive_answering(self) -> None:
        self.assertIn("핵심 답을 바로 제시", RAG_SYSTEM_PROMPT)
        self.assertIn("묻지 않은 관련 규정을 한꺼번에 나열하지", RAG_SYSTEM_PROMPT)
        self.assertIn("후속 질문", RAG_SYSTEM_PROMPT)

    def test_allows_detailed_answers_when_requested(self) -> None:
        self.assertIn("상세 설명", RAG_SYSTEM_PROMPT)
        self.assertIn("충분히 구조화", RAG_SYSTEM_PROMPT)

    def test_requires_friendly_tone_and_one_trailing_emoji(self) -> None:
        self.assertIn("친근한 온보딩 챗봇", RAG_SYSTEM_PROMPT)
        self.assertIn("정확히 1개", RAG_SYSTEM_PROMPT)
        self.assertIn("본문 답변의 마지막 글자", RAG_SYSTEM_PROMPT)
        self.assertIn("후속 질문이 아니라", RAG_SYSTEM_PROMPT)

    def test_hides_internal_citation_numbers_from_user(self) -> None:
        self.assertIn("참고 자료 번호", RAG_SYSTEM_PROMPT)
        self.assertIn("넣지 말아야", RAG_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
