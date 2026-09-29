import unittest

from app.rag.prompts import (
    CONTEXTUALIZE_SYSTEM_PROMPT,
    CONTEXTUALIZE_USER_PROMPT,
    RAG_SYSTEM_PROMPT,
)


class RagPromptTest(unittest.TestCase):
    def test_contextualizer_creates_only_a_standalone_search_question(self) -> None:
        self.assertIn("독립적인 검색 질문", CONTEXTUALIZE_SYSTEM_PROMPT)
        self.assertIn("직접 답하지", CONTEXTUALIZE_SYSTEM_PROMPT)
        self.assertIn("검색 질문 한 문장만", CONTEXTUALIZE_SYSTEM_PROMPT)
        self.assertIn("원문을 그대로 반환", CONTEXTUALIZE_SYSTEM_PROMPT)
        self.assertIn("{input}", CONTEXTUALIZE_USER_PROMPT)

    def test_requires_progressive_answering(self) -> None:
        self.assertIn("직접 물어본 정보만", RAG_SYSTEM_PROMPT)
        self.assertIn("답변 길이는 문장 수로 고정하지", RAG_SYSTEM_PROMPT)
        self.assertIn("정확히 이해하는 데 필요한 만큼", RAG_SYSTEM_PROMPT)
        self.assertIn("후속 질문으로만 제안", RAG_SYSTEM_PROMPT)
        self.assertIn("시차출퇴근제의 시간대나 운영방법", RAG_SYSTEM_PROMPT)

    def test_actively_suggests_answerable_contextual_follow_up(self) -> None:
        self.assertIn("후속 질문을 적극적으로 제안", RAG_SYSTEM_PROMPT)
        self.assertIn("추론이 필요", RAG_SYSTEM_PROMPT)
        self.assertIn("대상이 명확", RAG_SYSTEM_PROMPT)
        self.assertIn("사용할 수 있다", RAG_SYSTEM_PROMPT)
        self.assertIn("근거가 전혀 없을 때만", RAG_SYSTEM_PROMPT)

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
