import unittest

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.retrievers import BaseRetriever

from app.rag.chain import RagAnswerChain


class FakeRetriever(BaseRetriever):
    documents: list[Document]

    def _get_relevant_documents(self, query: str, *, run_manager):
        return self.documents


class RagAnswerChainTest(unittest.TestCase):
    def test_generates_answer_and_deduplicates_sources(self) -> None:
        documents = [
            Document(
                page_content="학자금은 임직원의 대학생 자녀에게 지원한다.",
                metadata={
                    "document_id": 1,
                    "file_name": "학자금지원기준.md",
                    "category": "복지제도",
                    "heading": "지원 대상",
                    "similarity": 0.91,
                },
            ),
            Document(
                page_content="지원 신청은 전자결재로 진행한다.",
                metadata={
                    "document_id": 1,
                    "file_name": "학자금지원기준.md",
                    "category": "복지제도",
                    "heading": "지원 대상",
                    "similarity": 0.89,
                },
            ),
        ]
        chain = RagAnswerChain(
            retriever=FakeRetriever(documents=documents),
            chat_model=FakeListChatModel(
                responses=["학자금은 임직원의 대학생 자녀에게 지원된다. [1]"]
            ),
        )

        answer = chain.answer(
            "학자금 지원 대상은 누구인가요?",
            history=[("user", "복지제도에 대해 알려줘")],
        )

        self.assertIn("대학생 자녀", answer.reply)
        self.assertEqual(len(answer.sources), 1)
        self.assertEqual(answer.sources[0].file_name, "학자금지원기준.md")

    def test_does_not_call_model_when_no_documents_are_found(self) -> None:
        model = FakeListChatModel(responses=["호출되면 안 된다."])
        chain = RagAnswerChain(
            retriever=FakeRetriever(documents=[]),
            chat_model=model,
        )

        answer = chain.answer("자료에 없는 질문")

        self.assertIn("찾지 못해", answer.reply)
        self.assertEqual(answer.sources, [])


if __name__ == "__main__":
    unittest.main()
