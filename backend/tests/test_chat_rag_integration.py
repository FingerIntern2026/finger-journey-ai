import unittest
from unittest.mock import MagicMock

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda
from pydantic import Field

from app.chat.intent_classifier import IntentClassifier
from app.chat.router import ChatRouter
from app.chat.schemas import ChatIntent, IntentResult
from app.rag.chain import RagAnswerChain


class RecordingRetriever(BaseRetriever):
    documents: list[Document]
    queries: list[str] = Field(default_factory=list)

    def _get_relevant_documents(self, query: str, *, run_manager):
        self.queries.append(query)
        return self.documents


class ChatRagIntegrationTest(unittest.TestCase):
    def test_routes_contextual_policy_follow_up_through_real_rag_chain(
        self,
    ) -> None:
        classifier_model = MagicMock()
        classifier_model.with_structured_output.return_value = RunnableLambda(
            lambda _: IntentResult(intent=ChatIntent.POLICY_QUESTION)
        )
        intent_classifier = IntentClassifier(chat_model=classifier_model)
        retriever = RecordingRetriever(
            documents=[
                Document(
                    page_content=(
                        "시차출퇴근제는 08:00~17:00 또는 "
                        "10:00~19:00로 운영한다."
                    ),
                    metadata={
                        "document_id": 1,
                        "file_name": "취업규칙.md",
                        "category": "취업규칙",
                        "heading": "시차출퇴근제 운영방법",
                        "similarity": 0.94,
                    },
                )
            ]
        )
        rag_chain = RagAnswerChain(
            retriever=retriever,
            chat_model=FakeListChatModel(
                responses=[
                    "시차출퇴근제 세부 운영방법을 알려줘",
                    "시차출퇴근제는 두 가지 시간대로 운영됩니다.",
                ]
            ),
        )
        router = ChatRouter(
            intent_classifier=intent_classifier,
            rag_chain=rag_chain,
        )
        history = [
            ("user", "시차출퇴근제 신청 방법을 알려줘"),
            ("assistant", "세부 운영방법도 안내해 드릴까요?"),
        ]

        result = router.answer("응, 알려줘", history)

        self.assertEqual(
            retriever.queries,
            ["시차출퇴근제 세부 운영방법을 알려줘"],
        )
        self.assertIn("두 가지 시간대", result.reply)
        self.assertEqual(len(result.sources), 1)
        self.assertEqual(result.sources[0].file_name, "취업규칙.md")


if __name__ == "__main__":
    unittest.main()
