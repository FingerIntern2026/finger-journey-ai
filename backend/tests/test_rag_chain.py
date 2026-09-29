import unittest

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from app.rag.chain import RagAnswerChain


class FakeRetriever(BaseRetriever):
    documents: list[Document]
    queries: list[str] = Field(default_factory=list)

    def _get_relevant_documents(self, query: str, *, run_manager):
        self.queries.append(query)
        return self.documents


class RagAnswerChainTest(unittest.TestCase):
    def test_official_history_aware_retriever_uses_rewritten_question(
        self,
    ) -> None:
        retriever = FakeRetriever(documents=[])
        chain = RagAnswerChain(
            retriever=retriever,
            chat_model=FakeListChatModel(
                responses=["시차출퇴근제의 세부 운영방법은 무엇인가요?"]
            ),
        )

        documents = chain.history_aware_retriever.invoke(
            {
                "chat_history": chain._to_chat_messages(
                    [
                        ("user", "시차출퇴근제 신청 방법을 알려줘"),
                        (
                            "assistant",
                            "적용 가능한 세부 운영방법도 안내해 드릴까요?",
                        ),
                    ]
                ),
                "input": "세부 운영방법",
            }
        )

        self.assertEqual(documents, [])
        self.assertEqual(
            retriever.queries,
            ["시차출퇴근제의 세부 운영방법은 무엇인가요?"],
        )

    def test_rewrites_context_dependent_follow_up_in_answer_flow(self) -> None:
        retriever = FakeRetriever(documents=[])
        chain = RagAnswerChain(
            retriever=retriever,
            chat_model=FakeListChatModel(
                responses=["시차출퇴근제의 세부 운영방법은 무엇인가요?"]
            ),
        )

        answer = chain.answer(
            "세부 운영방법",
            history=[
                ("user", "시차출퇴근제 신청 방법을 알려줘"),
                (
                    "assistant",
                    "선택 가능한 출퇴근 시간도 안내해 드릴까요?",
                ),
            ],
        )

        self.assertEqual(
            retriever.queries,
            ["시차출퇴근제의 세부 운영방법은 무엇인가요?"],
        )
        self.assertIn("찾지 못해", answer.reply)

    def test_keeps_complete_new_topic_during_conversation(self) -> None:
        retriever = FakeRetriever(documents=[])
        chain = RagAnswerChain(
            retriever=retriever,
            chat_model=FakeListChatModel(
                responses=["학자금 지원 대상은 누구인가요?"]
            ),
        )

        chain.answer(
            "학자금 지원 대상은 누구인가요?",
            history=[
                ("user", "시차출퇴근제 신청 방법을 알려줘"),
                ("assistant", "희망일 5일 전까지 신청해야 해요."),
            ],
        )

        self.assertEqual(
            retriever.queries,
            ["학자금 지원 대상은 누구인가요?"],
        )

    def test_converts_history_to_langchain_messages(self) -> None:
        messages = RagAnswerChain._to_chat_messages(
            [
                ("user", "시차출퇴근제 신청 방법을 알려줘"),
                ("assistant", "희망일 5일 전까지 신청해야 해요."),
            ]
        )

        self.assertIsInstance(messages[0], HumanMessage)
        self.assertEqual(
            messages[0].content,
            "시차출퇴근제 신청 방법을 알려줘",
        )
        self.assertIsInstance(messages[1], AIMessage)
        self.assertEqual(
            messages[1].content,
            "희망일 5일 전까지 신청해야 해요.",
        )

    def test_rejects_unknown_history_role(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported chat history role"):
            RagAnswerChain._to_chat_messages([("system", "잘못된 역할")])

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
        retriever = FakeRetriever(documents=documents)
        chain = RagAnswerChain(
            retriever=retriever,
            chat_model=FakeListChatModel(
                responses=[
                    "학자금 지원 대상은 누구인가요?",
                    "학자금은 임직원의 대학생 자녀에게 지원된다. [1]",
                ]
            ),
        )

        with self.assertLogs("uvicorn.error", level="INFO") as logs:
            answer = chain.answer(
                "학자금 지원 대상은 누구인가요?",
                history=[("user", "복지제도에 대해 알려줘")],
            )

        self.assertIn("대학생 자녀", answer.reply)
        self.assertEqual(
            retriever.queries,
            ["학자금 지원 대상은 누구인가요?"],
        )
        self.assertEqual(len(answer.sources), 1)
        self.assertEqual(answer.sources[0].file_name, "학자금지원기준.md")
        self.assertTrue(any("학자금지원기준.md" in log for log in logs.output))
        self.assertTrue(any("대학생 자녀" in log for log in logs.output))

    def test_does_not_call_model_when_no_documents_are_found(self) -> None:
        model = FakeListChatModel(responses=["호출되면 안 된다."])
        retriever = FakeRetriever(documents=[])
        chain = RagAnswerChain(
            retriever=retriever,
            chat_model=model,
        )

        answer = chain.answer("자료에 없는 질문")

        self.assertIn("찾지 못해", answer.reply)
        self.assertEqual(answer.sources, [])
        self.assertEqual(retriever.queries, ["자료에 없는 질문"])


if __name__ == "__main__":
    unittest.main()
