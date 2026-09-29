import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.chat.router import ChatRouter
from app.chat.schemas import ChatIntent, IntentResult
from app.main import app
from app.rag.chain import AnswerSource, RagAnswer


class ChatApiRoutesTest(unittest.TestCase):
    def setUp(self) -> None:
        warm_up_patcher = patch("app.main.warm_up_rag")
        self.addCleanup(warm_up_patcher.stop)
        warm_up_patcher.start()
        self.client_context = TestClient(app)
        self.client = self.client_context.__enter__()
        self.addCleanup(self.client_context.__exit__, None, None, None)

    def test_policy_question_returns_rag_answer_and_sources(self) -> None:
        classifier = MagicMock()
        classifier.classify.return_value = IntentResult(
            intent=ChatIntent.POLICY_QUESTION
        )
        rag_chain = MagicMock()
        rag_chain.answer.return_value = RagAnswer(
            reply="기본 근무시간은 오전 9시부터 오후 6시까지입니다.",
            sources=[
                AnswerSource(
                    file_name="취업규칙.md",
                    category="취업규칙",
                    heading="근로시간",
                    similarity=0.95,
                )
            ],
        )
        router = ChatRouter(
            intent_classifier=classifier,
            rag_chain=rag_chain,
        )

        with patch("app.main.get_chat_router", return_value=router):
            response = self.client.post(
                "/api/chat",
                json={
                    "message": "근무시간이 어떻게 되나요?",
                    "history": [],
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("오전 9시", body["reply"])
        self.assertEqual(len(body["sources"]), 1)
        self.assertEqual(body["sources"][0]["file_name"], "취업규칙.md")
        rag_chain.answer.assert_called_once()

    def test_non_policy_routes_return_reply_without_rag_search(self) -> None:
        cases = (
            (
                ChatIntent.CASUAL,
                "집에 가고 싶다",
                "오늘 많이 지치셨나 봐요.",
            ),
            (
                ChatIntent.AMBIGUOUS,
                "반차 쓰고 집에 갈까?",
                "사용 가능 여부와 신청 방법 중 무엇이 궁금하신가요?",
            ),
            (
                ChatIntent.OUT_OF_SCOPE,
                "오늘 날씨 알려줘",
                "핑거의 사내 규정과 업무 절차를 안내해 드릴 수 있어요.",
            ),
        )

        for intent, message, reply in cases:
            with self.subTest(intent=intent):
                classifier = MagicMock()
                classifier.classify.return_value = IntentResult(
                    intent=intent,
                    reply=reply,
                )
                rag_chain = MagicMock()
                router = ChatRouter(
                    intent_classifier=classifier,
                    rag_chain=rag_chain,
                )

                with patch("app.main.get_chat_router", return_value=router):
                    response = self.client.post(
                        "/api/chat",
                        json={"message": message, "history": []},
                    )

                self.assertEqual(response.status_code, 200)
                self.assertEqual(
                    response.json(),
                    {"reply": reply, "sources": []},
                )
                rag_chain.answer.assert_not_called()

    def test_rejects_empty_message_before_routing(self) -> None:
        with patch("app.main.get_chat_router") as get_chat_router:
            response = self.client.post(
                "/api/chat",
                json={"message": "", "history": []},
            )

        self.assertEqual(response.status_code, 422)
        get_chat_router.assert_not_called()


if __name__ == "__main__":
    unittest.main()
