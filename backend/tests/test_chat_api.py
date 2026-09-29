import unittest
from unittest.mock import MagicMock, patch

from fastapi import HTTPException

from app.main import chat
from app.rag.chain import AnswerSource, RagAnswer
from app.schemas import ChatRequest, ChatTurn


class ChatApiTest(unittest.TestCase):
    @patch("app.main.get_chat_router")
    def test_routes_request_message_and_history_to_chat_router(
        self,
        get_chat_router: MagicMock,
    ) -> None:
        get_chat_router.return_value.answer.return_value = RagAnswer(
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
        request = ChatRequest(
            message="근무시간이 어떻게 되나요?",
            history=[
                ChatTurn(role="user", text="근무제도가 궁금해"),
                ChatTurn(role="assistant", text="어떤 제도인가요?"),
            ],
        )

        response = chat(request)

        get_chat_router.return_value.answer.assert_called_once_with(
            message="근무시간이 어떻게 되나요?",
            history=[
                ("user", "근무제도가 궁금해"),
                ("assistant", "어떤 제도인가요?"),
            ],
        )
        self.assertIn("오전 9시", response.reply)
        self.assertEqual(len(response.sources), 1)
        self.assertEqual(response.sources[0].file_name, "취업규칙.md")

    @patch("app.main.get_chat_router")
    def test_returns_generic_chat_error_when_router_fails(
        self,
        get_chat_router: MagicMock,
    ) -> None:
        get_chat_router.return_value.answer.side_effect = RuntimeError(
            "분류 실패"
        )

        with self.assertRaises(HTTPException) as raised:
            chat(ChatRequest(message="안녕하세요", history=[]))

        self.assertEqual(raised.exception.status_code, 500)
        self.assertEqual(
            raised.exception.detail,
            "챗봇 답변 생성에 실패했습니다.",
        )


if __name__ == "__main__":
    unittest.main()
