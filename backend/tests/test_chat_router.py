import unittest
from unittest.mock import MagicMock

from app.chat.router import ChatRouter
from app.chat.schemas import ChatIntent, IntentResult
from app.rag.chain import RagAnswer


class ChatRouterTest(unittest.TestCase):
    def test_routes_policy_question_to_rag_with_history(self) -> None:
        classifier = MagicMock()
        classifier.classify.return_value = IntentResult(
            intent=ChatIntent.POLICY_QUESTION
        )
        rag_chain = MagicMock()
        expected = RagAnswer(
            reply="기본 근무시간은 오전 9시부터 오후 6시까지입니다.",
            sources=[],
        )
        rag_chain.answer.return_value = expected
        router = ChatRouter(
            intent_classifier=classifier,
            rag_chain=rag_chain,
        )
        history = [("user", "근무제도가 궁금해")]

        result = router.answer("근무시간이 어떻게 되나요?", history)

        self.assertEqual(result, expected)
        classifier.classify.assert_called_once_with(
            "근무시간이 어떻게 되나요?",
            history,
        )
        rag_chain.answer.assert_called_once_with(
            question="근무시간이 어떻게 되나요?",
            history=history,
        )

    def test_returns_classifier_reply_without_rag_for_non_policy_routes(
        self,
    ) -> None:
        cases = (
            (
                ChatIntent.CASUAL,
                "오늘 많이 지치셨나 봐요.",
            ),
            (
                ChatIntent.AMBIGUOUS,
                "사용 가능 여부와 신청 방법 중 무엇이 궁금하신가요?",
            ),
            (
                ChatIntent.OUT_OF_SCOPE,
                "핑거의 사내 규정과 업무 절차를 안내해 드릴 수 있어요.",
            ),
        )

        for intent, reply in cases:
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

                result = router.answer("테스트 메시지")

                self.assertEqual(result.reply, reply)
                self.assertEqual(result.sources, [])
                classifier.classify.assert_called_once_with(
                    "테스트 메시지",
                    (),
                )
                rag_chain.answer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
