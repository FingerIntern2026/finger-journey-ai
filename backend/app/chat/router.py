from collections.abc import Sequence
from functools import lru_cache

from app.rag.chain import RagAnswer, RagAnswerChain, get_rag_answer_chain

from .intent_classifier import IntentClassifier, get_intent_classifier
from .schemas import ChatIntent


class ChatRouter:
    def __init__(
        self,
        intent_classifier: IntentClassifier | None = None,
        rag_chain: RagAnswerChain | None = None,
    ) -> None:
        self.intent_classifier = intent_classifier or get_intent_classifier()
        self.rag_chain = rag_chain or get_rag_answer_chain()

    def answer(
        self,
        message: str,
        history: Sequence[tuple[str, str]] = (),
    ) -> RagAnswer:
        intent_result = self.intent_classifier.classify(message, history)

        if intent_result.intent == ChatIntent.POLICY_QUESTION:
            return self.rag_chain.answer(
                question=message,
                history=history,
            )

        if intent_result.reply is None:
            raise RuntimeError(
                f"Missing reply for {intent_result.intent.value}."
            )

        return RagAnswer(
            reply=intent_result.reply,
            sources=[],
        )


@lru_cache(maxsize=1)
def get_chat_router() -> ChatRouter:
    return ChatRouter()
