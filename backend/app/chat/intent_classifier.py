import logging
import os
from collections.abc import Sequence
from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.rag.config import GEMINI_CHAT_MODEL

from .prompts import INTENT_SYSTEM_PROMPT, INTENT_USER_PROMPT
from .schemas import IntentResult


logger = logging.getLogger("uvicorn.error")

MAX_INTENT_HISTORY_MESSAGES = 6


class IntentClassifier:
    def __init__(self, chat_model: BaseChatModel | None = None) -> None:
        self.chat_model = chat_model or self._create_chat_model()
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", INTENT_SYSTEM_PROMPT),
                ("human", INTENT_USER_PROMPT),
            ]
        )
        structured_model = self.chat_model.with_structured_output(IntentResult)
        self.classification_chain = prompt | structured_model

    def classify(
        self,
        message: str,
        history: Sequence[tuple[str, str]] = (),
    ) -> IntentResult:
        raw_result = self.classification_chain.invoke(
            {
                "history": self._format_recent_history(history),
                "message": message,
            }
        )
        result = IntentResult.model_validate(raw_result)
        logger.info("Chat intent=%s message=%r", result.intent.value, message)
        return result

    @staticmethod
    def _format_recent_history(
        history: Sequence[tuple[str, str]],
    ) -> str:
        recent_history = history[-MAX_INTENT_HISTORY_MESSAGES:]
        if not recent_history:
            return "이전 대화 없음"
        return "\n".join(
            f"{role}: {text}" for role, text in recent_history
        )

    @staticmethod
    def _create_chat_model() -> ChatGoogleGenerativeAI:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY must be configured.")
        return ChatGoogleGenerativeAI(
            model=GEMINI_CHAT_MODEL,
            google_api_key=api_key,
            temperature=0,
        )


@lru_cache(maxsize=1)
def get_intent_classifier() -> IntentClassifier:
    return IntentClassifier()
