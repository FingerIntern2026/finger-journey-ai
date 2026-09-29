import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_history_aware_retriever

from .config import GEMINI_CHAT_MODEL
from .prompts import (
    CONTEXTUALIZE_SYSTEM_PROMPT,
    CONTEXTUALIZE_USER_PROMPT,
    RAG_SYSTEM_PROMPT,
    RAG_USER_PROMPT,
)
from .retriever import PgVectorRetriever


logger = logging.getLogger("uvicorn.error")


@dataclass(frozen=True)
class AnswerSource:
    file_name: str
    category: str
    heading: str | None
    similarity: float


@dataclass(frozen=True)
class RagAnswer:
    reply: str
    sources: list[AnswerSource]


class RagAnswerChain:
    def __init__(
        self,
        retriever: BaseRetriever | None = None,
        chat_model: BaseChatModel | None = None,
    ) -> None:
        self.retriever = retriever or PgVectorRetriever()
        self.chat_model = chat_model or self._create_chat_model()
        contextualize_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CONTEXTUALIZE_SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", CONTEXTUALIZE_USER_PROMPT),
            ]
        )
        self.history_aware_retriever = create_history_aware_retriever(
            llm=self.chat_model,
            retriever=self.retriever,
            prompt=contextualize_prompt,
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", RAG_SYSTEM_PROMPT),
                ("human", RAG_USER_PROMPT),
            ]
        )
        self.answer_chain = prompt | self.chat_model | StrOutputParser()

    def answer(
        self,
        question: str,
        history: Sequence[tuple[str, str]] = (),
    ) -> RagAnswer:
        documents = self.history_aware_retriever.invoke(
            {
                "input": question,
                "chat_history": self._to_chat_messages(history),
            }
        )
        if not documents:
            return RagAnswer(
                reply="관련 사내 문서를 찾지 못해 답변을 확인할 수 없다.",
                sources=[],
            )

        self._log_evidence(question, documents)

        reply = self.answer_chain.invoke(
            {
                "context": self._format_context(documents),
                "history": self._format_history(history),
                "question": question,
            }
        )
        return RagAnswer(
            reply=reply,
            sources=self._collect_sources(documents),
        )

    @staticmethod
    def _log_evidence(
        question: str,
        documents: Sequence[Document],
    ) -> None:
        logger.info("RAG evidence for question=%r", question)
        for index, document in enumerate(documents, start=1):
            metadata = document.metadata
            evidence = " ".join(document.page_content.split())[:500]
            logger.info(
                "RAG evidence #%d file=%s heading=%s similarity=%.4f content=%s",
                index,
                metadata.get("file_name", "알 수 없음"),
                metadata.get("heading") or "제목 없음",
                float(metadata.get("similarity", 0.0)),
                evidence,
            )

    @staticmethod
    def _format_context(documents: Sequence[Document]) -> str:
        sections = []
        for index, document in enumerate(documents, start=1):
            metadata = document.metadata
            sections.append(
                "\n".join(
                    [
                        f"[참고 자료 {index}]",
                        f"문서: {metadata.get('file_name', '알 수 없음')}",
                        f"분류: {metadata.get('category', '알 수 없음')}",
                        f"항목: {metadata.get('heading') or '제목 없음'}",
                        f"내용:\n{document.page_content}",
                    ]
                )
            )
        return "\n\n".join(sections)

    @staticmethod
    def _format_history(history: Sequence[tuple[str, str]]) -> str:
        if not history:
            return "이전 대화 없음"
        return "\n".join(f"{role}: {text}" for role, text in history)

    @staticmethod
    def _to_chat_messages(
        history: Sequence[tuple[str, str]],
    ) -> list[BaseMessage]:
        messages: list[BaseMessage] = []
        for role, text in history:
            if role == "user":
                messages.append(HumanMessage(content=text))
            elif role == "assistant":
                messages.append(AIMessage(content=text))
            else:
                raise ValueError(f"Unsupported chat history role: {role}")
        return messages

    @staticmethod
    def _collect_sources(documents: Sequence[Document]) -> list[AnswerSource]:
        sources = []
        seen = set()
        for document in documents:
            metadata = document.metadata
            key = (
                metadata.get("document_id"),
                metadata.get("heading"),
            )
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                AnswerSource(
                    file_name=str(metadata.get("file_name", "알 수 없음")),
                    category=str(metadata.get("category", "알 수 없음")),
                    heading=(
                        str(metadata["heading"])
                        if metadata.get("heading")
                        else None
                    ),
                    similarity=float(metadata.get("similarity", 0.0)),
                )
            )
        return sources

    @staticmethod
    def _create_chat_model() -> ChatGoogleGenerativeAI:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY must be configured.")
        return ChatGoogleGenerativeAI(
            model=GEMINI_CHAT_MODEL,
            google_api_key=api_key,
        )


@lru_cache(maxsize=1)
def get_rag_answer_chain() -> RagAnswerChain:
    return RagAnswerChain()
