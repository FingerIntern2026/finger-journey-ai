from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from .config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL_NAME,
)


class E5Embeddings(Embeddings):
    """LangChain embeddings adapter with E5 retrieval prefixes."""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        device: str = EMBEDDING_DEVICE,
    ) -> None:
        self._delegate = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={
                "batch_size": EMBEDDING_BATCH_SIZE,
                "normalize_embeddings": True,
            },
            show_progress=False,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        passages = [f"passage: {text}" for text in texts]
        return self._delegate.embed_documents(passages)

    def embed_query(self, text: str) -> list[float]:
        return self._delegate.embed_query(f"query: {text}")


def build_document_embedding_text(
    content: str,
    heading: str | None = None,
) -> str:
    if heading:
        return f"{heading.strip()}\n\n{content.strip()}"
    return content.strip()


@lru_cache(maxsize=1)
def get_embeddings() -> E5Embeddings:
    return E5Embeddings()
