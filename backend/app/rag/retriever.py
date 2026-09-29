from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

from ..database import SessionLocal
from .config import DEFAULT_RETRIEVAL_LIMIT, MAX_RETRIEVAL_LIMIT
from .document_repository import DocumentRepository
from .embeddings import get_embeddings


class PgVectorRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    embeddings: Embeddings = Field(default_factory=get_embeddings)
    search_limit: int = Field(
        default=DEFAULT_RETRIEVAL_LIMIT,
        ge=1,
        le=MAX_RETRIEVAL_LIMIT,
    )
    category: str | None = None

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        query_embedding = self.embeddings.embed_query(query)
        with SessionLocal() as session:
            hits = DocumentRepository(session).search_similar(
                query_embedding=query_embedding,
                limit=self.search_limit,
                category=self.category,
            )

        return [
            Document(
                page_content=hit.content,
                metadata={
                    "chunk_id": hit.chunk_id,
                    "document_id": hit.document_id,
                    "heading": hit.heading,
                    "file_path": hit.file_path,
                    "file_name": hit.file_name,
                    "category": hit.category,
                    "similarity": hit.similarity,
                },
            )
            for hit in hits
        ]
