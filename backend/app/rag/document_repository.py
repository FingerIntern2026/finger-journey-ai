from collections.abc import Sequence

from langchain_core.documents import Document
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from ..models import RagDocument, RagDocumentChunk


class DocumentRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def find_by_file_path(self, file_path: str) -> RagDocument | None:
        return self.session.scalar(
            select(RagDocument).where(RagDocument.file_path == file_path)
        )

    def prepare_document(
        self,
        metadata: dict[str, object],
        checksum: str,
    ) -> tuple[RagDocument, bool, bool]:
        file_path = str(metadata["file_path"])
        document = self.find_by_file_path(file_path)

        if document and document.checksum == checksum and document.status == "COMPLETED":
            return document, False, True

        is_new = document is None
        if document is None:
            document = RagDocument(
                file_path=file_path,
                file_name=str(metadata["file_name"]),
                category=str(metadata["category"]),
                checksum=checksum,
                status="PENDING",
            )
            self.session.add(document)

        document.file_name = str(metadata["file_name"])
        document.category = str(metadata["category"])
        document.checksum = checksum
        document.status = "PROCESSING"
        document.error_message = None
        self.session.commit()

        return document, is_new, False

    def replace_chunks(
        self,
        document_id: int,
        chunks: Sequence[Document],
        embeddings: Sequence[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Chunk and embedding counts must match.")

        document = self.session.get(RagDocument, document_id)
        if document is None:
            raise ValueError(f"RAG document does not exist: {document_id}")

        self.session.execute(
            delete(RagDocumentChunk).where(
                RagDocumentChunk.document_id == document_id
            )
        )
        self.session.add_all(
            RagDocumentChunk(
                document_id=document_id,
                chunk_index=int(chunk.metadata["chunk_index"]),
                heading=self._optional_text(chunk.metadata.get("heading")),
                content=chunk.page_content,
                embedding=embedding,
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        )
        document.status = "COMPLETED"
        document.error_message = None
        self.session.commit()

    def mark_failed(self, file_path: str, error_message: str) -> None:
        document = self.find_by_file_path(file_path)
        if document is None:
            return
        document.status = "FAILED"
        document.error_message = error_message[:2_000]
        self.session.commit()

    @staticmethod
    def _optional_text(value: object) -> str | None:
        return str(value) if value else None
