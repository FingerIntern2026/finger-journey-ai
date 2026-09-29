from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from langchain_core.embeddings import Embeddings

from ..database import SessionLocal
from .config import DOCUMENTS_DIRECTORY
from .document_loader import MarkdownDocumentLoader
from .document_repository import DocumentRepository
from .embeddings import build_document_embedding_text, get_embeddings
from .text_splitter import MarkdownChunker


@dataclass
class IngestionError:
    file_path: str
    message: str


@dataclass
class IngestionResult:
    found: int = 0
    ingested: int = 0
    updated: int = 0
    skipped: int = 0
    failed: int = 0
    chunks_created: int = 0
    errors: list[IngestionError] = field(default_factory=list)


class IngestionService:
    def __init__(
        self,
        document_directory: Path = DOCUMENTS_DIRECTORY,
        embeddings: Embeddings | None = None,
    ) -> None:
        self.loader = MarkdownDocumentLoader(document_directory)
        self.chunker = MarkdownChunker()
        self.embeddings = embeddings

    def ingest_directory(self) -> IngestionResult:
        files = self.loader.discover()
        result = IngestionResult(found=len(files))

        for file_path in files:
            self._ingest_file(file_path, result)

        return result

    def ingest_file(self, file_path: Path) -> IngestionResult:
        result = IngestionResult(found=1)
        self._ingest_file(file_path, result)
        return result

    def _ingest_file(self, file_path: Path, result: IngestionResult) -> None:
        loaded_document = None
        try:
            loaded_document = self.loader.load(file_path)
            checksum = self.loader.checksum(file_path)

            with SessionLocal() as session:
                repository = DocumentRepository(session)
                db_document, is_new, is_skipped = repository.prepare_document(
                    loaded_document.metadata,
                    checksum,
                )
                document_id = db_document.id

            if is_skipped:
                result.skipped += 1
                return

            chunks = self.chunker.split(
                loaded_document.page_content,
                loaded_document.metadata,
            )
            if not chunks:
                raise ValueError("No searchable chunks were created from the document.")

            embedding_inputs = [
                build_document_embedding_text(
                    chunk.page_content,
                    self._optional_text(chunk.metadata.get("heading")),
                )
                for chunk in chunks
            ]
            embeddings = self.embeddings or get_embeddings()
            vectors = embeddings.embed_documents(embedding_inputs)

            with SessionLocal() as session:
                DocumentRepository(session).replace_chunks(
                    document_id,
                    chunks,
                    vectors,
                )

            if is_new:
                result.ingested += 1
            else:
                result.updated += 1
            result.chunks_created += len(chunks)
        except Exception as exc:
            file_path_text = (
                str(loaded_document.metadata["file_path"])
                if loaded_document
                else file_path.name
            )
            with SessionLocal() as session:
                DocumentRepository(session).mark_failed(file_path_text, str(exc))
            result.failed += 1
            result.errors.append(
                IngestionError(file_path=file_path_text, message=str(exc))
            )

    @staticmethod
    def _optional_text(value: object) -> str | None:
        return str(value) if value else None


@lru_cache(maxsize=1)
def get_ingestion_service() -> IngestionService:
    return IngestionService()
