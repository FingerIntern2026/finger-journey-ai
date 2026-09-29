import hashlib
from pathlib import Path

from langchain_core.documents import Document


class MarkdownDocumentLoader:
    def __init__(self, root_directory: Path) -> None:
        self.root_directory = root_directory.resolve()

    def discover(self) -> list[Path]:
        if not self.root_directory.is_dir():
            raise FileNotFoundError(
                f"Document directory does not exist: {self.root_directory}"
            )
        return sorted(self.root_directory.rglob("*.md"))

    def load(self, file_path: Path) -> Document:
        resolved_path = file_path.resolve()
        if not resolved_path.is_relative_to(self.root_directory):
            raise ValueError("Document path must be inside the document directory.")
        if resolved_path.suffix.lower() != ".md":
            raise ValueError("Only Markdown documents are supported.")

        relative_path = resolved_path.relative_to(self.root_directory)
        category = relative_path.parts[0] if len(relative_path.parts) > 1 else ""

        return Document(
            page_content=resolved_path.read_text(encoding="utf-8-sig"),
            metadata={
                "file_path": relative_path.as_posix(),
                "file_name": resolved_path.name,
                "category": category,
            },
        )

    @staticmethod
    def checksum(file_path: Path) -> str:
        digest = hashlib.sha256()
        with file_path.open("rb") as document_file:
            for block in iter(lambda: document_file.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
