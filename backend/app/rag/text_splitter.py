import re
from collections.abc import Mapping

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from .config import CHUNK_OVERLAP, CHUNK_SIZE, MIN_CHUNK_CONTENT_LENGTH


HEADER_KEYS = tuple(f"h{level}" for level in range(1, 7))
YAML_FRONT_MATTER_PATTERN = re.compile(r"\A---\s*\n.*?\n---\s*(?:\n|\Z)", re.DOTALL)
HTML_COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.DOTALL)


class MarkdownChunker:
    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        min_content_length: int = MIN_CHUNK_CONTENT_LENGTH,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be between zero and chunk_size.")
        if min_content_length < 0:
            raise ValueError("min_content_length must not be negative.")

        self.min_content_length = min_content_length

        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#" * level, f"h{level}")
                for level in range(1, 7)
            ],
            strip_headers=True,
        )
        self.size_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", ". ", "! ", "? ", " ", ""],
        )

    def split(
        self,
        markdown: str,
        metadata: Mapping[str, object] | None = None,
    ) -> list[Document]:
        cleaned_markdown = self._clean_markdown(markdown)
        if not cleaned_markdown:
            return []

        sections = self.header_splitter.split_text(cleaned_markdown)
        if not sections:
            sections = [Document(page_content=cleaned_markdown)]

        chunks = self.size_splitter.split_documents(sections)
        base_metadata = dict(metadata or {})
        result: list[Document] = []

        for chunk in chunks:
            content = chunk.page_content.strip()
            content_length = len(re.sub(r"\s+", "", content))
            if content_length < self.min_content_length:
                continue

            chunk_metadata = {
                **base_metadata,
                **chunk.metadata,
            }
            heading_parts = [
                str(chunk_metadata[key]).strip()
                for key in HEADER_KEYS
                if chunk_metadata.get(key)
            ]
            chunk_metadata["heading"] = " > ".join(heading_parts) or None
            chunk_metadata["chunk_index"] = len(result)

            result.append(
                Document(
                    page_content=content,
                    metadata=chunk_metadata,
                )
            )

        return result

    @staticmethod
    def _clean_markdown(markdown: str) -> str:
        without_front_matter = YAML_FRONT_MATTER_PATTERN.sub("", markdown, count=1)
        without_comments = HTML_COMMENT_PATTERN.sub("", without_front_matter)
        return without_comments.strip()
