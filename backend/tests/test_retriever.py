import unittest
from unittest.mock import MagicMock, patch

from langchain_core.embeddings import Embeddings

from app.rag.document_repository import SearchHit
from app.rag.retriever import PgVectorRetriever


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


class PgVectorRetrieverTest(unittest.TestCase):
    @patch("app.rag.retriever.DocumentRepository")
    @patch("app.rag.retriever.SessionLocal")
    def test_converts_database_hits_to_langchain_documents(
        self,
        session_local: MagicMock,
        repository_class: MagicMock,
    ) -> None:
        repository_class.return_value.search_similar.return_value = [
            SearchHit(
                chunk_id=11,
                document_id=3,
                content="학자금 지원 대상에 관한 내용이다.",
                heading="학자금 > 지원 대상",
                file_path="복지제도/학자금.md",
                file_name="학자금.md",
                category="복지제도",
                similarity=0.91,
            )
        ]
        session_local.return_value.__enter__.return_value = MagicMock()
        retriever = PgVectorRetriever(
            embeddings=FakeEmbeddings(),
            search_limit=5,
            category="복지제도",
        )

        with self.assertLogs("uvicorn.error", level="INFO") as logs:
            documents = retriever.invoke("학자금 지원 대상은 누구인가요?")

        self.assertEqual(len(documents), 1)
        self.assertTrue(
            any(
                "학자금 지원 대상은 누구인가요?" in log
                for log in logs.output
            )
        )
        self.assertEqual(documents[0].metadata["file_name"], "학자금.md")
        self.assertEqual(documents[0].metadata["similarity"], 0.91)
        repository_class.return_value.search_similar.assert_called_once_with(
            query_embedding=[1.0, 0.0],
            limit=5,
            category="복지제도",
        )


if __name__ == "__main__":
    unittest.main()
