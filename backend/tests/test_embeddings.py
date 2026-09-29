import unittest
from unittest.mock import MagicMock, patch

from app.rag.embeddings import E5Embeddings, build_document_embedding_text


class E5EmbeddingsTest(unittest.TestCase):
    @patch("app.rag.embeddings.HuggingFaceEmbeddings")
    def test_applies_document_and_query_prefixes(self, embeddings_class: MagicMock) -> None:
        delegate = embeddings_class.return_value
        delegate.embed_documents.return_value = [[1.0, 0.0]]
        delegate.embed_query.return_value = [1.0, 0.0]
        embeddings = E5Embeddings()

        document_result = embeddings.embed_documents(["문서 내용"])
        query_result = embeddings.embed_query("질문 내용")

        delegate.embed_documents.assert_called_once_with(["passage: 문서 내용"])
        delegate.embed_query.assert_called_once_with("query: 질문 내용")
        self.assertEqual(document_result, [[1.0, 0.0]])
        self.assertEqual(query_result, [1.0, 0.0])

    @patch("app.rag.embeddings.HuggingFaceEmbeddings")
    def test_configures_normalized_local_embeddings(
        self,
        embeddings_class: MagicMock,
    ) -> None:
        E5Embeddings(device="cpu")

        _, kwargs = embeddings_class.call_args
        self.assertEqual(kwargs["model_kwargs"], {"device": "cpu"})
        self.assertTrue(kwargs["encode_kwargs"]["normalize_embeddings"])

    def test_combines_heading_and_content_for_document_embedding(self) -> None:
        text = build_document_embedding_text(
            content="신청은 전자결재로 진행한다.",
            heading="복지제도 > 학자금 > 신청 방법",
        )

        self.assertEqual(
            text,
            "복지제도 > 학자금 > 신청 방법\n\n신청은 전자결재로 진행한다.",
        )


if __name__ == "__main__":
    unittest.main()
