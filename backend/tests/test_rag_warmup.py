import unittest
from unittest.mock import MagicMock, patch

from app.rag.warmup import WARMUP_QUERY, warm_up_rag


class RagWarmupTest(unittest.TestCase):
    @patch("app.rag.warmup.get_rag_answer_chain")
    @patch("app.rag.warmup.get_embeddings")
    def test_loads_model_and_runs_query_embedding(
        self,
        get_embeddings: MagicMock,
        get_rag_answer_chain: MagicMock,
    ) -> None:
        warm_up_rag()

        get_embeddings.return_value.embed_query.assert_called_once_with(
            WARMUP_QUERY
        )
        get_rag_answer_chain.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
