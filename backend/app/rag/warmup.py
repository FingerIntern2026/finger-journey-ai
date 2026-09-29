from .chain import get_rag_answer_chain
from .embeddings import get_embeddings


WARMUP_QUERY = "사내 규정 안내"


def warm_up_rag() -> None:
    """Load the local model and exercise the first query embedding at startup."""
    get_embeddings().embed_query(WARMUP_QUERY)
    get_rag_answer_chain()
