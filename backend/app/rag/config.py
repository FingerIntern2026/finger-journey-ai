import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DOCUMENTS_DIRECTORY = PROJECT_ROOT / "data" / "documents"

CHUNK_SIZE = 1_000
CHUNK_OVERLAP = 150
MIN_CHUNK_CONTENT_LENGTH = 10

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"
EMBEDDING_DIMENSION = 768
EMBEDDING_BATCH_SIZE = 16
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
