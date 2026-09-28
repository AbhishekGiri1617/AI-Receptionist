from typing import Optional
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from config import settings

_embedder: Optional[SentenceTransformer] = None
_qdrant_client: Optional[QdrantClient] = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(settings.embedding_model)
    return _embedder


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        qdrant_url = getattr(settings, "qdrant_url", None)
        if qdrant_url:
            _qdrant_client = QdrantClient(url=qdrant_url, api_key=getattr(settings, "qdrant_api_key", None))
        else:
            _qdrant_client = QdrantClient(path=settings.qdrant_path)
    return _qdrant_client
