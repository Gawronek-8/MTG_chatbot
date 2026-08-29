from qdrant_client import QdrantClient
from src.config import settings

# Global singleton to hold the client
_client = None

def get_qdrant_client() -> QdrantClient:
    """
    Returns a singleton instance of the QdrantClient configured from environment settings.
    """
    global _client
    if _client is None:
        _client = QdrantClient(url=settings.qdrant_url)
    return _client
