from typing import Iterator

import pytest
from qdrant_client import QdrantClient
from testcontainers.community.qdrant import QdrantContainer

from src.utils.qdrant import QdrantSession

QDRANT_IMAGE = "qdrant/qdrant:latest"


def pytest_collection_modifyitems(items):
    """Marks every test in this folder as `integration`, so `pytest -m "not integration"` skips them."""
    for item in items:
        if "integration" in item.path.parts:
            item.add_marker(pytest.mark.integration)


@pytest.fixture(scope="session")
def qdrant_container() -> Iterator[QdrantContainer]:
    """One throwaway Qdrant server for the whole test run, removed afterwards."""
    with QdrantContainer(QDRANT_IMAGE) as container:
        yield container


@pytest.fixture
def qdrant_client(qdrant_container: QdrantContainer) -> Iterator[QdrantClient]:
    """A client on the test server; every collection is dropped after each test, so tests can't leak data."""
    client = qdrant_container.get_client()
    yield client
    for collection in client.get_collections().collections:
        client.delete_collection(collection.name)
    client.close()


@pytest.fixture
def session(qdrant_client: QdrantClient) -> QdrantSession:
    return QdrantSession(qdrant_client)
