from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env.test", override=False)

import pytest

from tests.mocks.ollama import FakeOllamaEmbedder


@pytest.fixture
def fake_embedder() -> FakeOllamaEmbedder:
    return FakeOllamaEmbedder(size=768)
