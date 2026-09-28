from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Mapping

class Parser(ABC):
    """Base interface for parsing a raw downloaded file into model-ready field dicts."""

    @abstractmethod
    def parse(self, raw: bytes) -> Iterator[List[Mapping[str, Any]]]:
        """Yields one kwargs dict per entry, ready to construct a QdrantBaseModel subclass."""
        pass
