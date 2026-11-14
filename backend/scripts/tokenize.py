from typing import Any, Mapping

from loguru import logger

from src.connections import StorageConnection
from src.connections.qdrant import get_qdrant_client
from src.parsers import Parser
from src.utils.langchain_utils import ollama_embedder
from src.utils.qdrant import QdrantBaseModel, QdrantSession


class Tokenizer[QdrantModel: QdrantBaseModel]:
    """
    Reads a file from storage, parses it into entries, embeds them and stores them in Qdrant.

    The parser yields entries with a deterministic `id` and no vector. Entries whose id is
    already stored are unchanged and skipped (not re-embedded), stored ids the file no longer
    produces are deleted, and only the remaining entries are embedded and stored.
    """

    # Kept small so one embedding request stays quick and progress is persisted often:
    # a failure halfway through (e.g. Ollama going down) keeps every batch already stored,
    # and a re-run only embeds what is still missing.
    BATCH_SIZE = 64

    def __init__(
        self,
        file_path: str,
        connection_cls: type[StorageConnection],
        parser: Parser,
        model_cls: type[QdrantModel],
        text_field: str,
    ):
        """
        :param text_field: The entry field whose text is embedded into the model's vector field.
        """
        vector_fields = list(model_cls.get_vector_fields())
        if len(vector_fields) != 1:
            raise ValueError(f"{model_cls.__name__} must define exactly one vector field, found {vector_fields}")

        self.file_path = file_path
        self.connection_cls = connection_cls
        self.parser = parser
        self.model_cls = model_cls
        self.text_field = text_field
        self.vector_field = vector_fields[0]

    def __call__(self) -> None:
        task_logger = logger.bind(task="Tokenizer")
        task_logger.info(f"Tokenizing '{self.file_path}' into '{self.model_cls.__name__}' objects...")

        connection = self.connection_cls()
        raw = connection.read(self.file_path)

        session = QdrantSession(get_qdrant_client())

        entries = [entry for parse_results in self.parser.parse(raw) for entry in parse_results]  # full in memory, could blow up
        stored_ids = self._remove_stale(session, {entry["id"] for entry in entries})

        fresh = [entry for entry in entries if entry["id"] not in stored_ids]
        task_logger.info(f"{len(entries) - len(fresh)} entries unchanged, {len(fresh)} to embed.")

        for i in range(0, len(fresh), self.BATCH_SIZE):
            batch = fresh[i:i + self.BATCH_SIZE]
            session.put(self._embed(batch), create_missing=True)

        task_logger.info(f"Stored {len(fresh)} '{self.model_cls.__name__}' objects.")

    def _embed(self, entries: list[Mapping[str, Any]]) -> list[QdrantModel]:
        vectors = ollama_embedder.embed_documents([entry[self.text_field] for entry in entries])
        return [
            self.model_cls(**entry, **{self.vector_field: vector})
            for entry, vector in zip(entries, vectors)
        ]

    def _remove_stale(self, session: QdrantSession, new_ids: set[str]) -> set[str]:
        """
        Compares the stored ids with the freshly parsed ones page by page, deleting stored ids the
        file no longer produces. Returns the ids that are stored and unchanged.
        """
        if not session.collection_exists(self.model_cls):
            return set()

        stored = set()
        for batch in session.get_batches(self.model_cls, fields=["id"]):
            batch_ids = {obj.id for obj in batch}
            stored |= batch_ids & new_ids
            session.refresh_collection(self.model_cls, batch_ids - new_ids)
        return stored
