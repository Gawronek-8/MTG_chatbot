from loguru import logger

from src.connections import StorageConnection
from src.connections.qdrant import get_qdrant_client
from src.parsers import Parser
from src.utils.qdrant import QdrantBaseModel, QdrantSession


class Tokenizer[T: QdrantBaseModel]:
    """
    Reads a file from storage, parses it into entries and builds one model object per entry,
    ready to be stored in Qdrant.
    """

    def __init__(
        self,
        file_path: str,
        connection_cls: type[StorageConnection],
        parser: Parser,
        model_cls: type[T],
    ):
        self.file_path = file_path
        self.connection_cls = connection_cls
        self.parser = parser
        self.model_cls = model_cls

    def __call__(self) -> None:
        task_logger = logger.bind(task="Tokenizer")
        task_logger.info(f"Tokenizing '{self.file_path}' into '{self.model_cls.__name__}' objects...")

        connection = self.connection_cls()
        raw = connection.read(self.file_path)

        session = QdrantSession(get_qdrant_client())

        objects: list[T] = []
        for parse_results in self.parser.parse(raw):
            objs = [self.model_cls(**fields) for fields in parse_results]
            objects.append(*objs)

        task_logger.info(f"Built {len(objects)} '{self.model_cls.__name__}' objects.")

        # TODO: persist the objects once QdrantSession.put() is implemented.
