from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any

class StorageConnection(ABC):
    """Base interface for all storage adapters."""
    _instances: Dict[type, Any] = {}

    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__new__(cls)
        return cls._instances[cls]

    @abstractmethod
    def setup(self, **kwargs):
        """Setup the connection with configuration."""
        pass

    @abstractmethod
    def get_file_info(self, destination_path: str) -> Tuple[Optional[str], Optional[int]]:
        """Returns (md5_hash, file_size) if file exists in storage, else (None, None)"""
        pass

    @abstractmethod
    def upload(self, local_file_path: str, destination_path: str):
        """Uploads a local file to the storage destination."""
        pass
