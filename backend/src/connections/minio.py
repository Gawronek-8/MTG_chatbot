from typing import Optional, Tuple
from minio import Minio
from minio.error import S3Error
from pydantic import BaseModel
from src.config import settings
from .base import StorageConnection

class MinioConfig(BaseModel):
    bucket_name: str = "mtg-data"

class MinioConnection(StorageConnection):
    def __init__(self):
        if not hasattr(self, "client"):
            self.client = None
            self.bucket_name = None

    def setup(self, **kwargs):
        config = MinioConfig(**kwargs)
        self.bucket_name = config.bucket_name
        
        endpoint = settings.minio_url.replace("http://", "").replace("https://", "")
        self.client = Minio(
            endpoint,
            access_key=settings.minio_root_user,
            secret_key=settings.minio_root_password,
            secure=settings.minio_url.startswith("https")
        )
        
        if not self.client.bucket_exists(self.bucket_name):
            self.client.make_bucket(self.bucket_name)

    def get_file_info(self, destination_path: str) -> Tuple[Optional[str], Optional[int]]:
        if not self.client:
            raise RuntimeError("MinioConnection not initialized. Call setup() first.")
        try:
            stat = self.client.stat_object(self.bucket_name, destination_path)
            return stat.etag.strip('"'), stat.size
        except S3Error as e:
            if e.code == "NoSuchKey":
                return None, None
            raise

    def upload(self, local_file_path: str, destination_path: str):
        if not self.client:
            raise RuntimeError("MinioConnection not initialized. Call setup() first.")
        self.client.fput_object(self.bucket_name, destination_path, local_file_path)
