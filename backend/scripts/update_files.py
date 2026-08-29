from pathlib import Path
from typing import List, Dict
from loguru import logger

from src.downloaders import HttpDownloader
from src.connections import StorageConnection, MinioConnection
from src.download_tasks import DownloadTask, CR_DOWNLOAD_TASK

class SyncManager:
    """Orchestrates the downloading of files and syncing them to storage connections."""
    def __init__(self, downloader: HttpDownloader):
        self.downloader = downloader

    def process_tasks(self, tasks: Dict[StorageConnection, List[DownloadTask]]):
        for storage, task_list in tasks.items():
            for task in task_list:
                self._process_single_task(storage, task)

    def _process_single_task(self, storage: StorageConnection, task: DownloadTask):
        task_logger = logger.bind(task="SyncManager")
        
        tmp_path, local_hash, local_size = self.downloader.download(task.url, task.headers)
        
        try:
            remote_hash, remote_size = storage.get_file_info(task.path)
            
            if remote_hash == local_hash and remote_size == local_size:
                task_logger.info(f"File '{task.path}' is already up-to-date. Skipping.")
                return
                
            task_logger.info(f"File '{task.path}' has changed (or is new). Updating storage...")
            
            storage.upload(tmp_path, task.path)
            task_logger.info(f"Successfully updated '{task.path}'.")
            
            task.on_update()
            
        finally:
            if tmp_path and (path := Path(tmp_path)).exists():
                path.unlink()

def update_files():
    
    downloader = HttpDownloader()
    minio_conn = MinioConnection()
    minio_conn.setup(bucket_name="mtg-data")
    
    manager = SyncManager(downloader)
    manager.process_tasks({minio_conn: [CR_DOWNLOAD_TASK]})

if __name__ == "__main__":
    update_files()
