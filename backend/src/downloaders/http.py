import hashlib
import tempfile
import requests
from pathlib import Path
from typing import Tuple, Dict
from loguru import logger

class HttpDownloader:
    """Handles downloading files over HTTP(s) and calculating their MD5 hash iteratively."""
    
    def download(self, url: str, headers: Dict[str, str]) -> Tuple[str, str, int]:
        if headers is None:
            raise ValueError("headers must be provided (can be an empty dict)")
            
        task_logger = logger.bind(task="HttpDownloader")
        task_logger.info(f"Starting iterative download from {url}...")
        
        tmp_file_path = None
        md5_hash = hashlib.md5()
        local_size = 0
        
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file_path = tmp_file.name
                
                with requests.get(url, stream=True, headers=headers) as r:
                    r.raise_for_status()
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            tmp_file.write(chunk)
                            md5_hash.update(chunk)
                            local_size += len(chunk)
                            
            local_md5 = md5_hash.hexdigest()
            task_logger.info(f"Download complete. Size: {local_size} bytes, MD5: {local_md5}")
            
        except Exception as e:
            if tmp_file_path and (path := Path(tmp_file_path)).exists():
                path.unlink()
            raise e
        return tmp_file_path, local_md5, local_size
