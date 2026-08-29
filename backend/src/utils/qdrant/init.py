import argparse
import importlib
from pathlib import Path

from qdrant_client import QdrantClient
from src.utils.qdrant import QdrantBaseModel

__MODELS_FILE_NAME = "qmodels.py"


def initialize_qdrant_collections(url: str, strict_mode: bool = False):
    current_path = Path.cwd()
    for path in current_path.rglob(__MODELS_FILE_NAME):
        module_name = path.relative_to(current_path).with_suffix("").as_posix().replace("/", ".")
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            print(f"Failed to import module {module_name}: {e}")
            continue
            
    client = QdrantClient(url=url, strict_mode=strict_mode)
    QdrantBaseModel.create_all(client=client)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize Qdrant collections from qmodels.py files.")
    parser.add_argument("--url", required=True, help="Qdrant server URL")
    parser.add_argument("--strict-mode", action="store_true", help="Enable strict mode for Qdrant client")
    
    args = parser.parse_args()
    initialize_qdrant_collections(url=args.url, strict_mode=args.strict_mode)
