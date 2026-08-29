from dataclasses import dataclass
from typing import Callable, Dict
from scripts.tokenize import tokenize

@dataclass
class DownloadTask:
    path: str
    url: str
    headers: Dict[str, str]
    on_update: Callable[[], None]

COMPREHENSIVE_RULES_FILE = "cr.json"

CR_DOWNLOAD_TASK = DownloadTask(
    path=COMPREHENSIVE_RULES_FILE,
    url="https://api.academyruins.com/cr",
    headers={
        "User-Agent": "MTGChatbot/1.0",
        "Accept": "application/json;q=0.9,*/*;q=0.8"
    },
    on_update=tokenize
)

ALL_TASKS = [
    CR_DOWNLOAD_TASK,
]
