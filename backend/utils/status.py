import os
import json
from threading import Lock
from typing import Dict, Any

from config import UPLOAD_DIR

_status_file = os.path.join(UPLOAD_DIR, "status.json")
_lock = Lock()

def _load_all() -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(_status_file):
        return {}
    try:
        with open(_status_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_all(data: Dict[str, Dict[str, Any]]) -> None:
    tmp = _status_file + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, _status_file)

def get_status(filename: str) -> Dict[str, Any]:
    with _lock:
        all_status = _load_all()
        return all_status.get(filename, {"uploaded": False, "processing": False, "indexed": False})

def set_status(filename: str, status: Dict[str, Any]) -> None:
    with _lock:
        all_status = _load_all()
        all_status[filename] = {**all_status.get(filename, {}), **status}
        _save_all(all_status)
