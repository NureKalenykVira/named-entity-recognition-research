import json
import random
from pathlib import Path
from typing import Any
import numpy as np

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def save_json(data: Any, path: Path, indent: int = 2) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)