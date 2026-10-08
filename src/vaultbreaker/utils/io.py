import hashlib
import os
from pathlib import Path
from typing import Union

def compute_sha256(filepath: Union[str, Path]) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def ensure_dir(dirpath: Union[str, Path]) -> Path:
    """Ensure directory exists."""
    p = Path(dirpath)
    p.mkdir(parents=True, exist_ok=True)
    return p
