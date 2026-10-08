import os
import urllib.request
from pathlib import Path
from typing import Optional
from vaultbreaker.utils.logger import get_logger

logger = get_logger("Downloader")

def download_file(url: str, dest_path: Path, timeout: int = 10) -> bool:
    """Download a file with timeout; returns True on success, False on failure."""
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "VaultBreaker/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as response, open(dest_path, "wb") as out_file:
            out_file.write(response.read())
        logger.info(f"Successfully downloaded {dest_path.name}")
        return True
    except Exception as e:
        logger.warning(f"Download from {url} failed: {e}. Falling back to procedural generation.")
        if dest_path.exists():
            dest_path.unlink()
        return False
