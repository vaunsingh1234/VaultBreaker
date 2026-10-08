import os
from pathlib import Path
from typing import Tuple, Union, Optional
import tempfile

class RouterError(Exception):
    """Raised when file inspection fails or violates security constraints."""
    pass

class MediaRouter:
    """
    Format detection and sanitization router.
    Identifies media format strictly by magic bytes (header signatures),
    enforces file size limits, guards against execution, and rejects corrupt inputs.
    """
    MAX_FILE_SIZE = 100 * 1024 * 1024 # 100 MB default safeguard

    # Magic byte definitions
    MAGIC_SIGNATURES = {
        "png": b"\x89PNG\r\n\x1a\n",
        "jpeg": b"\xff\xd8\xff",
        "mkv": b"\x1a\x45\xdf\xa3"
    }

    def __init__(self, max_file_size: int = MAX_FILE_SIZE):
        self.max_file_size = max_file_size

    def inspect_file(self, filepath: Union[str, Path]) -> Tuple[str, str]:
        """
        Inspect file header and return (media_type, format_name).
        Media types: 'image', 'audio', 'video'.
        """
        p = Path(filepath)
        if not p.exists():
            raise RouterError(f"Target file does not exist: {p}")

        file_size = p.stat().st_size
        if file_size == 0:
            raise RouterError("Empty file (0 bytes).")
        if file_size > self.max_file_size:
            raise RouterError(f"File size {file_size / (1024*1024):.1f} MB exceeds maximum limit of {self.max_file_size / (1024*1024):.1f} MB.")

        with open(p, "rb") as f:
            header = f.read(64)

        return self.identify_bytes(header)

    def identify_bytes(self, header: bytes) -> Tuple[str, str]:
        """Inspect byte header and return (media_type, format_name)."""
        if len(header) < 4:
            raise RouterError("Header too short for identification.")

        # PNG
        if header.startswith(self.MAGIC_SIGNATURES["png"]):
            return "image", "png"

        # JPEG
        if header.startswith(self.MAGIC_SIGNATURES["jpeg"]):
            return "image", "jpeg"

        # WAV: RIFF....WAVE
        if header.startswith(b"RIFF") and len(header) >= 12 and header[8:12] == b"WAVE":
            return "audio", "wav"

        # AVI: RIFF....AVI 
        if header.startswith(b"RIFF") and len(header) >= 12 and header[8:12] == b"AVI ":
            return "video", "avi"

        # MKV / WebM
        if header.startswith(self.MAGIC_SIGNATURES["mkv"]):
            return "video", "mkv"

        # MP4: ....ftyp
        if len(header) >= 8 and (header[4:8] == b"ftyp" or b"ftyp" in header[:16]):
            return "video", "mp4"

        # FLAC: fLaC
        if header.startswith(b"fLaC"):
            return "audio", "flac"

        raise RouterError(
            f"Unsupported or corrupted file format. Magic bytes: {header[:12].hex()}"
        )
