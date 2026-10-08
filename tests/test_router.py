import pytest
import tempfile
from pathlib import Path
from vaultbreaker.inference.router import MediaRouter, RouterError
from vaultbreaker.inference.predict import VaultBreakerPredictor

def test_router_magic_bytes_detection():
    router = MediaRouter()

    # PNG
    assert router.identify_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR") == ("image", "png")

    # JPEG
    assert router.identify_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF") == ("image", "jpeg")

    # WAV
    assert router.identify_bytes(b"RIFF\x24\x08\x00\x00WAVEfmt ") == ("audio", "wav")

    # AVI
    assert router.identify_bytes(b"RIFF\x24\x08\x00\x00AVI LIST") == ("video", "avi")

    # MKV
    assert router.identify_bytes(b"\x1a\x45\xdf\xa3\x9f\x42\x86\x81") == ("video", "mkv")

    # MP4
    assert router.identify_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00") == ("video", "mp4")

def test_router_rejection_unsupported():
    router = MediaRouter()
    with pytest.raises(RouterError):
        # Random binary
        router.identify_bytes(b"MZ\x90\x00\x03\x00\x00\x00")

def test_router_empty_file():
    router = MediaRouter()
    with tempfile.NamedTemporaryFile() as tmp:
        with pytest.raises(RouterError):
            router.inspect_file(tmp.name)

def test_router_oversized_file():
    router = MediaRouter(max_file_size=100) # 100 bytes max
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(b"0" * 200)
        tmp.flush()
        with pytest.raises(RouterError):
            router.inspect_file(tmp.name)
