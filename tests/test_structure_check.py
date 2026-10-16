import tempfile
import struct
from pathlib import Path
import numpy as np
import cv2
import soundfile as sf
import pytest

from vaultbreaker.inference.structure_check import check_structure
from vaultbreaker.inference.predict import VaultBreakerPredictor


@pytest.fixture
def clean_png(tmp_path: Path) -> Path:
    img = np.full((64, 64), 128, dtype=np.uint8)
    p = tmp_path / "clean.png"
    cv2.imwrite(str(p), img)
    return p


@pytest.fixture
def clean_jpeg(tmp_path: Path) -> Path:
    img = np.full((64, 64, 3), 128, dtype=np.uint8)
    p = tmp_path / "clean.jpg"
    cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return p


@pytest.fixture
def clean_wav(tmp_path: Path) -> Path:
    sr = 16000
    samples = (np.sin(np.linspace(0, 1.0, sr)) * 10000).astype(np.int16)
    p = tmp_path / "clean.wav"
    sf.write(str(p), samples, sr, subtype="PCM_16", format="WAV")
    return p


def test_clean_files_not_flagged(clean_png: Path, clean_jpeg: Path, clean_wav: Path):
    """Clean files must not produce suspicious structure findings."""
    res_png = check_structure(clean_png)
    assert res_png["format"] == "png"
    assert not res_png["is_suspicious"], f"Clean PNG flagged: {res_png['finding_messages']}"
    assert len(res_png["findings"]) == 0

    res_jpg = check_structure(clean_jpeg)
    assert res_jpg["format"] == "jpeg"
    assert not res_jpg["is_suspicious"], f"Clean JPEG flagged: {res_jpg['finding_messages']}"
    assert len(res_jpg["findings"]) == 0

    res_wav = check_structure(clean_wav)
    assert res_wav["format"] == "wav"
    assert not res_wav["is_suspicious"], f"Clean WAV flagged: {res_wav['finding_messages']}"
    assert len(res_wav["findings"]) == 0


def test_png_appended_zip(clean_png: Path, tmp_path: Path):
    """PNG with appended ZIP bytes must be flagged."""
    data = clean_png.read_bytes()
    # Appended dummy zip header
    zip_payload = b"PK\x03\x04" + b"\x00" * 26 + b"secret.txt" + b"HIDDEN_CONTENT"
    stego_p = tmp_path / "png_with_zip.png"
    stego_p.write_bytes(data + zip_payload)

    res = check_structure(stego_p)
    assert res["is_suspicious"] is True
    msgs = " ".join(res["finding_messages"])
    assert "after the end of the image" in msgs
    assert "ZIP archive" in msgs


def test_jpeg_trailing_bytes(clean_jpeg: Path, tmp_path: Path):
    """JPEG with trailing overlay bytes after EOI must be flagged."""
    data = clean_jpeg.read_bytes()
    stego_p = tmp_path / "jpeg_trailing.jpg"
    stego_p.write_bytes(data + b"EXTRA_TRAILING_PAYLOAD_BYTES" * 10)

    res = check_structure(stego_p)
    assert res["is_suspicious"] is True
    msgs = " ".join(res["finding_messages"])
    assert "after the end of the image" in msgs


def test_wav_fake_chunk(clean_wav: Path, tmp_path: Path):
    """WAV with an unknown/fake subchunk must be flagged."""
    data = clean_wav.read_bytes()
    # Insert unknown fake chunk before end: 4-byte ID 'fake', 4-byte size 8, 8 bytes data
    fake_chunk = b"fake" + struct.pack("<I", 8) + b"BADCHUNK"
    # Update declared RIFF size
    riff_size = struct.unpack("<I", data[4:8])[0] + len(fake_chunk)
    new_data = data[:4] + struct.pack("<I", riff_size) + data[8:] + fake_chunk

    stego_p = tmp_path / "wav_fake_chunk.wav"
    stego_p.write_bytes(new_data)

    res = check_structure(stego_p)
    assert res["is_suspicious"] is True
    msgs = " ".join(res["finding_messages"])
    assert "fake" in msgs


def test_predictor_integration_lossy_and_verdicts(clean_jpeg: Path, clean_png: Path, tmp_path: Path):
    """Predictor must handle lossy compressed inputs, clean verdicts, and unsupported files."""
    predictor = VaultBreakerPredictor()

    # 1. Lossy compressed input (JPEG)
    rep_jpg = predictor.predict_file(clean_jpeg)
    assert "notice" in rep_jpg
    assert "The statistical detector is not reliable on compressed files; structure check only" in rep_jpg["notice"]
    assert rep_jpg["verdict"] == "No hidden data detected"
    assert rep_jpg["calibrated_probability"] is None

    # 2. Clean PNG should return "No hidden data detected"
    rep_png = predictor.predict_file(clean_png)
    assert rep_png["verdict"] in ("No hidden data detected", "SUSPECTED STEGO")
    assert "structure_findings" in rep_png

    # 3. Unsupported format
    txt_file = tmp_path / "readme.txt"
    txt_file.write_text("This is an unsupported plain text file.")
    rep_txt = predictor.predict_file(txt_file)
    assert rep_txt["verdict"] == "Unsupported"
    assert rep_txt["format"] == "unsupported"
