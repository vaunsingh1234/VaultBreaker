import tempfile
import html
from pathlib import Path
import numpy as np
import cv2
import soundfile as sf
import imageio.v3 as iio
import pytest

from vaultbreaker.stego_tools.codec import (
    encode_payload,
    decode_payload,
    CapacityError,
    ExtractionResult
)
from vaultbreaker.stego_tools.hide import (
    hide_message,
    get_container_capacity
)
from vaultbreaker.stego_tools.reveal import (
    reveal_message
)


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    img = np.full((128, 128, 3), 120, dtype=np.uint8)
    # Add varying texture
    img += np.random.RandomState(42).randint(-15, 15, size=img.shape).astype(np.uint8)
    p = tmp_path / "cover.png"
    cv2.imwrite(str(p), img)
    return p


@pytest.fixture
def sample_audio(tmp_path: Path) -> Path:
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    samples = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    p = tmp_path / "cover.wav"
    sf.write(str(p), samples, sr, subtype="PCM_16", format="WAV")
    return p


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    frames = [np.full((64, 64, 3), (i * 10) % 256, dtype=np.uint8) for i in range(15)]
    stack = np.stack(frames, axis=0)
    p = tmp_path / "cover.mkv"
    iio.imwrite(str(p), stack, extension=".mkv", codec="ffv1", fps=15, pixelformat="rgb24")
    return p


def test_image_roundtrip_with_and_without_password(sample_image: Path, tmp_path: Path):
    """Test image hiding and revealing both with and without password."""
    secret = "TopSecretForensicMessage_2026!"

    # 1. Without password (plain VBK1)
    out_plain = tmp_path / "stego_plain.png"
    res_hide = hide_message(sample_image, out_plain, secret, password=None)
    assert res_hide["verified"] is True
    assert out_plain.exists()

    res_reveal = reveal_message(out_plain, password=None)
    assert res_reveal.success is True
    assert res_reveal.text == secret

    # 2. With password (scrypt + AES-GCM + scatter)
    out_enc = tmp_path / "stego_enc.png"
    res_hide_enc = hide_message(sample_image, out_enc, secret, password="StrongMasterPassword#123")
    assert res_hide_enc["verified"] is True

    res_reveal_enc = reveal_message(out_enc, password="StrongMasterPassword#123")
    assert res_reveal_enc.success is True
    assert res_reveal_enc.text == secret


def test_audio_roundtrip_with_password(sample_audio: Path, tmp_path: Path):
    """Test audio hiding and revealing with password."""
    secret = "AudioWatermarkPayload"
    out_wav = tmp_path / "stego_audio.wav"
    res_hide = hide_message(sample_audio, out_wav, secret, password="AudioKey$99")
    assert res_hide["verified"] is True

    res_reveal = reveal_message(out_wav, password="AudioKey$99")
    assert res_reveal.success is True
    assert res_reveal.text == secret


def test_video_roundtrip(sample_video: Path, tmp_path: Path):
    """Test video hiding and revealing."""
    secret = "VideoFrameStegoVerification"
    out_vid = tmp_path / "stego_video.mkv"
    res_hide = hide_message(sample_video, out_vid, secret, password="VideoPassword!44")
    assert res_hide["verified"] is True

    res_reveal = reveal_message(out_vid, password="VideoPassword!44")
    assert res_reveal.success is True
    assert res_reveal.text == secret


def test_wrong_password_fails_cleanly(sample_image: Path, tmp_path: Path):
    """Providing a wrong password must fail cleanly without returning garbage text."""
    secret = "ConfidentialClassifiedData"
    out_p = tmp_path / "stego_pass.png"
    hide_message(sample_image, out_p, secret, password="CorrectPassword_123")

    # Reveal with wrong password
    res_wrong = reveal_message(out_p, password="TotallyWrongPassword")
    assert res_wrong.success is False
    assert res_wrong.text is None
    assert "incorrect password" in res_wrong.error.lower() or "decryption failed" in res_wrong.error.lower()


def test_capacity_overflow(sample_image: Path, tmp_path: Path):
    """Attempting to embed a message exceeding container capacity must raise CapacityError."""
    cap = get_container_capacity(sample_image)
    max_bytes = cap["max_bytes"]

    # Massive payload exceeding container size
    oversized = "A" * (max_bytes + 5000)
    out_p = tmp_path / "overflow.png"

    with pytest.raises(CapacityError) as exc_info:
        hide_message(sample_image, out_p, oversized, password=None)

    assert "capacity" in str(exc_info.value).lower()


def test_malformed_and_truncated_files(tmp_path: Path):
    """Truncated streams and malformed VBK1 headers must fail cleanly."""
    truncated = b"VBK1\x01\x00\x00"
    res = decode_payload(truncated, password="test")
    assert res.success is False
    assert "truncated" in res.error.lower()

    corrupt = b"VBK1\x00" + b"\xff" * 30
    res_corrupt = decode_payload(corrupt, password=None)
    assert res_corrupt.success is False


def test_html_escaped_output():
    """ExtractionResult must safely HTML-escape malicious script tags."""
    malicious_text = "<script>alert('xss attack!')</script> <b>bold text</b> & 'quoted'"
    res = ExtractionResult(
        success=True,
        text=malicious_text,
        is_binary=False
    )
    safe_display = res.get_safe_display_text()
    assert "<script>" not in safe_display
    assert "&lt;script&gt;" in safe_display
    assert "&amp;" in safe_display
