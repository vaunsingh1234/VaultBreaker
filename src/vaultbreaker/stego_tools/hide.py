"""
Steganographic Embedding Engine for VaultBreaker.
Supports:
- Images: LSB matching (default) or LSB replacement (option), saved as lossless PNG.
- Audio: 16-bit PCM WAV LSB embedding, saved as lossless WAV.
- Video: Lossless FFV1 MKV multi-frame LSB embedding.
Guarantees:
- Pre-flight capacity checks (throws CapacityError if too long).
- Automatic lossless conversion for JPEG, MP3, and MP4 inputs.
- Self-verification (immediately runs reveal code to confirm integrity).
- Destruction warnings for lossy re-encoding.
"""

import os
import shutil
import tempfile
import struct
from pathlib import Path
from typing import Union, Optional, Dict, Any, Tuple
import numpy as np
import cv2
import soundfile as sf
import imageio.v3 as iio
import imageio_ffmpeg

from vaultbreaker.utils.logger import get_logger
from vaultbreaker.stego_tools.codec import (
    encode_payload,
    get_scatter_indices,
    CapacityError,
    format_size
)

logger = get_logger("StegoHide")

LOSSY_DESTRUCTION_WARNING = (
    "WARNING: The steganographic message is embedded losslessly. "
    "Saving, uploading, or re-encoding this file as lossy JPEG, MP3, AAC, or H.264 "
    "will permanently destroy the hidden payload."
)


def get_container_capacity(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Inspects media container and returns capacity in bits and maximum payload bytes.
    Overhead allowance: 48 bytes (header + salt + nonce + CRC + tag).
    """
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    suffix = p.suffix.lower()
    conversion_notice = None

    # 1. Images
    if suffix in (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"):
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            from PIL import Image
            with Image.open(p) as pil_img:
                img = np.array(pil_img)

        if img is None:
            raise ValueError(f"Unable to decode image: {p}")

        if suffix in (".jpg", ".jpeg", ".webp"):
            conversion_notice = f"Lossy {suffix.upper()} container will be converted to lossless PNG to preserve message."

        total_bits = img.size
        # Overhead: 48 bytes * 8 = 384 bits
        max_bytes = max(0, (total_bits // 8) - 48)
        return {
            "media_type": "image",
            "detected_format": suffix[1:],
            "target_format": "png",
            "shape": img.shape,
            "total_bits": total_bits,
            "max_bytes": max_bytes,
            "conversion_notice": conversion_notice
        }

    # 2. Audio
    elif suffix in (".wav", ".mp3", ".flac", ".m4a", ".ogg"):
        if suffix in (".mp3", ".m4a", ".ogg"):
            conversion_notice = f"Lossy {suffix.upper()} audio will be converted to 16 kHz 16-bit mono WAV to preserve message."

        try:
            samples, sr = sf.read(str(p))
            n_samples = len(samples) * (samples.shape[1] if samples.ndim > 1 else 1)
        except Exception:
            # Fallback to ffmpeg decode probe
            probe = imageio_ffmpeg.read_frames(str(p))
            meta = probe.__next__()
            dur = meta.get("duration", 3.0)
            n_samples = int(16000 * dur)

        total_bits = n_samples
        max_bytes = max(0, (total_bits // 8) - 48)
        return {
            "media_type": "audio",
            "detected_format": suffix[1:],
            "target_format": "wav",
            "shape": (n_samples,),
            "total_bits": total_bits,
            "max_bytes": max_bytes,
            "conversion_notice": conversion_notice
        }

    # 3. Video
    elif suffix in (".mp4", ".mkv", ".avi", ".mov", ".webm"):
        if suffix in (".mp4", ".mov", ".webm"):
            conversion_notice = f"Container {suffix.upper()} will be converted to lossless FFV1 MKV to preserve message."

        cap = cv2.VideoCapture(str(p))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if n_frames <= 0 or w <= 0 or h <= 0:
            n_frames, w, h = 30, 128, 128

        total_bits = n_frames * h * w * 3
        max_bytes = max(0, (total_bits // 8) - 48)
        return {
            "media_type": "video",
            "detected_format": suffix[1:],
            "target_format": "mkv",
            "shape": (n_frames, h, w, 3),
            "total_bits": total_bits,
            "max_bytes": max_bytes,
            "conversion_notice": conversion_notice
        }

    else:
        raise ValueError(f"Unsupported media file type: {suffix}")


def hide_message(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    message: Union[str, bytes],
    password: Optional[str] = None,
    method: str = "lsb_matching"  # "lsb_matching" or "lsb_replacement"
) -> Dict[str, Any]:
    """
    Embeds message into media container, guarantees lossless output,
    and runs self-verification.
    """
    inp = Path(input_path)
    outp = Path(output_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    # 1. Encode payload into bitstream
    stream = encode_payload(message, password=password)
    stream_bits = np.unpackbits(np.frombuffer(stream, dtype=np.uint8))
    n_bits = len(stream_bits)

    # 2. Check capacity
    cap_info = get_container_capacity(inp)
    total_bits = cap_info["total_bits"]
    if n_bits > total_bits:
        raise CapacityError(
            f"Message capacity exceeded: requires {n_bits} bits ({format_size(len(stream))}), "
            f"but container only provides capacity for {total_bits} bits ({format_size(cap_info['max_bytes'])} max payload). "
            f"Please shorten message or choose a larger media file."
        )

    # 3. Derive embedding indices
    # Header is stored in first hdr_bits sequentially (41 bytes = 328 bits if encrypted, 13 bytes = 104 bits if unencrypted)
    # Body/ciphertext bits are scattered across the remaining container bits
    hdr_bytes = 41 if password else 13
    hdr_bits = hdr_bytes * 8
    payload_bits = n_bits - hdr_bits

    if password:
        import hashlib
        h = hashlib.sha256(password.encode("utf-8") + b"_scatter_seed").digest()
        seed = int.from_bytes(h[:4], "big")
        rng = np.random.RandomState(seed)
        body_indices = hdr_bits + rng.permutation(total_bits - hdr_bits)[:payload_bits]
    else:
        body_indices = np.arange(hdr_bits, hdr_bits + payload_bits)

    indices = np.concatenate([np.arange(hdr_bits), body_indices])
    media_type = cap_info["media_type"]

    # 4. Perform embedding by format
    if media_type == "image":
        # Always output as PNG
        if outp.suffix.lower() != ".png":
            outp = outp.with_suffix(".png")

        img = cv2.imread(str(inp), cv2.IMREAD_UNCHANGED)
        if img is None:
            from PIL import Image
            with Image.open(inp) as pil_img:
                img = np.array(pil_img)

        orig_shape = img.shape
        flat = img.flatten()

        if method == "lsb_replacement":
            flat[indices] = (flat[indices] & 0xFE) | stream_bits
        else: # lsb_matching
            curr_bits = flat[indices] & 1
            diff_mask = curr_bits != stream_bits
            diff_indices = indices[diff_mask]
            if len(diff_indices) > 0:
                rng = np.random.RandomState(42)
                # Random +1 or -1
                adjust = rng.choice([-1, 1], size=len(diff_indices))
                # Adjust boundaries: if 0, must be +1; if 255, must be -1
                vals = flat[diff_indices].astype(np.int16)
                adjust[vals == 0] = 1
                adjust[vals == 255] = -1
                flat[diff_indices] = np.clip(vals + adjust, 0, 255).astype(np.uint8)

        stego_img = flat.reshape(orig_shape)
        cv2.imwrite(str(outp), stego_img, [cv2.IMWRITE_PNG_COMPRESSION, 4])

    elif media_type == "audio":
        # Always output as WAV
        if outp.suffix.lower() != ".wav":
            outp = outp.with_suffix(".wav")

        # Convert to 16 kHz 16-bit PCM WAV if not already
        temp_wav = None
        if inp.suffix.lower() != ".wav":
            temp_wav = outp.parent / f"_temp_conv_{os.getpid()}.wav"
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            cmd = [
                ffmpeg_exe, "-y", "-v", "error",
                "-i", str(inp),
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                "-f", "wav", str(temp_wav)
            ]
            import subprocess
            subprocess.run(cmd, check=True)
            read_path = temp_wav
        else:
            read_path = inp

        samples, sr = sf.read(str(read_path), dtype="int16")
        if temp_wav and temp_wav.exists():
            temp_wav.unlink()

        orig_shape = samples.shape
        flat = samples.flatten()

        if method == "lsb_replacement":
            flat[indices] = np.bitwise_and(flat[indices], ~1) | stream_bits.astype(np.int16)
        else: # lsb_matching
            curr_bits = flat[indices] & 1
            diff_mask = curr_bits != stream_bits
            diff_indices = indices[diff_mask]
            if len(diff_indices) > 0:
                rng = np.random.RandomState(42)
                adjust = rng.choice([-1, 1], size=len(diff_indices))
                vals = flat[diff_indices].astype(np.int32)
                adjust[vals <= -32768] = 1
                adjust[vals >= 32767] = -1
                flat[diff_indices] = np.clip(vals + adjust, -32768, 32767).astype(np.int16)

        stego_samples = flat.reshape(orig_shape)
        sf.write(str(outp), stego_samples, sr, subtype="PCM_16", format="WAV")

    elif media_type == "video":
        # Always output as FFV1 MKV
        if outp.suffix.lower() != ".mkv":
            outp = outp.with_suffix(".mkv")

        cap = cv2.VideoCapture(str(inp))
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        if not frames:
            raise ValueError(f"Could not read frames from video: {inp}")

        stack = np.stack(frames, axis=0) # (T, H, W, 3)
        orig_shape = stack.shape
        flat = stack.flatten()

        if method == "lsb_replacement":
            flat[indices] = (flat[indices] & 0xFE) | stream_bits
        else:
            curr_bits = flat[indices] & 1
            diff_mask = curr_bits != stream_bits
            diff_indices = indices[diff_mask]
            if len(diff_indices) > 0:
                rng = np.random.RandomState(42)
                adjust = rng.choice([-1, 1], size=len(diff_indices))
                vals = flat[diff_indices].astype(np.int16)
                adjust[vals == 0] = 1
                adjust[vals == 255] = -1
                flat[diff_indices] = np.clip(vals + adjust, 0, 255).astype(np.uint8)

        stego_stack = flat.reshape(orig_shape)
        iio.imwrite(
            str(outp),
            stego_stack,
            extension=".mkv",
            codec="ffv1",
            fps=fps,
            pixelformat="rgb24"
        )

    # 5. Immediate self-verification
    from vaultbreaker.stego_tools.reveal import reveal_message
    res = reveal_message(outp, password=password)
    if not res.success:
        raise RuntimeError(f"Self-verification failed after embedding: {res.error}")

    raw_expected = message.encode("utf-8") if isinstance(message, str) else message
    if res.data != raw_expected:
        raise RuntimeError("Self-verification failed: extracted payload bytes do not match original message.")

    logger.info(f"Stego embedding verified successfully in {outp.name} ({len(stream_bits)} bits embedded)")

    return {
        "output_path": str(outp),
        "verified": True,
        "format": outp.suffix[1:],
        "media_type": media_type,
        "payload_bytes": len(raw_expected),
        "total_bits_embedded": n_bits,
        "capacity_bytes": cap_info["max_bytes"],
        "capacity_used_ratio": n_bits / total_bits,
        "warning": LOSSY_DESTRUCTION_WARNING,
        "conversion_notice": cap_info.get("conversion_notice")
    }
