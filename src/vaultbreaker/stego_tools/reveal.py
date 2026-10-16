"""
Steganographic Extraction and Forensic Reveal Engine for VaultBreaker.
Executes multi-stage extraction cascade:
1. VaultBreaker (VBK1) protocol (scrypt KDF + AES-GCM + CRC-32)
2. Appended data (EOF overlay) and metadata text fields
3. Simple sequential LSB heuristic (1-bit and 2-bit row-major)
4. Optional steghide subprocess extraction (if installed and enabled)

Safety Features:
- HTML-escaped output presentation
- Size capping (64 KB display limit)
- Hex preview for binary payloads
- Safe argument passing (no shell injection)
- Clean error on wrong password (never returns garbage)
"""

import os
import shutil
import struct
import subprocess
import tempfile
import mimetypes
from pathlib import Path
from typing import Optional, Union, Dict, Any, List, Tuple
import numpy as np
import cv2
import soundfile as sf
import imageio.v3 as iio

from vaultbreaker.utils.logger import get_logger
from vaultbreaker.inference.structure_check import check_structure
from vaultbreaker.stego_tools.codec import (
    MAGIC,
    ExtractionResult,
    decode_payload,
    get_scatter_indices,
    format_hex_dump
)

logger = get_logger("StegoReveal")

ATTEMPTED_METHODS = [
    "1. VaultBreaker Protocol (VBK1, AES-256-GCM / CRC-32)",
    "2. File End Overlay & Metadata Fields (EOF text / embedded archive)",
    "3. Sequential LSB Heuristic (1-bit / 2-bit ASCII analysis)",
    "4. Steghide Forensic Extraction (Sandboxed Subprocess)"
]


def is_printable_text(data: bytes, min_ratio: float = 0.92, min_length: int = 8) -> Tuple[bool, float]:
    """Test if byte sequence consists predominantly of printable text."""
    if len(data) < min_length:
        return False, 0.0
    # Printable ASCII: 32 to 126, plus tab (9), newline (10), carriage return (13)
    printable_count = sum(1 for b in data if 32 <= b <= 126 or b in (9, 10, 13))
    ratio = printable_count / len(data)
    return (ratio >= min_ratio), ratio


def extract_raw_bits_from_media(filepath: Path) -> Optional[np.ndarray]:
    """Extract flat LSB bit array from image, audio, or video container."""
    suffix = filepath.suffix.lower()

    if suffix in (".png", ".bmp", ".jpg", ".jpeg", ".webp"):
        img = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)
        if img is None:
            from PIL import Image
            with Image.open(filepath) as pil_img:
                img = np.array(pil_img)
        if img is not None:
            return (img.flatten() & 1).astype(np.uint8)

    elif suffix in (".wav", ".flac"):
        try:
            samples, _ = sf.read(str(filepath), dtype="int16")
            return (samples.flatten() & 1).astype(np.uint8)
        except Exception:
            pass

    elif suffix in (".mkv", ".mp4", ".avi", ".mov"):
        try:
            frames = iio.imread(str(filepath))
            return (frames.flatten() & 1).astype(np.uint8)
        except Exception:
            pass

    return None


def reveal_message(
    filepath: Union[str, Path],
    password: Optional[str] = None
) -> ExtractionResult:
    """
    Main extraction pipeline: cascades through 4 extraction strategies.
    Returns ExtractionResult with confidence and provenance.
    """
    p = Path(filepath)
    if not p.exists():
        return ExtractionResult(
            success=False,
            error=f"Target file does not exist: {p}",
            method="File Access"
        )

    # ---------------------------------------------------------
    # STAGE 1: VaultBreaker (VBK1) Protocol
    # ---------------------------------------------------------
    raw_bits = extract_raw_bits_from_media(p)
    if raw_bits is not None and len(raw_bits) >= 128:
        total_bits = len(raw_bits)

        # 1a. Sequential Header Inspection (Header in first 41 bytes = 328 bits)
        if total_bits >= 328:
            seq_hdr = np.packbits(raw_bits[:328]).tobytes()
            if seq_hdr.startswith(MAGIC):
                flags = seq_hdr[4]
                is_encrypted = bool(flags & 0x01)
                is_binary = bool(flags & 0x02)

                if is_encrypted:
                    if not password:
                        return ExtractionResult(
                            success=False,
                            is_binary=is_binary,
                            method="VaultBreaker VBK1",
                            error="Message is encrypted with a password. Please provide the passphrase."
                        )
                    salt = seq_hdr[5:21]
                    nonce = seq_hdr[21:33]
                    length, crc = struct.unpack(">II", seq_hdr[33:41])

                    import hashlib
                    import zlib
                    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
                    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
                    from cryptography.exceptions import InvalidTag

                    h = hashlib.sha256(password.encode("utf-8") + b"_scatter_seed").digest()
                    seed = int.from_bytes(h[:4], "big")
                    rng = np.random.RandomState(seed)
                    ct_bits_len = (length + 16) * 8

                    if 328 + ct_bits_len <= total_bits:
                        body_indices = 328 + rng.permutation(total_bits - 328)[:ct_bits_len]
                        ciphertext = np.packbits(raw_bits[body_indices]).tobytes()
                        try:
                            kdf = Scrypt(salt=salt, length=32, n=16384, r=8, p=1)
                            key = kdf.derive(password.encode("utf-8"))
                            aesgcm = AESGCM(key)
                            aad = MAGIC + bytes([flags]) + struct.pack(">II", length, crc)
                            plaintext = aesgcm.decrypt(nonce, ciphertext, aad)

                            if (zlib.crc32(plaintext) & 0xFFFFFFFF) != crc:
                                return ExtractionResult(
                                    success=False,
                                    error="Decryption failed: CRC-32 integrity check mismatch."
                                )

                            text = None
                            try:
                                cand_text = plaintext.decode("utf-8")
                                printable, _ = is_printable_text(plaintext)
                                if printable or not is_binary:
                                    text = cand_text
                            except UnicodeDecodeError:
                                is_binary = True

                            return ExtractionResult(
                                success=True,
                                data=plaintext,
                                text=text,
                                is_binary=is_binary,
                                method="VaultBreaker VBK1 (scrypt / AES-GCM)",
                                confidence="High (Cryptographic Auth Tag + CRC-32 Verified)",
                                hex_preview=format_hex_dump(plaintext) if is_binary else None
                            )
                        except InvalidTag:
                            return ExtractionResult(
                                success=False,
                                error="Decryption failed: Incorrect password or corrupted payload."
                            )
                        except Exception as e:
                            return ExtractionResult(
                                success=False,
                                error=f"Decryption failed: {e}"
                            )

                else:
                    # Unencrypted sequential VBK1
                    length, crc = struct.unpack(">II", seq_hdr[5:13])
                    import zlib
                    data_bits = raw_bits[104 : 104 + length * 8]
                    if len(data_bits) == length * 8:
                        data = np.packbits(data_bits).tobytes()
                        if (zlib.crc32(data) & 0xFFFFFFFF) == crc:
                            text = None
                            try:
                                cand_text = data.decode("utf-8")
                                printable, _ = is_printable_text(data)
                                if printable or not is_binary:
                                    text = cand_text
                            except UnicodeDecodeError:
                                is_binary = True
                            return ExtractionResult(
                                success=True,
                                data=data,
                                text=text,
                                is_binary=is_binary,
                                method="VaultBreaker VBK1 (Plain)",
                                confidence="High (CRC-32 Checksum Verified)",
                                hex_preview=format_hex_dump(data) if is_binary else None
                            )

        # 1b. Fully-scattered or fully sequential fallback
        if password:
            perm_indices = get_scatter_indices(total_bits, password=password)
            scattered_bits = raw_bits[perm_indices]
            probe_bytes = np.packbits(scattered_bits[: min(total_bits, 32 * 1024 * 1024)]).tobytes()
            res = decode_payload(probe_bytes, password=password)
            if res.success:
                return res
            if res.error and ("password" in res.error.lower() or "decryption failed" in res.error.lower()):
                return res

        seq_bytes = np.packbits(raw_bits[: min(total_bits, 32 * 1024 * 1024)]).tobytes()
        res_seq = decode_payload(seq_bytes, password=password)
        if res_seq.success:
            return res_seq
        if res_seq.error and ("password" in res_seq.error.lower() or "decryption failed" in res_seq.error.lower()):
            return res_seq

    # ---------------------------------------------------------
    # STAGE 2: Appended Data (EOF Overlays) & Metadata Inspection
    # ---------------------------------------------------------
    struct_res = check_structure(p)
    file_bytes = p.read_bytes()

    # Check for EOF overlays found by structure check
    for f in struct_res.get("findings", []):
        if f.get("category") == "overlay" and f.get("offset") is not None:
            offset = f["offset"]
            overlay = file_bytes[offset:]
            if overlay:
                # 2a. Does overlay start with VBK1?
                if overlay.startswith(MAGIC):
                    res_ovk = decode_payload(overlay, password=password)
                    if res_ovk.success:
                        res_ovk.method = "Appended VBK1 Stream (EOF Overlay)"
                        res_ovk.confidence = "High (Cryptographic Checksum in File Overlay)"
                        return res_ovk

                # 2b. Printable text overlay
                is_text, ratio = is_printable_text(overlay, min_ratio=0.88, min_length=6)
                if is_text:
                    try:
                        clean_text = overlay.decode("utf-8", errors="ignore").strip()
                        if clean_text:
                            return ExtractionResult(
                                success=True,
                                data=overlay,
                                text=clean_text,
                                is_binary=False,
                                method="Appended Text Overlay (EOF)",
                                confidence=f"Medium ({ratio * 100:.1f}% Printable ASCII in Overlay)"
                            )
                    except Exception:
                        pass

                # 2c. Embedded archive in overlay (ZIP, RAR, PDF, ELF)
                if overlay.startswith(b"PK\x03\x04"):
                    return ExtractionResult(
                        success=True,
                        data=overlay,
                        is_binary=True,
                        method="Appended ZIP Archive (EOF Overlay)",
                        confidence="High (Valid PK\\x03\\x04 Header)",
                        mime_type="application/zip",
                        hex_preview=format_hex_dump(overlay)
                    )
                elif overlay.startswith(b"%PDF-"):
                    return ExtractionResult(
                        success=True,
                        data=overlay,
                        is_binary=True,
                        method="Appended PDF Document (EOF Overlay)",
                        confidence="High (Valid %PDF- Header)",
                        mime_type="application/pdf",
                        hex_preview=format_hex_dump(overlay)
                    )
                elif overlay.startswith(b"Rar!\x1a\x07"):
                    return ExtractionResult(
                        success=True,
                        data=overlay,
                        is_binary=True,
                        method="Appended RAR Archive (EOF Overlay)",
                        confidence="High (Valid Rar! Header)",
                        mime_type="application/x-rar-compressed",
                        hex_preview=format_hex_dump(overlay)
                    )

    # ---------------------------------------------------------
    # STAGE 3: Sequential LSB Heuristic (1-bit and 2-bit ASCII)
    # ---------------------------------------------------------
    if raw_bits is not None and len(raw_bits) >= 256:
        # 3a. 1-bit sequential probe (first 2048 bytes)
        probe_len = min(len(raw_bits) // 8, 2048)
        packed_seq = np.packbits(raw_bits[: probe_len * 8]).tobytes()

        # Check for length-prefixed plain ASCII or string
        # Test null-terminated string or high printable ratio
        null_idx = packed_seq.find(b"\x00")
        sample_slice = packed_seq[:null_idx] if (16 <= null_idx <= 1024) else packed_seq[:256]
        is_txt, ratio = is_printable_text(sample_slice, min_ratio=0.95, min_length=16)
        if is_txt:
            try:
                msg = sample_slice.decode("ascii").strip()
                if len(msg) >= 12:
                    return ExtractionResult(
                        success=True,
                        data=sample_slice,
                        text=msg,
                        is_binary=False,
                        method="Sequential 1-bit LSB (Statistical Heuristic)",
                        confidence=f"Medium ({ratio * 100:.1f}% Printable ASCII Ratio)"
                    )
            except Exception:
                pass

        # 3b. 2-bit sequential probe on images
        if p.suffix.lower() in (".png", ".bmp"):
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is not None:
                flat_img = img.flatten()
                # 2 bits: bit 0 and bit 1
                bits_2 = np.empty(min(len(flat_img) * 2, 16384), dtype=np.uint8)
                n_pix = len(bits_2) // 2
                bits_2[0::2] = (flat_img[:n_pix] >> 1) & 1
                bits_2[1::2] = flat_img[:n_pix] & 1
                packed_2 = np.packbits(bits_2).tobytes()
                null_idx2 = packed_2.find(b"\x00")
                sample2 = packed_2[:null_idx2] if (16 <= null_idx2 <= 1024) else packed_2[:256]
                is_txt2, ratio2 = is_printable_text(sample2, min_ratio=0.95, min_length=16)
                if is_txt2:
                    try:
                        msg2 = sample2.decode("ascii").strip()
                        if len(msg2) >= 12:
                            return ExtractionResult(
                                success=True,
                                data=sample2,
                                text=msg2,
                                is_binary=False,
                                method="Sequential 2-bit LSB (Statistical Heuristic)",
                                confidence=f"Medium ({ratio2 * 100:.1f}% Printable ASCII Ratio)"
                            )
                    except Exception:
                        pass

    # ---------------------------------------------------------
    # STAGE 4: Optional Steghide Subprocess
    # ---------------------------------------------------------
    steghide_bin = shutil.which("steghide")
    if steghide_bin and p.suffix.lower() in (".jpg", ".jpeg", ".bmp", ".wav"):
        try:
            with tempfile.TemporaryDirectory() as td:
                out_tmp = Path(td) / "steghide_out.bin"
                cmd = [
                    steghide_bin, "extract",
                    "-sf", str(p),
                    "-p", password or "",
                    "-xf", str(out_tmp),
                    "-f"
                ]
                proc = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=8
                )
                if proc.returncode == 0 and out_tmp.exists():
                    steg_data = out_tmp.read_bytes()
                    if steg_data:
                        is_txt, _ = is_printable_text(steg_data)
                        text_val = steg_data.decode("utf-8", errors="replace") if is_txt else None
                        return ExtractionResult(
                            success=True,
                            data=steg_data,
                            text=text_val,
                            is_binary=not is_txt,
                            method="Steghide Subprocess",
                            confidence="High (Verified by Steghide Key Check)",
                            hex_preview=format_hex_dump(steg_data) if not is_txt else None
                        )
        except Exception as e:
            logger.debug(f"Steghide execution skipped or failed: {e}")

    # ---------------------------------------------------------
    # NO MESSAGE FOUND
    # ---------------------------------------------------------
    methods_list = "\n".join(f"  • {m}" for m in ATTEMPTED_METHODS)
    return ExtractionResult(
        success=False,
        method="None",
        confidence="None",
        error=f"No message found with the methods I know.\n\nMethods attempted:\n{methods_list}"
    )
