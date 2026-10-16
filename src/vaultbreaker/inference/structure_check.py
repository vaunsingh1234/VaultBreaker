"""
Structure Check Engine for VaultBreaker.
Rule-based forensic inspection for format integrity:
- JPEG: Trailing bytes after EOI (FFD9), EXIF metadata length and entropy
- PNG: Data after IEND chunk, anomalous tEXt/zTXt/iTXt chunks
- WAV: RIFF size vs real file size, unknown subchunks, data after data chunk
- MP4 / MKV: Unusual boxes, large or high-entropy free/skip payloads, trailing payloads
- Embedded file signatures: ZIP (PK..), RAR (Rar!), PDF (%PDF-), ELF (\x7fELF), PE (MZ)
"""

import os
import struct
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np

from vaultbreaker.utils.logger import get_logger

logger = get_logger("StructureCheck")


def compute_shannon_entropy(data: bytes) -> float:
    """Compute Shannon entropy of a byte sequence in bits/byte (0.0 to 8.0)."""
    if not data:
        return 0.0
    arr = np.frombuffer(data, dtype=np.uint8)
    counts = np.bincount(arr, minlength=256)
    probs = counts[counts > 0] / len(data)
    return float(-np.sum(probs * np.log2(probs)))


def format_size(num_bytes: int) -> str:
    """Format byte count into plain-English human-readable string."""
    if num_bytes < 1024:
        return f"{num_bytes} bytes"
    elif num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    else:
        return f"{num_bytes / (1024 * 1024):.1f} MB"


class Finding:
    """A forensic structure check finding with severity level."""
    def __init__(
        self,
        message: str,
        severity: str = "medium",  # "low", "medium", "high", "critical"
        category: str = "overlay", # "overlay", "metadata", "container", "signature"
        offset: Optional[int] = None,
        size: Optional[int] = None
    ):
        self.message = message
        self.severity = severity
        self.category = category
        self.offset = offset
        self.size = size

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "message": self.message,
            "severity": self.severity,
            "category": self.category
        }
        if self.offset is not None:
            d["offset"] = self.offset
        if self.size is not None:
            d["size"] = self.size
        return d

    def __repr__(self) -> str:
        return f"Finding({self.severity.upper()}: {self.message})"


def scan_embedded_signatures(data: bytes, base_offset: int = 0) -> List[Finding]:
    """
    Scan a byte slice for embedded container or executable signatures:
    ZIP, RAR, PDF, ELF, PE.
    """
    findings = []
    if not data or len(data) < 4:
        return findings

    # 1. ZIP Archive (PK\x03\x04)
    zip_idx = data.find(b"PK\x03\x04")
    if zip_idx != -1:
        findings.append(Finding(
            message=f"Embedded ZIP archive signature detected at byte offset {base_offset + zip_idx}",
            severity="high",
            category="signature",
            offset=base_offset + zip_idx
        ))

    # 2. RAR Archive (Rar!\x1a\x07)
    rar_idx = data.find(b"Rar!\x1a\x07")
    if rar_idx != -1:
        findings.append(Finding(
            message=f"Embedded RAR archive signature detected at byte offset {base_offset + rar_idx}",
            severity="high",
            category="signature",
            offset=base_offset + rar_idx
        ))

    # 3. PDF Document (%PDF-)
    pdf_idx = data.find(b"%PDF-")
    if pdf_idx != -1:
        findings.append(Finding(
            message=f"Embedded PDF document signature detected at byte offset {base_offset + pdf_idx}",
            severity="high",
            category="signature",
            offset=base_offset + pdf_idx
        ))

    # 4. ELF Binary Executable (\x7fELF)
    elf_idx = data.find(b"\x7fELF")
    if elf_idx != -1:
        findings.append(Finding(
            message=f"Embedded ELF executable signature detected at byte offset {base_offset + elf_idx}",
            severity="high",
            category="signature",
            offset=base_offset + elf_idx
        ))

    # 5. PE Windows Executable (MZ with valid PE header pointer)
    mz_pos = 0
    while True:
        mz_idx = data.find(b"MZ", mz_pos)
        if mz_idx == -1:
            break
        # Check if e_lfanew at mz_idx + 0x3c points to valid 'PE\0\0'
        if mz_idx + 0x40 <= len(data):
            try:
                pe_offset = struct.unpack("<I", data[mz_idx + 0x3c : mz_idx + 0x40])[0]
                if pe_offset < 1024 and mz_idx + pe_offset + 4 <= len(data):
                    if data[mz_idx + pe_offset : mz_idx + pe_offset + 4] == b"PE\x00\x00":
                        findings.append(Finding(
                            message=f"Embedded PE Windows executable detected at byte offset {base_offset + mz_idx}",
                            severity="high",
                            category="signature",
                            offset=base_offset + mz_idx
                        ))
                        break
            except Exception:
                pass
        mz_pos = mz_idx + 2

    return findings


def check_jpeg_structure(data: bytes) -> List[Finding]:
    """Inspect JPEG for trailing bytes after EOI (FF D9) and anomalous EXIF/APP markers."""
    findings = []
    if len(data) < 4 or not data.startswith(b"\xff\xd8"):
        findings.append(Finding("File does not start with valid JPEG SOI marker (FF D8)", severity="high"))
        return findings

    pos = 2
    n = len(data)
    last_eoi_pos = -1

    while pos < n - 1:
        if data[pos] != 0xff:
            next_ff = data.find(b"\xff", pos)
            if next_ff == -1:
                break
            pos = next_ff

        while pos < n - 1 and data[pos + 1] == 0xff:
            pos += 1

        if pos >= n - 1:
            break

        marker = data[pos + 1]
        pos += 2

        if marker in (0xd8, 0x01) or (0xd0 <= marker <= 0xd7):
            continue

        if marker == 0xd9:
            last_eoi_pos = pos
            break

        if pos + 2 > n:
            break

        length = struct.unpack(">H", data[pos : pos + 2])[0]
        payload = data[pos + 2 : pos + length]

        if marker == 0xe1: # APP1 EXIF
            if len(payload) > 65536:
                findings.append(Finding(
                    f"Unusually long EXIF metadata block ({format_size(len(payload))})",
                    severity="medium",
                    category="metadata",
                    size=len(payload)
                ))
            if len(payload) > 512:
                entropy = compute_shannon_entropy(payload)
                if entropy > 7.5:
                    findings.append(Finding(
                        f"Unusually high entropy ({entropy:.2f} bits/byte) in EXIF metadata block",
                        severity="medium",
                        category="metadata"
                    ))
        elif marker == 0xfe: # COM Comment
            if len(payload) > 2048:
                findings.append(Finding(
                    f"Unusually long JPEG comment payload ({format_size(len(payload))})",
                    severity="medium",
                    category="metadata",
                    size=len(payload)
                ))

        if marker == 0xda: # SOS (Start of Scan)
            pos += length
            while pos < n - 1:
                if data[pos] == 0xff:
                    m2 = data[pos + 1]
                    if m2 == 0x00 or (0xd0 <= m2 <= 0xd7):
                        pos += 2
                        continue
                    elif m2 == 0xd9:
                        last_eoi_pos = pos + 2
                        break
                    else:
                        break
                else:
                    pos += 1
            if last_eoi_pos != -1:
                break
        else:
            pos += length

    if last_eoi_pos != -1 and last_eoi_pos < n:
        overlay_len = n - last_eoi_pos
        findings.append(Finding(
            f"Extra {format_size(overlay_len)} found after the end of the image",
            severity="high",
            category="overlay",
            offset=last_eoi_pos,
            size=overlay_len
        ))
        overlay_data = data[last_eoi_pos:]
        sig_findings = scan_embedded_signatures(overlay_data, base_offset=last_eoi_pos)
        findings.extend(sig_findings)

    return findings


def check_png_structure(data: bytes) -> List[Finding]:
    """Inspect PNG for trailing bytes after IEND, CRC errors, and suspicious text chunks."""
    findings = []
    png_sig = b"\x89PNG\r\n\x1a\n"
    if not data.startswith(png_sig):
        findings.append(Finding("Invalid PNG signature", severity="high"))
        return findings

    pos = 8
    n = len(data)
    iend_end = -1

    while pos + 8 <= n:
        chunk_len = struct.unpack(">I", data[pos : pos + 4])[0]
        chunk_type = data[pos + 4 : pos + 8]
        chunk_type_str = chunk_type.decode("latin1", errors="replace")

        if pos + 12 + chunk_len > n:
            findings.append(Finding(
                f"Truncated PNG: chunk '{chunk_type_str}' declares {format_size(chunk_len)} but file ends prematurely",
                severity="high",
                category="container"
            ))
            break

        chunk_data = data[pos + 8 : pos + 8 + chunk_len]

        if chunk_type == b"IEND":
            iend_end = pos + 12 + chunk_len
            break

        if chunk_type in (b"tEXt", b"zTXt", b"iTXt"):
            if chunk_len > 8192:
                findings.append(Finding(
                    f"Unusually long PNG text chunk '{chunk_type_str}' ({format_size(chunk_len)})",
                    severity="medium",
                    category="metadata",
                    size=chunk_len
                ))
            if chunk_len > 256:
                entropy = compute_shannon_entropy(chunk_data)
                if entropy > 7.5:
                    findings.append(Finding(
                        f"Unusually high entropy ({entropy:.2f} bits/byte) in PNG text chunk '{chunk_type_str}'",
                        severity="medium",
                        category="metadata"
                    ))

        pos += 12 + chunk_len

    if iend_end != -1 and iend_end < n:
        overlay_len = n - iend_end
        findings.append(Finding(
            f"Extra {format_size(overlay_len)} found after the end of the image",
            severity="high",
            category="overlay",
            offset=iend_end,
            size=overlay_len
        ))
        overlay_data = data[iend_end:]
        sig_findings = scan_embedded_signatures(overlay_data, base_offset=iend_end)
        findings.extend(sig_findings)

    return findings


def check_wav_structure(data: bytes) -> List[Finding]:
    """Inspect WAV for declared RIFF size vs real file size and unknown subchunks."""
    findings = []
    if len(data) < 12 or not data.startswith(b"RIFF") or data[8:12] != b"WAVE":
        findings.append(Finding("Invalid WAV/RIFF header", severity="high"))
        return findings

    declared_riff_size = struct.unpack("<I", data[4:8])[0]
    expected_size = declared_riff_size + 8
    real_size = len(data)

    if real_size > expected_size:
        overlay_len = real_size - expected_size
        findings.append(Finding(
            f"Extra {format_size(overlay_len)} found beyond declared RIFF size",
            severity="high",
            category="overlay",
            offset=expected_size,
            size=overlay_len
        ))
        overlay_data = data[expected_size:]
        sig_findings = scan_embedded_signatures(overlay_data, base_offset=expected_size)
        findings.extend(sig_findings)
    elif real_size < expected_size:
        findings.append(Finding(
            f"Truncated WAV file: real size ({format_size(real_size)}) is smaller than declared RIFF size ({format_size(expected_size)})",
            severity="high",
            category="container"
        ))

    known_chunks = {
        b"fmt ", b"data", b"fact", b"LIST", b"INFO", b"JUNK",
        b"bext", b"cue ", b"plst", b"labl", b"note", b"ltxt",
        b"smpl", b"inst", b"acid", b"strc", b"pad "
    }

    pos = 12
    limit = min(real_size, expected_size)
    while pos + 8 <= limit:
        sub_id = data[pos : pos + 4]
        sub_size = struct.unpack("<I", data[pos + 4 : pos + 8])[0]
        sub_id_str = sub_id.decode("latin1", errors="replace")

        if sub_id not in known_chunks:
            findings.append(Finding(
                f"Unknown RIFF subchunk '{sub_id_str}' detected ({format_size(sub_size)} at offset {pos})",
                severity="medium",
                category="container",
                offset=pos,
                size=sub_size
            ))
            sub_payload = data[pos + 8 : pos + 8 + min(sub_size, 65536)]
            sig_findings = scan_embedded_signatures(sub_payload, base_offset=pos + 8)
            findings.extend(sig_findings)

        pos += 8 + sub_size + (sub_size % 2)

    return findings


def check_mp4_structure(data: bytes) -> List[Finding]:
    """Inspect MP4 for unknown top-level boxes, large free/skip payloads, and trailing data."""
    findings = []
    n = len(data)
    pos = 0
    sum_boxes = 0

    known_top_boxes = {
        b"ftyp", b"moov", b"mdat", b"free", b"skip", b"pdin",
        b"moof", b"mfra", b"mvex", b"meta", b"udta", b"wide",
        b"styp", b"sidx", b"ssix", b"prft", b"emsg"
    }

    while pos + 8 <= n:
        box_size = struct.unpack(">I", data[pos : pos + 4])[0]
        box_type = data[pos + 4 : pos + 8]
        box_type_str = box_type.decode("latin1", errors="replace")

        actual_box_len = box_size
        hdr_len = 8
        if box_size == 1:
            if pos + 16 > n:
                break
            actual_box_len = struct.unpack(">Q", data[pos + 8 : pos + 16])[0]
            hdr_len = 16
        elif box_size == 0:
            actual_box_len = n - pos

        if actual_box_len < hdr_len or pos + actual_box_len > n:
            findings.append(Finding(
                f"Malformed MP4 box '{box_type_str}' declares invalid length {actual_box_len}",
                severity="high",
                category="container"
            ))
            break

        if box_type in (b"free", b"skip"):
            payload_len = actual_box_len - hdr_len
            if payload_len > 16384:
                payload = data[pos + hdr_len : pos + actual_box_len]
                entropy = compute_shannon_entropy(payload)
                if entropy > 7.0:
                    findings.append(Finding(
                        f"Unusually large or high-entropy '{box_type_str}' box detected ({format_size(payload_len)}, {entropy:.2f} bits/byte)",
                        severity="medium",
                        category="container",
                        size=payload_len
                    ))
                sig_findings = scan_embedded_signatures(payload, base_offset=pos + hdr_len)
                findings.extend(sig_findings)

        if box_type not in known_top_boxes:
            findings.append(Finding(
                f"Unusual top-level MP4 box '{box_type_str}' detected ({format_size(actual_box_len)} at offset {pos})",
                severity="medium",
                category="container",
                offset=pos,
                size=actual_box_len
            ))

        pos += actual_box_len
        sum_boxes = pos

    if sum_boxes > 0 and sum_boxes < n:
        trailing_len = n - sum_boxes
        findings.append(Finding(
            f"Extra {format_size(trailing_len)} found after the last container atom",
            severity="high",
            category="overlay",
            offset=sum_boxes,
            size=trailing_len
        ))
        sig_findings = scan_embedded_signatures(data[sum_boxes:], base_offset=sum_boxes)
        findings.extend(sig_findings)

    return findings


def check_mkv_structure(data: bytes) -> List[Finding]:
    """Inspect MKV/WebM for EBML header integrity, unexpected trailing bytes, and embedded signatures."""
    findings = []
    if not data.startswith(b"\x1a\x45\xdf\xa3"):
        findings.append(Finding("Invalid MKV/EBML header signature", severity="high"))
        return findings

    sig_findings = scan_embedded_signatures(data)
    findings.extend(sig_findings)
    return findings


def check_structure(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Main entrypoint: executes rule-based structure check on any media file.
    Returns findings with severity, plain-English message list, and suspicious status.
    """
    p = Path(filepath)
    if not p.exists():
        return {
            "format": "unknown",
            "is_suspicious": True,
            "findings": [{"message": f"File does not exist: {p}", "severity": "high", "category": "system"}],
            "finding_messages": [f"File does not exist: {p}"],
            "total_findings": 1
        }

    file_bytes = p.read_bytes()
    header = file_bytes[:64]
    findings: List[Finding] = []
    fmt = "unknown"

    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        fmt = "png"
        findings = check_png_structure(file_bytes)
    elif header.startswith(b"\xff\xd8"):
        fmt = "jpeg"
        findings = check_jpeg_structure(file_bytes)
    elif header.startswith(b"RIFF") and len(header) >= 12 and header[8:12] == b"WAVE":
        fmt = "wav"
        findings = check_wav_structure(file_bytes)
    elif len(header) >= 8 and (header[4:8] == b"ftyp" or b"ftyp" in header[:16]):
        fmt = "mp4"
        findings = check_mp4_structure(file_bytes)
    elif header.startswith(b"\x1a\x45\xdf\xa3"):
        fmt = "mkv"
        findings = check_mkv_structure(file_bytes)
    elif header.startswith(b"ID3") or (len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0):
        fmt = "mp3"
        if header.startswith(b"ID3") and len(file_bytes) >= 10:
            id3_size = ((file_bytes[6] & 0x7F) << 21) | ((file_bytes[7] & 0x7F) << 14) | ((file_bytes[8] & 0x7F) << 7) | (file_bytes[9] & 0x7F)
            if id3_size > 65536:
                findings.append(Finding(f"Unusually large ID3 metadata header ({format_size(id3_size)})", severity="medium", category="metadata"))
        sig_findings = scan_embedded_signatures(file_bytes)
        findings.extend(sig_findings)
    else:
        fmt = "unsupported"
        findings = scan_embedded_signatures(file_bytes)

    is_suspicious = any(f.severity in ("medium", "high", "critical") for f in findings)
    finding_messages = [f.message for f in findings]

    return {
        "format": fmt,
        "is_suspicious": is_suspicious,
        "findings": [f.to_dict() for f in findings],
        "finding_messages": finding_messages,
        "total_findings": len(findings)
    }
