"""
Codec for VaultBreaker Steganography Protocol (VBK1).
Framing:
- Magic: 4 bytes b"VBK1"
- Flags: 1 byte (bit 0: encrypted, bit 1: is_binary)
- If encrypted:
  - Salt: 16 bytes
  - Nonce: 12 bytes
  - Length: 4 bytes (uint32, length of plaintext)
  - CRC32: 4 bytes (uint32, crc32 of plaintext)
  - Ciphertext + 16-byte AES-GCM tag
- If unencrypted:
  - Length: 4 bytes (uint32)
  - CRC32: 4 bytes (uint32)
  - Data: raw bytes
- Key derivation: scrypt(passphrase, salt, length=32, n=16384, r=8, p=1)
- Pseudorandom scattering: Key seeds numpy permutation to scatter bits.
"""

import os
import zlib
import struct
import hashlib
import html
from dataclasses import dataclass
from typing import Optional, Union, Tuple, List, Dict, Any
import numpy as np

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.exceptions import InvalidTag

from vaultbreaker.utils.logger import get_logger

logger = get_logger("StegoCodec")

MAGIC = b"VBK1"


def format_size(num_bytes: int) -> str:
    """Format byte count into plain-English human-readable string."""
    if num_bytes < 1024:
        return f"{num_bytes} bytes"
    elif num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    else:
        return f"{num_bytes / (1024 * 1024):.1f} MB"


class CapacityError(Exception):
    """Raised when message size exceeds container capacity."""
    pass


@dataclass
class ExtractionResult:
    """Standardized forensic payload result from steganographic extraction."""
    success: bool
    data: bytes = b""
    text: Optional[str] = None
    is_binary: bool = False
    method: str = "Unknown"
    confidence: str = "Low"
    error: Optional[str] = None
    hex_preview: Optional[str] = None
    mime_type: Optional[str] = None

    def get_safe_display_text(self, max_length: int = 65536) -> str:
        """Return HTML-escaped text safely truncated for UI display."""
        if self.text is None:
            return ""
        truncated = self.text[:max_length]
        escaped = html.escape(truncated)
        if len(self.text) > max_length:
            escaped += f"\n... [Truncated at {max_length} characters]"
        return escaped

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "is_binary": self.is_binary,
            "method": self.method,
            "confidence": self.confidence,
            "length": len(self.data),
            "text": self.get_safe_display_text() if self.text is not None else None,
            "hex_preview": self.hex_preview,
            "mime_type": self.mime_type,
            "error": self.error
        }


def format_hex_dump(data: bytes, max_bytes: int = 256) -> str:
    """Format bytes into canonical xxd-style hex dump."""
    slice_data = data[:max_bytes]
    lines = []
    for i in range(0, len(slice_data), 16):
        chunk = slice_data[i : i + 16]
        hex_bytes = " ".join(f"{b:02x}" for b in chunk)
        ascii_chars = "".join(chr(b) if 32 <= b <= 126 else "." for b in chunk)
        lines.append(f"{i:08x}  {hex_bytes:<48}  |{ascii_chars}|")
    if len(data) > max_bytes:
        lines.append(f"... [{len(data) - max_bytes} more bytes]")
    return "\n".join(lines)


def get_scatter_indices(total_bits: int, password: Optional[str] = None) -> np.ndarray:
    """
    Generate embedding index permutation.
    If password is provided, seeds a deterministic RandomState.
    If password is None or empty, returns sequential range.
    """
    if not password:
        return np.arange(total_bits, dtype=np.int64)

    # Derive 32-bit seed from password
    h = hashlib.sha256(password.encode("utf-8") + b"_scatter_seed").digest()
    seed = int.from_bytes(h[:4], "big")
    rng = np.random.RandomState(seed)
    return rng.permutation(total_bits)


def encode_payload(
    message: Union[str, bytes],
    password: Optional[str] = None
) -> bytes:
    """
    Pack message into VaultBreaker (VBK1) protocol stream.
    Applies scrypt + AES-GCM if password is provided.
    """
    if isinstance(message, str):
        payload_bytes = message.encode("utf-8")
        is_binary = False
    elif isinstance(message, bytes):
        payload_bytes = message
        is_binary = True
    else:
        raise ValueError("Message must be either str or bytes.")

    crc = zlib.crc32(payload_bytes) & 0xFFFFFFFF
    length = len(payload_bytes)

    if password:
        # Flag bit 0: encrypted, bit 1: is_binary
        flags = 0x01 | (0x02 if is_binary else 0x00)
        salt = os.urandom(16)
        nonce = os.urandom(12)

        kdf = Scrypt(salt=salt, length=32, n=16384, r=8, p=1)
        key = kdf.derive(password.encode("utf-8"))
        aesgcm = AESGCM(key)

        # Authenticate header metadata in AAD
        aad = MAGIC + bytes([flags]) + struct.pack(">II", length, crc)
        ciphertext = aesgcm.encrypt(nonce, payload_bytes, aad)

        header = MAGIC + bytes([flags]) + salt + nonce + struct.pack(">II", length, crc)
        return header + ciphertext
    else:
        flags = 0x02 if is_binary else 0x00
        header = MAGIC + bytes([flags]) + struct.pack(">II", length, crc)
        return header + payload_bytes


def decode_payload(
    stream: bytes,
    password: Optional[str] = None
) -> ExtractionResult:
    """
    Parse and authenticate a VBK1 stream.
    Validates magic bytes, decrypts with AES-GCM (if encrypted), and verifies CRC32.
    Cleanly fails without garbage if password is wrong or stream is corrupted.
    """
    if len(stream) < 4:
        return ExtractionResult(success=False, error="Stream too short to contain VBK1 header")

    if not stream.startswith(MAGIC):
        return ExtractionResult(success=False, error="Invalid magic signature: not a VBK1 stream")

    if len(stream) < 5:
        return ExtractionResult(success=False, error="Truncated VBK1 flags header")

    flags = stream[4]
    is_encrypted = bool(flags & 0x01)
    is_binary = bool(flags & 0x02)

    if is_encrypted:
        # Expected header: Magic (4) + Flags (1) + Salt (16) + Nonce (12) + Length (4) + CRC32 (4) = 41 bytes
        if len(stream) < 41:
            return ExtractionResult(success=False, error="Truncated VBK1 encrypted header")

        salt = stream[5:21]
        nonce = stream[21:33]
        length, crc = struct.unpack(">II", stream[33:41])

        if password is None or password == "":
            return ExtractionResult(
                success=False,
                is_binary=is_binary,
                error="Message is encrypted with a password. Please provide the passphrase."
            )

        ciphertext = stream[41 : 41 + length + 16] # 16 bytes for AES-GCM tag
        if len(ciphertext) < length + 16:
            return ExtractionResult(success=False, error="Truncated ciphertext payload")

        try:
            kdf = Scrypt(salt=salt, length=32, n=16384, r=8, p=1)
            key = kdf.derive(password.encode("utf-8"))
            aesgcm = AESGCM(key)
            aad = MAGIC + bytes([flags]) + struct.pack(">II", length, crc)
            plaintext = aesgcm.decrypt(nonce, ciphertext, aad)
        except InvalidTag:
            return ExtractionResult(
                success=False,
                error="Decryption failed: Incorrect password or corrupted payload."
            )
        except Exception as e:
            return ExtractionResult(
                success=False,
                error=f"Cryptographic error: {e}"
            )

        # Verify CRC32
        actual_crc = zlib.crc32(plaintext) & 0xFFFFFFFF
        if actual_crc != crc:
            return ExtractionResult(
                success=False,
                error=f"Integrity check failed: CRC32 mismatch (expected {crc:#x}, got {actual_crc:#x})"
            )

        # Successful extraction
        text = None
        if not is_binary:
            try:
                text = plaintext.decode("utf-8")
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
    else:
        # Unencrypted header: Magic (4) + Flags (1) + Length (4) + CRC32 (4) = 13 bytes
        if len(stream) < 13:
            return ExtractionResult(success=False, error="Truncated VBK1 unencrypted header")

        length, crc = struct.unpack(">II", stream[5:13])
        data = stream[13 : 13 + length]
        if len(data) < length:
            return ExtractionResult(success=False, error="Truncated payload data")

        actual_crc = zlib.crc32(data) & 0xFFFFFFFF
        if actual_crc != crc:
            return ExtractionResult(
                success=False,
                error=f"Integrity check failed: CRC32 mismatch (expected {crc:#x}, got {actual_crc:#x})"
            )

        text = None
        if not is_binary:
            try:
                text = data.decode("utf-8")
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
