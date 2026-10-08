import numpy as np
from typing import Tuple, Optional

def embed_audio_lsb_replacement(
    audio_samples: np.ndarray,
    payload_bits: np.ndarray,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Embeds binary payload into 16-bit PCM audio samples via LSB Replacement.
    Returns:
        stego_samples: modified samples (int16)
        indices: sample indices used for embedding
    """
    samples = audio_samples.copy().astype(np.int16)
    total_samples = len(samples)
    n_bits = len(payload_bits)

    if n_bits > total_samples:
        raise ValueError(f"Payload size {n_bits} exceeds audio capacity {total_samples}")

    if seed is not None:
        rng = np.random.RandomState(seed)
        indices = rng.permutation(total_samples)[:n_bits]
    else:
        indices = np.arange(n_bits)

    # Convert to unsigned 16-bit view to perform bitwise operations safely
    u_samples = samples.view(np.uint16)
    u_samples[indices] = (u_samples[indices] & ~1) | (payload_bits.astype(np.uint16) & 1)

    return samples, indices

def extract_audio_lsb_replacement(
    stego_samples: np.ndarray,
    num_bits: int,
    seed: Optional[int] = None,
    indices: Optional[np.ndarray] = None
) -> np.ndarray:
    """Extracts bits embedded in audio via LSB replacement."""
    samples = stego_samples.astype(np.int16)
    if indices is None:
        if seed is not None:
            rng = np.random.RandomState(seed)
            indices = rng.permutation(len(samples))[:num_bits]
        else:
            indices = np.arange(num_bits)
    u_samples = samples.view(np.uint16)
    return (u_samples[indices] & 1).astype(np.uint8)

def embed_audio_lsb_matching(
    audio_samples: np.ndarray,
    payload_bits: np.ndarray,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Embeds binary payload into 16-bit PCM audio samples via LSB Matching (+-1).
    Avoids asymmetric distribution changes caused by simple replacement.
    """
    samples = audio_samples.copy().astype(np.int32)
    total_samples = len(samples)
    n_bits = len(payload_bits)

    if n_bits > total_samples:
        raise ValueError(f"Payload size {n_bits} exceeds audio capacity {total_samples}")

    if seed is not None:
        rng = np.random.RandomState(seed)
        indices = rng.permutation(total_samples)[:n_bits]
    else:
        rng = np.random.RandomState(42)
        indices = np.arange(n_bits)

    for i, idx in enumerate(indices):
        current_lsb = samples[idx] & 1
        target_lsb = int(payload_bits[i]) & 1
        if current_lsb != target_lsb:
            val = samples[idx]
            if val <= -32768:
                samples[idx] = -32767
            elif val >= 32767:
                samples[idx] = 32766
            else:
                samples[idx] += 1 if rng.rand() > 0.5 else -1

    stego_int16 = np.clip(samples, -32768, 32767).astype(np.int16)
    return stego_int16, indices

def extract_audio_lsb_matching(
    stego_samples: np.ndarray,
    num_bits: int,
    seed: Optional[int] = None,
    indices: Optional[np.ndarray] = None
) -> np.ndarray:
    """Extracts bits embedded via audio LSB matching."""
    samples = stego_samples.astype(np.int16)
    if indices is None:
        if seed is not None:
            rng = np.random.RandomState(seed)
            indices = rng.permutation(len(samples))[:num_bits]
        else:
            indices = np.arange(num_bits)
    u_samples = samples.view(np.uint16)
    return (u_samples[indices] & 1).astype(np.uint8)
