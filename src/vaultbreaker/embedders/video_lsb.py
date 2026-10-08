import numpy as np
from typing import Tuple, List, Optional
from vaultbreaker.embedders.image_lsb import (
    embed_image_lsb_replacement, extract_image_lsb_replacement,
    embed_image_lsb_matching, extract_image_lsb_matching
)

def embed_video_frame_lsb(
    frames: List[np.ndarray],
    payload_bits: np.ndarray,
    frame_indices: Optional[List[int]] = None,
    seed: Optional[int] = None
) -> Tuple[List[np.ndarray], dict]:
    """
    Embeds binary payload across video frames using LSB replacement.
    Embeds across all frames by default to guarantee high spatial and
    temporal steganographic visibility regardless of temporal frame sampling.
    """
    num_frames = len(frames)
    if frame_indices is None:
        frame_indices = list(range(num_frames))

    n_bits = len(payload_bits)
    bits_per_frame = n_bits // len(frame_indices)
    remainder = n_bits % len(frame_indices)

    stego_frames = [f.copy() for f in frames]
    bit_offset = 0
    frame_metadata = []

    for idx, f_idx in enumerate(frame_indices):
        chunk_len = bits_per_frame + (1 if idx < remainder else 0)
        if chunk_len == 0:
            continue
        chunk = payload_bits[bit_offset : bit_offset + chunk_len]
        f_seed = (seed + f_idx * 17) if seed is not None else None
        stego_frame, used_indices = embed_image_lsb_replacement(
            stego_frames[f_idx], chunk, seed=f_seed
        )
        stego_frames[f_idx] = stego_frame
        frame_metadata.append({
            "frame_idx": f_idx,
            "num_bits": chunk_len,
            "seed": f_seed
        })
        bit_offset += chunk_len

    meta = {
        "frame_metadata": frame_metadata,
        "total_bits": n_bits,
        "method": "frame_lsb"
    }
    return stego_frames, meta

def extract_video_frame_lsb(
    frames: List[np.ndarray],
    meta: dict
) -> np.ndarray:
    """Extracts bits from stego video frames using recorded metadata."""
    extracted = []
    for info in meta["frame_metadata"]:
        f_idx = info["frame_idx"]
        num_bits = info["num_bits"]
        seed = info["seed"]
        bits = extract_image_lsb_replacement(frames[f_idx], num_bits, seed=seed)
        extracted.append(bits)
    if not extracted:
        return np.array([], dtype=np.uint8)
    return np.concatenate(extracted)

def embed_video_frame_lsb_matching(
    frames: List[np.ndarray],
    payload_bits: np.ndarray,
    frame_indices: Optional[List[int]] = None,
    seed: Optional[int] = None
) -> Tuple[List[np.ndarray], dict]:
    """Embeds binary payload across video frames using LSB matching (+-1)."""
    num_frames = len(frames)
    if frame_indices is None:
        frame_indices = list(range(num_frames))

    n_bits = len(payload_bits)
    bits_per_frame = n_bits // len(frame_indices)
    remainder = n_bits % len(frame_indices)

    stego_frames = [f.copy() for f in frames]
    bit_offset = 0
    frame_metadata = []

    for idx, f_idx in enumerate(frame_indices):
        chunk_len = bits_per_frame + (1 if idx < remainder else 0)
        if chunk_len == 0:
            continue
        chunk = payload_bits[bit_offset : bit_offset + chunk_len]
        f_seed = (seed + f_idx * 17) if seed is not None else None
        stego_frame, used_indices = embed_image_lsb_matching(
            stego_frames[f_idx], chunk, seed=f_seed
        )
        stego_frames[f_idx] = stego_frame
        frame_metadata.append({
            "frame_idx": f_idx,
            "num_bits": chunk_len,
            "seed": f_seed
        })
        bit_offset += chunk_len

    meta = {
        "frame_metadata": frame_metadata,
        "total_bits": n_bits,
        "method": "frame_lsb_matching"
    }
    return stego_frames, meta

def extract_video_frame_lsb_matching(
    frames: List[np.ndarray],
    meta: dict
) -> np.ndarray:
    """Extracts bits embedded in video via LSB matching."""
    extracted = []
    for info in meta["frame_metadata"]:
        f_idx = info["frame_idx"]
        num_bits = info["num_bits"]
        seed = info["seed"]
        bits = extract_image_lsb_matching(frames[f_idx], num_bits, seed=seed)
        extracted.append(bits)
    if not extracted:
        return np.array([], dtype=np.uint8)
    return np.concatenate(extracted)

def embed_video_audio_track(
    audio_samples: np.ndarray,
    payload_bits: np.ndarray,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, dict]:
    """Embeds payload into the audio track of the video."""
    from vaultbreaker.embedders.audio_lsb import embed_audio_lsb_replacement
    stego_samples, indices = embed_audio_lsb_replacement(audio_samples, payload_bits, seed=seed)
    meta = {
        "num_bits": len(payload_bits),
        "seed": seed,
        "type": "video_audio_track"
    }
    return stego_samples, meta

def extract_video_audio_track(
    stego_audio_samples: np.ndarray,
    meta: dict
) -> np.ndarray:
    """Extracts payload from the video audio track."""
    from vaultbreaker.embedders.audio_lsb import extract_audio_lsb_replacement
    return extract_audio_lsb_replacement(
        stego_audio_samples, meta["num_bits"], seed=meta["seed"]
    )
