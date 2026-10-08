import numpy as np
from typing import Tuple, Optional
from scipy.ndimage import sobel

def embed_image_lsb_replacement(
    image: np.ndarray,
    payload_bits: np.ndarray,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Embeds binary payload into image using LSB Replacement.
    If seed is provided, pixel positions are permuted pseudo-randomly.
    Returns:
        stego_image: modified image (uint8)
        indices: 1D array of indices where bits were embedded
    """
    stego = image.copy().astype(np.uint8)
    flat = stego.ravel()
    total_elements = flat.size
    n_bits = len(payload_bits)

    if n_bits > total_elements:
        raise ValueError(f"Payload size {n_bits} exceeds image capacity {total_elements}")

    if seed is not None:
        rng = np.random.RandomState(seed)
        indices = rng.permutation(total_elements)[:n_bits]
    else:
        indices = np.arange(n_bits)

    # LSB Replacement: clear bit 0 and insert payload bit
    flat[indices] = (flat[indices] & ~1) | (payload_bits.astype(np.uint8) & 1)
    return flat.reshape(stego.shape), indices

def extract_image_lsb_replacement(
    stego_image: np.ndarray,
    num_bits: int,
    seed: Optional[int] = None,
    indices: Optional[np.ndarray] = None
) -> np.ndarray:
    """Extracts bits embedded via LSB Replacement."""
    flat = stego_image.ravel()
    if indices is None:
        if seed is not None:
            rng = np.random.RandomState(seed)
            indices = rng.permutation(flat.size)[:num_bits]
        else:
            indices = np.arange(num_bits)
    return (flat[indices] & 1).astype(np.uint8)

def embed_image_lsb_matching(
    image: np.ndarray,
    payload_bits: np.ndarray,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Embeds binary payload into image using LSB Matching (+-1 embedding).
    If pixel's LSB doesn't match payload, randomly increment or decrement by 1.
    This avoids the pair-of-values asymmetry created by LSB replacement.
    """
    stego = image.copy().astype(np.int32)
    flat = stego.ravel()
    total_elements = flat.size
    n_bits = len(payload_bits)

    if n_bits > total_elements:
        raise ValueError(f"Payload size {n_bits} exceeds image capacity {total_elements}")

    if seed is not None:
        rng = np.random.RandomState(seed)
        indices = rng.permutation(total_elements)[:n_bits]
    else:
        rng = np.random.RandomState(42)
        indices = np.arange(n_bits)

    for i, idx in enumerate(indices):
        current_lsb = flat[idx] & 1
        target_lsb = int(payload_bits[i]) & 1
        if current_lsb != target_lsb:
            val = flat[idx]
            if val == 0:
                flat[idx] = 1
            elif val == 255:
                flat[idx] = 254
            else:
                flat[idx] += 1 if rng.rand() > 0.5 else -1

    stego_uint8 = np.clip(flat, 0, 255).astype(np.uint8).reshape(image.shape)
    return stego_uint8, indices

def extract_image_lsb_matching(
    stego_image: np.ndarray,
    num_bits: int,
    seed: Optional[int] = None,
    indices: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Extracts bits embedded via LSB Matching.
    Since matching alters pixel values until LSB == target bit,
    extraction is identical to reading the LSB.
    """
    flat = stego_image.ravel()
    if indices is None:
        if seed is not None:
            rng = np.random.RandomState(seed)
            indices = rng.permutation(flat.size)[:num_bits]
        else:
            indices = np.arange(num_bits)
    return (flat[indices] & 1).astype(np.uint8)

def embed_image_edge_adaptive(
    image: np.ndarray,
    payload_bits: np.ndarray,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Edge-adaptive steganography:
    Computes Sobel gradient magnitude to locate edges and high-frequency textures.
    Embeds payload into the highest-gradient pixels using LSB matching.
    """
    if image.ndim == 3:
        gray = np.mean(image, axis=2).astype(np.float32)
    else:
        gray = image.astype(np.float32)

    # Calculate gradient magnitude
    sx = sobel(gray, axis=0)
    sy = sobel(gray, axis=1)
    grad_mag = np.hypot(sx, sy).ravel()

    n_bits = len(payload_bits)
    if n_bits > grad_mag.size:
        raise ValueError(f"Payload size {n_bits} exceeds image capacity {grad_mag.size}")

    # Sort pixels by texture gradient descending
    sorted_order = np.argsort(-grad_mag)
    candidate_pool = sorted_order[: max(n_bits * 2, n_bits)]

    if seed is not None:
        rng = np.random.RandomState(seed)
        chosen_indices = rng.choice(candidate_pool, size=n_bits, replace=False)
    else:
        chosen_indices = candidate_pool[:n_bits]

    # Perform LSB matching on chosen high-gradient indices
    stego = image.copy().astype(np.int32)
    flat = stego.ravel()
    rng_flip = np.random.RandomState(seed if seed is not None else 42)

    for i, idx in enumerate(chosen_indices):
        current_lsb = flat[idx] & 1
        target_lsb = int(payload_bits[i]) & 1
        if current_lsb != target_lsb:
            val = flat[idx]
            if val == 0:
                flat[idx] = 1
            elif val == 255:
                flat[idx] = 254
            else:
                flat[idx] += 1 if rng_flip.rand() > 0.5 else -1

    stego_uint8 = np.clip(flat, 0, 255).astype(np.uint8).reshape(image.shape)
    return stego_uint8, chosen_indices

def extract_image_edge_adaptive(
    stego_image: np.ndarray,
    indices: np.ndarray
) -> np.ndarray:
    """Extracts bits from edge-adaptive stego given the embedding indices."""
    flat = stego_image.ravel()
    return (flat[indices] & 1).astype(np.uint8)
