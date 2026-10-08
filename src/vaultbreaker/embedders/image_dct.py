import numpy as np
from scipy.fftpack import dct, idct
from typing import Tuple, Optional

# Standard JPEG Luminance Quantization Matrix
JPEG_LUMA_QUANT = np.array([
    [16, 11, 10, 16,  24,  40,  51,  61],
    [12, 12, 14, 19,  26,  58,  60,  55],
    [14, 13, 16, 24,  40,  57,  69,  56],
    [14, 17, 22, 29,  51,  87,  80,  62],
    [18, 22, 37, 56,  68, 109, 103,  77],
    [24, 35, 55, 64,  81, 104, 113,  92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103,  99]
], dtype=np.float32)

def block2d_dct(block: np.ndarray) -> np.ndarray:
    """Compute 2D Type-II orthogonal DCT on an 8x8 block."""
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def block2d_idct(block: np.ndarray) -> np.ndarray:
    """Compute 2D Type-II orthogonal IDCT on an 8x8 block."""
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def embed_image_dct(
    image: np.ndarray,
    payload_bits: np.ndarray,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, dict]:
    """
    Pure Python DCT mid-frequency coefficient LSB steganography.
    Splits image into 8x8 blocks, computes DCT, quantizes, and embeds
    bits into non-zero mid-frequency AC coefficients.
    Returns:
        stego_image: modified image (uint8)
        meta: dictionary with block coordinates and coefficient offsets
    """
    if image.ndim == 3:
        img_gray = image[:, :, 0].copy().astype(np.float32)
        is_color = True
    else:
        img_gray = image.copy().astype(np.float32)
        is_color = False

    H, W = img_gray.shape
    H_blocks = H // 8
    W_blocks = W // 8

    # We use mid-frequency indices in 8x8 block (avoiding DC at (0,0) and extreme high frequencies)
    mid_freq_coords = [
        (1, 2), (2, 1), (2, 2), (1, 3), (3, 1),
        (2, 3), (3, 2), (3, 3), (2, 4), (4, 2),
        (3, 4), (4, 3), (4, 4), (3, 5), (5, 3)
    ]

    total_slots = H_blocks * W_blocks * len(mid_freq_coords)
    n_bits = len(payload_bits)
    if n_bits > total_slots:
        raise ValueError(f"Payload size {n_bits} exceeds DCT slot capacity {total_slots}")

    # Generate slot list
    all_slots = []
    for bi in range(H_blocks):
        for bj in range(W_blocks):
            for ci, cj in mid_freq_coords:
                all_slots.append((bi, bj, ci, cj))

    if seed is not None:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(all_slots))[:n_bits]
        chosen_slots = [all_slots[i] for i in perm]
    else:
        chosen_slots = all_slots[:n_bits]

    # Map slot -> bit
    slot_map = {slot: payload_bits[idx] for idx, slot in enumerate(chosen_slots)}

    stego_gray = np.zeros_like(img_gray)

    for bi in range(H_blocks):
        for bj in range(W_blocks):
            r_start, r_end = bi * 8, (bi + 1) * 8
            c_start, c_end = bj * 8, (bj + 1) * 8
            block = img_gray[r_start:r_end, c_start:c_end] - 128.0

            # 2D DCT
            d_block = block2d_dct(block)
            q_block = np.round(d_block / JPEG_LUMA_QUANT).astype(np.int32)

            for ci, cj in mid_freq_coords:
                key = (bi, bj, ci, cj)
                if key in slot_map:
                    bit = int(slot_map[key]) & 1
                    coeff = q_block[ci, cj]
                    q_block[ci, cj] = (coeff & ~1) | bit

            # Inverse DCT
            recon = block2d_idct(q_block.astype(np.float32) * JPEG_LUMA_QUANT) + 128.0
            stego_gray[r_start:r_end, c_start:c_end] = recon

    stego_gray = np.clip(stego_gray, 0, 255).astype(np.uint8)

    if is_color:
        stego_final = image.copy()
        stego_final[:, :, 0] = stego_gray
    else:
        stego_final = stego_gray

    meta = {"slots": chosen_slots, "num_bits": n_bits}
    return stego_final, meta

def extract_image_dct(
    stego_image: np.ndarray,
    meta: dict
) -> np.ndarray:
    """Extract embedded bits from DCT domain using recorded slot metadata."""
    if stego_image.ndim == 3:
        img_gray = stego_image[:, :, 0].astype(np.float32)
    else:
        img_gray = stego_image.astype(np.float32)

    slots = meta["slots"]
    extracted = []

    for bi, bj, ci, cj in slots:
        r_start, r_end = bi * 8, (bi + 1) * 8
        c_start, c_end = bj * 8, (bj + 1) * 8
        block = img_gray[r_start:r_end, c_start:c_end] - 128.0
        d_block = block2d_dct(block)
        q_block = np.round(d_block / JPEG_LUMA_QUANT).astype(np.int32)
        bit = q_block[ci, cj] & 1
        extracted.append(bit)

    return np.array(extracted, dtype=np.uint8)
