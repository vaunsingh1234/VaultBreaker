import numpy as np
import cv2
from scipy.signal import convolve2d
from typing import Tuple, Dict, Any
from vaultbreaker.features.image_features import KV_KERNEL_5X5

def explain_image(image: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Generate visual explainability artifacts for an image:
    1. High-Pass Spatial Residual Map (SRM KV kernel): reveals local noise concentrations
    2. LSB Plane Bit-Map: reveals artificial spatial bit uniformity or patterns
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.shape[2] == 3 else image[:, :, 0]
    else:
        gray = image.copy()

    gray_f = gray.astype(np.float32)
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)

    # 1. SRM KV Residual Heatmap
    residual = convolve2d(gray_f, KV_KERNEL_5X5, mode="same", boundary="symm")
    abs_res = np.abs(residual)
    # Normalize to 0..255 for visualization
    res_norm = np.clip((abs_res / max(1e-4, np.percentile(abs_res, 99))) * 255.0, 0, 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(res_norm, cv2.COLORMAP_INFERNO)

    # 2. LSB Plane (0 or 255)
    lsb_plane = (gray_u8 & 1) * 255

    return {
        "residual_heatmap": heatmap_colored,
        "lsb_plane": lsb_plane,
        "mean_residual": float(np.mean(abs_res)),
        "max_residual": float(np.max(abs_res))
    }
