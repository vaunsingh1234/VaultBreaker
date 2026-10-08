import numpy as np
import cv2
from scipy.signal import convolve2d
from typing import List, Dict, Any, Union
from vaultbreaker.features.image_features import LAPLACIAN_KERNEL_3X3

def explain_video(frames: Union[np.ndarray, List[np.ndarray]]) -> Dict[str, Any]:
    """
    Generate forensic explainability artifacts for video:
    1. Per-frame suspicion timeline (spatial noise residual & LSB density)
    2. Inter-frame difference residual map
    """
    if isinstance(frames, list):
        frames = np.stack(frames, axis=0)

    n_frames = frames.shape[0]
    frame_scores = []

    # Frame-by-frame anomaly score
    for i in range(n_frames):
        f = frames[i]
        gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) if f.ndim == 3 and f.shape[2] == 3 else f
        res = convolve2d(gray.astype(np.float32), LAPLACIAN_KERNEL_3X3, mode="same")
        score = float(np.var(res))
        frame_scores.append(score)

    # Frame difference map between middle frames
    if n_frames > 1:
        mid = n_frames // 2
        f1 = frames[mid - 1]
        f2 = frames[mid]
        g1 = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY) if f1.ndim == 3 and f1.shape[2] == 3 else f1
        g2 = cv2.cvtColor(f2, cv2.COLOR_RGB2GRAY) if f2.ndim == 3 and f2.shape[2] == 3 else f2
        diff = np.abs(g2.astype(np.float32) - g1.astype(np.float32))
        diff_u8 = np.clip(diff * 4.0, 0, 255).astype(np.uint8)
        diff_heatmap = cv2.applyColorMap(diff_u8, cv2.COLORMAP_VIRIDIS)
    else:
        diff_heatmap = np.zeros((128, 128, 3), dtype=np.uint8)

    return {
        "frame_timeline": frame_scores,
        "middle_diff_heatmap": diff_heatmap,
        "max_frame_idx": int(np.argmax(frame_scores)) if frame_scores else 0
    }
