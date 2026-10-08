import numpy as np
from typing import List, Tuple, Dict

# Descriptive names for core features
IMAGE_FEATURE_NAMES = [
    "LSB Density", "LSB Variance", "LSB Horiz Transitions", "LSB Vert Transitions",
    "LSB Shannon Entropy", "LSB Run Mean", "LSB Run Variance", "LSB-Bit1 Correlation",
    "Chi-Square Norm", "Chi-Square Log", "SPA Difference", "Diag Gradient Var",
    "Parity Autocorr", "Parity Imbalance", "Pixel Mean", "Pixel Std",
    "Pixel Skewness", "Pixel Kurtosis", "Hist Diff Mean", "Hist Diff Std",
    "Zero Pixel Ratio", "Saturated Pixel Ratio", "SRM Horiz Res Var", "SRM Horiz Res Kurt",
    "SRM Horiz Dyn Range", "SRM Vert Res Var", "SRM Vert Res Kurt", "SRM Vert Dyn Range",
    "Laplacian Res Var", "Laplacian Res Kurt", "SRM KV Res Var", "SRM KV MAD",
    "SRM KV Skew", "SRM KV Kurt"
] + [f"SPAM Transition {i}" for i in range(24)] + [
    "Blockiness Vert", "Blockiness Horiz", "Blockiness Total", "DCT DC Variance",
    "DCT Mid-Freq Energy", "DCT High-Freq Energy", "DCT Mid Zero Frac", "DCT Mid Skew",
    "DCT Mid Kurt", "DCT Quant Artifacts", "DCT AC Mean Abs", "DCT AC Std",
    "DCT AC Max", "DCT AC Median MAD"
]

AUDIO_FEATURE_NAMES = [
    "LSB Density", "LSB Variance", "LSB Transitions", "LSB Entropy",
    "LSB Run Mean", "LSB Run Variance", "Chi-Square Norm", "Chi-Square Log",
    "Amp Mean", "Amp Std", "Amp Skewness", "Amp Kurtosis",
    "Zero Crossing Rate Mean", "Zero Crossing Rate Std", "Zero Sample Ratio", "Dynamic Range",
    "Spectral Centroid Mean", "Spectral Centroid Std", "Spectral Flatness Mean", "Spectral Flatness Std",
    "Spectral Rolloff Mean", "Spectral Rolloff Std", "Spectral Bandwidth Mean", "Spectral Bandwidth Std",
    "High-Freq Energy Mean", "High-Freq Energy Std", "Noise Floor 10th", "Noise Floor Median",
    "Spectral Flux Mean", "Spectral Flux Std", "Autocorr Lag 1", "Autocorr Lag 2",
    "Autocorr Lag 3", "Autocorr Peak Ratio", "LPC Res Var", "LPC Res Kurt",
    "LPC Res Skew", "LPC Res ZCR", "LPC Err Mean", "LPC Err Max"
] + [f"MFCC Mean {i}" for i in range(13)] + [
    "MFCC Std Summary", "MFCC Delta Summary", "MFCC Delta-Delta Summary"
]

VIDEO_FEATURE_NAMES = [
    f"Spatial Agg Mean {i}" for i in range(6)
] + [
    f"Spatial Agg Std {i}" for i in range(6)
] + [
    f"Spatial Agg Max {i}" for i in range(6)
] + [
    f"Spatial Agg Range {i}" for i in range(6)
] + [
    "Frame MAD Diff", "Frame Var Diff", "Frame Skew Diff", "Frame Kurt Diff",
    "Noise Res Inter-Frame Corr Mean", "Noise Res Inter-Frame Corr Std",
    "LSB Flicker Mean", "LSB Flicker Std", "LSB Flicker Max",
    "2nd Order Accel Var", "2nd Order Accel Kurt", "Temp Dynamic Range",
    "Temp Dynamic Var", "Pixel Trajectory Flips", "Pixel Trajectory Std", "Pixel Trajectory Max"
] + [
    f"Audio Track Metric {i}" for i in range(16)
] + [
    "FPS", "Total Frames", "Estimated Duration", "Aspect Ratio",
    "Bytes Per Pixel", "Lossless Flag", "Width Norm", "Height Norm"
]

def get_top_features(
    norm_vec: np.ndarray,
    media_type: str,
    top_k: int = 8
) -> List[Tuple[str, float]]:
    """
    Returns top-k features ranking by absolute deviation in normalized feature space.
    """
    if media_type == "image":
        names = IMAGE_FEATURE_NAMES
    elif media_type == "audio":
        names = AUDIO_FEATURE_NAMES
    else:
        names = VIDEO_FEATURE_NAMES

    k = min(top_k, len(norm_vec))
    indices = np.argsort(-np.abs(norm_vec))[:k]

    result = []
    for idx in indices:
        name = names[idx] if idx < len(names) else f"Feature {idx}"
        val = float(norm_vec[idx])
        result.append((name, val))
    return result
