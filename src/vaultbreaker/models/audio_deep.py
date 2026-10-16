"""Raw 1D Waveform Convolutional Neural Network for Audio Steganalysis.

Features:
- Input: Raw 16-bit PCM waveform scaled by 1/32768.0 (no loudness normalization, no resampling)
- Layer 1: 8 frozen 1D difference and high-pass residual filter kernels (1st, 2nd, 3rd, 4th order)
- Early 1D unpooled residual blocks to capture sample-level LSB perturbations and phase discontinuities
- Progressive strided 1D residual blocks for temporal abstraction
- Global Average Pooling -> 128-d latent projection
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional


def get_audio_diff_filters_8() -> torch.Tensor:
    """
    Construct 8 1D difference and high-pass residual kernels (length 5).
    Captures 1st, 2nd, 3rd, and 4th order discrete sample differences.
    """
    kernels: List[np.ndarray] = [
        # 1st order forward & backward
        np.array([0.0, 0.0, -1.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.0, -1.0, 1.0, 0.0, 0.0], dtype=np.float32),
        # 2nd order central difference
        np.array([0.0, -1.0, 2.0, -1.0, 0.0], dtype=np.float32),
        # 3rd order difference
        np.array([-1.0, 3.0, -3.0, 1.0, 0.0], dtype=np.float32),
        # 4th order difference
        np.array([-1.0, 4.0, -6.0, 4.0, -1.0], dtype=np.float32),
        # Alternating high-pass residual
        np.array([-0.5, 1.0, -1.0, 1.0, -0.5], dtype=np.float32),
        # 1D discrete laplacian
        np.array([0.5, -1.0, 1.0, -1.0, 0.5], dtype=np.float32),
        # High-frequency notch
        np.array([0.25, -0.75, 1.0, -0.75, 0.25], dtype=np.float32)
    ]

    norm_kernels = []
    for k in kernels:
        abs_sum = np.sum(np.abs(k))
        if abs_sum > 1e-6:
            k = k / abs_sum
        norm_kernels.append(k)

    # Shape: (8, 1, 5)
    tensor_bank = torch.from_numpy(np.stack(norm_kernels, axis=0)).unsqueeze(1)
    return tensor_bank


class AudioDiffConv1d(nn.Module):
    """Frozen Conv1d layer with 8 difference and high-pass kernels."""
    def __init__(self):
        super().__init__()
        weights = get_audio_diff_filters_8()
        self.conv = nn.Conv1d(1, 8, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv.weight.data.copy_(weights)
        self.conv.weight.requires_grad = False # strictly frozen

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AudioResBlock1d(nn.Module):
    """1D Residual Block with optional striding."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.stride = stride
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += res
        return self.gelu(out)


class Audio1DBackbone(nn.Module):
    """
    Deep 1D CNN for raw audio waveform steganalysis.
    Accepts raw 16-bit audio waveforms scaled by 1/32768.0.
    Outputs a normalized 128-d latent feature vector.
    """
    def __init__(self, base_channels: int = 32, latent_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        c = base_channels
        self.latent_dim = latent_dim

        # Layer 1: Frozen difference filter bank
        self.diff_layer = AudioDiffConv1d()
        self.diff_filter = self.diff_layer

        # Input projection
        self.in_proj = nn.Sequential(
            nn.Conv1d(8, c, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(c),
            nn.GELU()
        )

        # Early unpooled residual blocks (sample-level residual fidelity)
        self.unpooled1 = AudioResBlock1d(c, c, stride=1)
        self.unpooled2 = AudioResBlock1d(c, c, stride=1)

        # Progressive strided blocks (downsampling 48000 -> 12000 -> 3000 -> 750 -> 187)
        self.down1 = AudioResBlock1d(c, c * 2, stride=4)
        self.down2 = AudioResBlock1d(c * 2, c * 2, stride=4)
        self.down3 = AudioResBlock1d(c * 2, c * 4, stride=4)
        self.down4 = AudioResBlock1d(c * 4, c * 4, stride=4)

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Latent projection
        self.head = nn.Sequential(
            nn.Linear(c * 4, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, N) or (B, N) raw 16-bit audio waveform in [-32768, 32767] or float [-1, 1].
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Scale strictly by 1/32768 if in integer range
        if x.abs().max() > 1.5:
            x = x.float() / 32768.0

        # Layer 1: Difference residuals
        diff_feat = self.diff_layer(x)

        # Convolutions
        h = self.in_proj(diff_feat)
        h = self.unpooled1(h)
        h = self.unpooled2(h)
        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)
        h = self.down4(h)

        # Global pooling and projection
        pooled = self.gap(h).flatten(1)
        z = self.head(pooled)
        return z
