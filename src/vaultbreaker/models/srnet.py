"""SRNet-style Deep Convolutional Neural Network for Image Steganalysis.

Based on Boroumand et al. (2018):
"Deep Residual Network for Steganalysis of Digital Images" (IEEE TIFS).

Features:
- Layer 1: 30 frozen SRM (Spatial Rich Model) high-pass linear filter kernels
- Early Blocks (Layers 2-7): Unpooled residual blocks (no pooling to preserve weak stego signals)
- Late Blocks (Layers 8-11): Average-pooled residual blocks for spatial abstraction
- Global Average Pooling -> 128-d latent projection
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, List, Optional


def get_srm_filter_bank_30() -> torch.Tensor:
    """
    Construct standard 30 5x5 SRM high-pass residual filter kernels.
    Includes 1st, 2nd, 3rd, and 4th order derivative kernels, laplacian,
    edge, and corner residual detectors.
    """
    kernels: List[np.ndarray] = []

    # 1. 1st-order horizontal, vertical, and diagonals (4 kernels)
    k1_h = np.zeros((5, 5), dtype=np.float32)
    k1_h[2, 2] = -1.0; k1_h[2, 3] = 1.0
    k1_v = np.zeros((5, 5), dtype=np.float32)
    k1_v[2, 2] = -1.0; k1_v[3, 2] = 1.0
    k1_d1 = np.zeros((5, 5), dtype=np.float32)
    k1_d1[2, 2] = -1.0; k1_d1[3, 3] = 1.0
    k1_d2 = np.zeros((5, 5), dtype=np.float32)
    k1_d2[2, 2] = -1.0; k1_d2[3, 1] = 1.0
    kernels.extend([k1_h, k1_v, k1_d1, k1_d2])

    # 2. 2nd-order horizontal, vertical, and diagonals (4 kernels)
    k2_h = np.zeros((5, 5), dtype=np.float32)
    k2_h[2, 1] = -1.0; k2_h[2, 2] = 2.0; k2_h[2, 3] = -1.0
    k2_v = np.zeros((5, 5), dtype=np.float32)
    k2_v[1, 2] = -1.0; k2_v[2, 2] = 2.0; k2_v[3, 2] = -1.0
    k2_d1 = np.zeros((5, 5), dtype=np.float32)
    k2_d1[1, 1] = -1.0; k2_d1[2, 2] = 2.0; k2_d1[3, 3] = -1.0
    k2_d2 = np.zeros((5, 5), dtype=np.float32)
    k2_d2[1, 3] = -1.0; k2_d2[2, 2] = 2.0; k2_d2[3, 1] = -1.0
    kernels.extend([k2_h, k2_v, k2_d1, k2_d2])

    # 3. 3rd-order horizontal and vertical (4 kernels)
    k3_h1 = np.zeros((5, 5), dtype=np.float32)
    k3_h1[2, 0:4] = [-1.0, 3.0, -3.0, 1.0]
    k3_h2 = np.zeros((5, 5), dtype=np.float32)
    k3_h2[2, 1:5] = [-1.0, 3.0, -3.0, 1.0]
    k3_v1 = np.zeros((5, 5), dtype=np.float32)
    k3_v1[0:4, 2] = [-1.0, 3.0, -3.0, 1.0]
    k3_v2 = np.zeros((5, 5), dtype=np.float32)
    k3_v2[1:5, 2] = [-1.0, 3.0, -3.0, 1.0]
    kernels.extend([k3_h1, k3_h2, k3_v1, k3_v2])

    # 4. 4th-order horizontal and vertical (2 kernels)
    k4_h = np.zeros((5, 5), dtype=np.float32)
    k4_h[2, :] = [-1.0, 4.0, -6.0, 4.0, -1.0]
    k4_v = np.zeros((5, 5), dtype=np.float32)
    k4_v[:, 2] = [-1.0, 4.0, -6.0, 4.0, -1.0]
    kernels.extend([k4_h, k4_v])

    # 5. 3x3 Laplacian and cross kernels (4 kernels)
    lap3 = np.zeros((5, 5), dtype=np.float32)
    lap3[1:4, 1:4] = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    lap3_diag = np.zeros((5, 5), dtype=np.float32)
    lap3_diag[1:4, 1:4] = [[1, 0, 1], [0, -4, 0], [1, 0, 1]]
    lap3_full = np.zeros((5, 5), dtype=np.float32)
    lap3_full[1:4, 1:4] = [[1, 1, 1], [1, -8, 1], [1, 1, 1]]
    cross3 = np.zeros((5, 5), dtype=np.float32)
    cross3[1:4, 1:4] = [[-1, 2, -1], [2, -4, 2], [-1, 2, -1]]
    kernels.extend([lap3, lap3_diag, lap3_full, cross3])

    # 6. Edge and Corner filters (4 kernels)
    edge_h = np.zeros((5, 5), dtype=np.float32)
    edge_h[1:4, 1:4] = [[-1, -1, -1], [2, 2, 2], [-1, -1, -1]]
    edge_v = np.zeros((5, 5), dtype=np.float32)
    edge_v[1:4, 1:4] = [[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]]
    corner_ul = np.zeros((5, 5), dtype=np.float32)
    corner_ul[1:4, 1:4] = [[2, -1, 0], [-1, 0, 0], [0, 0, 0]]
    corner_ur = np.zeros((5, 5), dtype=np.float32)
    corner_ur[1:4, 1:4] = [[0, -1, 2], [0, 0, -1], [0, 0, 0]]
    kernels.extend([edge_h, edge_v, corner_ul, corner_ur])

    # 7. 5x5 High-order SRM kernels (8 kernels)
    srm_square = np.array([
        [-1, 2, -2, 2, -1],
        [ 2, -6, 8, -6,  2],
        [-2,  8,-12, 8, -2],
        [ 2, -6, 8, -6,  2],
        [-1,  2, -2, 2, -1]
    ], dtype=np.float32)

    srm_circ = np.array([
        [0,  1,  1,  1, 0],
        [1,  2, -4,  2, 1],
        [1, -4,  8, -4, 1],
        [1,  2, -4,  2, 1],
        [0,  1,  1,  1, 0]
    ], dtype=np.float32)

    srm_diag5_1 = np.array([
        [-1, 0, 0, 0, 0],
        [ 0, 2, 0, 0, 0],
        [ 0, 0,-2, 0, 0],
        [ 0, 0, 0, 2, 0],
        [ 0, 0, 0, 0,-1]
    ], dtype=np.float32)

    srm_diag5_2 = np.fliplr(srm_diag5_1)
    srm_cross5 = np.zeros((5, 5), dtype=np.float32)
    srm_cross5[2, :] = [-1, 2, -2, 2, -1]
    srm_cross5[:, 2] += [-1, 2, -2, 2, -1]
    srm_cross5[2, 2] = -4.0

    srm_kb = np.array([
        [-1,  2, -1,  0, 0],
        [ 2, -4,  2,  0, 0],
        [-1,  2, -1,  0, 0],
        [ 0,  0,  0,  0, 0],
        [ 0,  0,  0,  0, 0]
    ], dtype=np.float32)

    srm_kb_rot1 = np.rot90(srm_kb, 1)
    srm_kb_rot2 = np.rot90(srm_kb, 2)

    kernels.extend([srm_square, srm_circ, srm_diag5_1, srm_diag5_2, srm_cross5, srm_kb, srm_kb_rot1, srm_kb_rot2])

    assert len(kernels) == 30, f"Expected 30 kernels, constructed {len(kernels)}"

    # Normalize each filter so sum of absolute weights = 1.0 or sum = 0
    norm_kernels = []
    for k in kernels:
        abs_sum = np.sum(np.abs(k))
        if abs_sum > 1e-6:
            k = k / abs_sum
        norm_kernels.append(k)

    tensor_bank = torch.from_numpy(np.stack(norm_kernels, axis=0)).unsqueeze(1) # (30, 1, 5, 5)
    return tensor_bank


class SRMConv2d(nn.Module):
    """Frozen Conv2d layer initialized with 30 Spatial Rich Model high-pass filters."""
    def __init__(self):
        super().__init__()
        weights = get_srm_filter_bank_30()
        self.conv = nn.Conv2d(1, 30, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv.weight.data.copy_(weights)
        self.conv.weight.requires_grad = False # strictly frozen

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UnpooledResBlock(nn.Module):
    """SRNet Type 1 Block: Unpooled residual block (preserves pixel-level stego residuals)."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class PooledResBlock(nn.Module):
    """SRNet Type 2 Block: Average-pooled residual block for progressive spatial abstraction."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.pool(out)
        out += residual
        return self.relu(out)


class SRNetBackbone(nn.Module):
    """
    SRNet deep convolutional neural network for 256x256 grayscale image steganalysis.
    Outputs a normalized 128-d latent feature vector.
    """
    def __init__(self, base_channels: int = 32, latent_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        c = base_channels
        self.latent_dim = latent_dim

        # Layer 1: Frozen SRM filter bank
        self.srm_layer = SRMConv2d()

        # Layers 2-4: Unpooled residual blocks (256x256 preserved!)
        self.block1 = nn.Sequential(
            nn.Conv2d(30, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True)
        )
        self.unpooled1 = UnpooledResBlock(c, c)
        self.unpooled2 = UnpooledResBlock(c, c)
        self.unpooled3 = UnpooledResBlock(c, c * 2)

        # Layers 5-8: Pooled residual blocks (progressive 256 -> 128 -> 64 -> 32 -> 16)
        self.pooled1 = PooledResBlock(c * 2, c * 2) # -> 128x128
        self.pooled2 = PooledResBlock(c * 2, c * 4) # -> 64x64
        self.pooled3 = PooledResBlock(c * 4, c * 4) # -> 32x32
        self.pooled4 = PooledResBlock(c * 4, c * 8) # -> 16x16

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Latent projection
        self.head = nn.Sequential(
            nn.Linear(c * 8, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, 256, 256) in [0, 255] or float [-1, 1].
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.size(1) == 3:
            # Grayscale conversion if 3-channel RGB: Y = 0.299R + 0.587G + 0.114B
            x = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

        # Standardize scale: ensure centered around 0
        if x.max() > 1.5: # assuming [0, 255]
            x = (x - 128.0) / 128.0

        # Layer 1: SRM Residuals
        srm_feat = self.srm_layer(x)

        # Unpooled layers
        h = self.block1(srm_feat)
        h = self.unpooled1(h)
        h = self.unpooled2(h)
        h = self.unpooled3(h)

        # Pooled layers
        h = self.pooled1(h)
        h = self.pooled2(h)
        h = self.pooled3(h)
        h = self.pooled4(h)

        # Global pooling & latent
        pooled = self.gap(h).flatten(1)
        z = self.head(pooled)
        return z
