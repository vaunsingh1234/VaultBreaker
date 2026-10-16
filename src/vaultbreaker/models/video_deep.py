"""Video Temporal Attention Steganalysis Neural Network.

Features:
- Processes K sampled frames using a shared deep spatial residual frame backbone
- Temporal Multi-Pool Aggregation:
    1. Temporal Mean Pooling (steady-state background residue)
    2. Temporal Max Pooling (peak transient pixel modification)
    3. Learned Temporal Self-Attention (focuses dynamically on flickering frames)
- Deep fusion projection into the shared 128-d latent space
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple

from vaultbreaker.models.srnet import SRNetBackbone


class TemporalAttentionPool(nn.Module):
    """Learned temporal attention pooling across K video frames."""
    def __init__(self, feat_dim: int = 128):
        super().__init__()
        self.query = nn.Parameter(torch.randn(feat_dim, 1) * (1.0 / np.sqrt(feat_dim)))
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, K, D)
        Returns: (B, D) attention-weighted summary
        """
        # (B, K, D) -> proj -> (B, K, D)
        h = self.proj(x)
        # Attention scores: (B, K, 1)
        scores = torch.matmul(h, self.query)
        weights = torch.softmax(scores, dim=1)
        # Weighted sum: (B, D)
        out = (x * weights).sum(dim=1)
        return out


class VideoTemporalBackbone(nn.Module):
    """
    Video Steganalysis Network.
    Applies shared frame CNN across K sampled frames, aggregates across time
    using mean, max, and temporal self-attention, and projects to 128-d latent space.
    """
    def __init__(
        self,
        frame_backbone: Optional[nn.Module] = None,
        base_channels: int = 32,
        latent_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # Frame backbone: reuse SRNet or create specialized frame backbone
        if frame_backbone is not None:
            self.frame_backbone = frame_backbone
        else:
            self.frame_backbone = SRNetBackbone(base_channels=base_channels, latent_dim=latent_dim, dropout=dropout)

        # Temporal attention
        self.temporal_attn = TemporalAttentionPool(feat_dim=latent_dim)

        # Temporal fusion head: combines mean (128) + max (128) + attention (128) = 384
        self.fusion_head = nn.Sequential(
            nn.Linear(latent_dim * 3, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames: (B, K, 1, H, W) or (B, K, H, W) or (B, K, 3, H, W)
        Returns: (B, 128) deep video latent representation
        """
        if frames.dim() == 4:
            # (B, K, H, W) -> (B, K, 1, H, W)
            frames = frames.unsqueeze(2)

        B, K, C, H, W = frames.shape

        # Flatten batch and temporal dimensions for frame backbone
        flat_frames = frames.view(B * K, C, H, W)
        flat_feats = self.frame_backbone(flat_frames) # (B * K, D)
        temporal_feats = flat_feats.view(B, K, -1) # (B, K, D)

        # Temporal pooling operations
        mean_pool = temporal_feats.mean(dim=1) # (B, D)
        max_pool = temporal_feats.max(dim=1)[0] # (B, D)
        attn_pool = self.temporal_attn(temporal_feats) # (B, D)

        # Concatenate and fuse
        multi_pool = torch.cat([mean_pool, max_pool, attn_pool], dim=-1) # (B, D * 3)
        z = self.fusion_head(multi_pool) # (B, D)
        return z
