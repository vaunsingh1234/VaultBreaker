"""Unified Deep Steganography Neural Network (VaultBreakerDeepNet).

Combines:
- Image: SRNet-style CNN with 30 frozen SRM filters
- Audio: 1D Raw Waveform CNN with frozen difference filters
- Video: Temporal Attention CNN across K sampled frames
- Multi-Modal Handcrafted Feature Fusion branches
- Shared 128-d latent space
- Shared binary classification head and auxiliary method head
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Union

from vaultbreaker.models.srnet import SRNetBackbone
from vaultbreaker.models.audio_deep import Audio1DBackbone
from vaultbreaker.models.video_deep import VideoTemporalBackbone
from vaultbreaker.models.shared_head import SharedStegoHead, AuxiliaryMethodHead
from vaultbreaker.models.unified_net import METHOD_LIST, METHOD_TO_IDX, IDX_TO_METHOD


class VaultBreakerDeepNet(nn.Module):
    """
    Unified multi-modal deep neural network architecture for steganography detection.
    Maps format-specialized deep representations into a unified 128-d latent space.
    """
    def __init__(
        self,
        img_dim: int = 72,
        aud_dim: int = 56,
        vid_dim: int = 64,
        latent_dim: int = 128,
        base_channels: int = 32,
        dropout: float = 0.2,
        use_aux_head: bool = True,
        use_fusion: bool = True,
        num_methods: int = len(METHOD_LIST)
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_aux_head = use_aux_head
        self.use_fusion = use_fusion

        # 1. Specialized Deep Backbones
        self.image_deep = SRNetBackbone(base_channels=base_channels, latent_dim=latent_dim, dropout=dropout * 0.5)
        self.audio_deep = Audio1DBackbone(base_channels=base_channels, latent_dim=latent_dim, dropout=dropout * 0.5)
        self.video_deep = VideoTemporalBackbone(base_channels=base_channels, latent_dim=latent_dim, dropout=dropout * 0.5)

        # 2. Modality Fusion Branches (Deep + Handcrafted -> 128-d latent)
        self.fusion = nn.ModuleDict({
            "image": nn.Sequential(
                nn.Linear(latent_dim + img_dim, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(latent_dim, latent_dim)
            ),
            "audio": nn.Sequential(
                nn.Linear(latent_dim + aud_dim, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(latent_dim, latent_dim)
            ),
            "video": nn.Sequential(
                nn.Linear(latent_dim + vid_dim, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(latent_dim, latent_dim)
            )
        })

        # 3. Shared Heads
        self.shared_head = SharedStegoHead(latent_dim=latent_dim, hidden_dim=64, dropout=dropout)
        if use_aux_head:
            self.aux_head = AuxiliaryMethodHead(latent_dim=latent_dim, num_methods=num_methods, hidden_dim=64, dropout=dropout)
        else:
            self.aux_head = None

    def encode(
        self,
        x: torch.Tensor,
        media_type: str,
        hc_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Route input through format-specialized backbone into unified 128-d latent space."""
        if media_type == "image":
            deep_z = self.image_deep(x)
        elif media_type == "audio":
            deep_z = self.audio_deep(x)
        elif media_type == "video":
            deep_z = self.video_deep(x)
        else:
            raise ValueError(f"Unknown media_type '{media_type}'")

        if self.use_fusion and hc_features is not None:
            if hc_features.dim() == 1:
                hc_features = hc_features.unsqueeze(0)
            fused_in = torch.cat([deep_z, hc_features], dim=-1)
            z = self.fusion[media_type](fused_in)
        else:
            z = deep_z

        return z

    def forward(
        self,
        x: torch.Tensor,
        media_type: str,
        hc_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass.
        Returns:
            binary_logits: (B, 1) stego decision logits
            aux_logits: (B, num_methods) auxiliary classification logits or None
            z: (B, 128) unified latent space representation
        """
        z = self.encode(x, media_type, hc_features=hc_features)
        binary_logits = self.shared_head(z)
        aux_logits = self.aux_head(z) if self.aux_head is not None else None
        return binary_logits, aux_logits, z

    def predict_prob(
        self,
        x: torch.Tensor,
        media_type: str,
        hc_features: Optional[torch.Tensor] = None
    ) -> float:
        """Compute calibrated probability score for a single inference sample."""
        self.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x).float()
            if media_type == "image" and x.dim() == 2:
                x = x.unsqueeze(0).unsqueeze(0)
            elif media_type == "image" and x.dim() == 3 and x.size(0) != 1:
                x = x.unsqueeze(0)
            elif media_type == "audio" and x.dim() == 1:
                x = x.unsqueeze(0).unsqueeze(0)
            elif media_type == "video" and x.dim() == 4:
                x = x.unsqueeze(0)

            if hc_features is not None and not isinstance(hc_features, torch.Tensor):
                hc_features = torch.from_numpy(hc_features).float()

            logits, _, _ = self.forward(x, media_type, hc_features=hc_features)
            prob = torch.sigmoid(logits).item()
        return float(prob)
