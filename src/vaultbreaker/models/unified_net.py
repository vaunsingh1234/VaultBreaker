import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union

from vaultbreaker.models.encoders import ImageEncoder, AudioEncoder, VideoEncoder
from vaultbreaker.models.shared_head import SharedStegoHead, AuxiliaryMethodHead

# Canonical method class mapping
METHOD_LIST = [
    "clean",
    "lsb_replacement",
    "lsb_matching",
    "edge_adaptive",
    "dct_ac",
    "echo_hiding",
    "frame_lsb",
    "audio_lsb"
]
METHOD_TO_IDX = {m: i for i, m in enumerate(METHOD_LIST)}
IDX_TO_METHOD = {i: m for i, m in enumerate(METHOD_LIST)}

class VaultBreakerUnifiedNet(nn.Module):
    """
    Unified multi-modal neural network architecture.
    Features format-specific encoders mapping to a shared 128-d latent space,
    evaluated by a shared binary classification head and an auxiliary method head.
    """
    def __init__(
        self,
        img_dim: int = 72,
        aud_dim: int = 56,
        vid_dim: int = 64,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        use_aux_head: bool = True,
        num_methods: int = len(METHOD_LIST)
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_aux_head = use_aux_head

        # Encoders
        self.encoders = nn.ModuleDict({
            "image": ImageEncoder(in_dim=img_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, dropout=dropout),
            "audio": AudioEncoder(in_dim=aud_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, dropout=dropout),
            "video": VideoEncoder(in_dim=vid_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, dropout=dropout)
        })

        # Heads
        self.shared_head = SharedStegoHead(latent_dim=latent_dim, hidden_dim=64, dropout=dropout)
        if use_aux_head:
            self.aux_head = AuxiliaryMethodHead(latent_dim=latent_dim, num_methods=num_methods, hidden_dim=64, dropout=dropout)
        else:
            self.aux_head = None

    def encode(self, x: torch.Tensor, media_type: str) -> torch.Tensor:
        """Route input through format-specific encoder into common latent space."""
        if media_type not in self.encoders:
            raise ValueError(f"Unknown media_type '{media_type}'. Valid types: {list(self.encoders.keys())}")
        return self.encoders[media_type](x)

    def forward(
        self,
        x: torch.Tensor,
        media_type: str
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass.
        Returns:
            binary_logits: (B, 1) or scalar
            aux_logits: (B, num_methods) or None
            z: (B, latent_dim) latent representation
        """
        z = self.encode(x, media_type)
        binary_logits = self.shared_head(z)
        aux_logits = self.aux_head(z) if self.aux_head is not None else None
        return binary_logits, aux_logits, z

    def predict_prob(self, x: torch.Tensor, media_type: str) -> float:
        """Compute stego probability score for a single sample."""
        self.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x).float()
            if x.dim() == 1:
                x = x.unsqueeze(0)
            logits, _, _ = self.forward(x, media_type)
            prob = torch.sigmoid(logits).item()
        return float(prob)
