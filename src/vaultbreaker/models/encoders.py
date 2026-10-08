import torch
import torch.nn as nn
from typing import Optional

class FormatEncoder(nn.Module):
    """
    Format-specific MLP encoder mapping handcrafted features
    into a unified latent representation space Z in R^latent_dim.
    Architecture: in_dim -> hidden_dim -> latent_dim with BatchNorm, GELU, and Dropout.
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        dropout: float = 0.2
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Support single sample 1D tensor by expanding batch dim if needed
        if x.dim() == 1:
            x = x.unsqueeze(0)
            return self.net(x).squeeze(0)
        return self.net(x)

class ImageEncoder(FormatEncoder):
    def __init__(self, in_dim: int = 72, hidden_dim: int = 256, latent_dim: int = 128, dropout: float = 0.2):
        super().__init__(in_dim=in_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, dropout=dropout)

class AudioEncoder(FormatEncoder):
    def __init__(self, in_dim: int = 56, hidden_dim: int = 256, latent_dim: int = 128, dropout: float = 0.2):
        super().__init__(in_dim=in_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, dropout=dropout)

class VideoEncoder(FormatEncoder):
    def __init__(self, in_dim: int = 64, hidden_dim: int = 256, latent_dim: int = 128, dropout: float = 0.2):
        super().__init__(in_dim=in_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, dropout=dropout)
