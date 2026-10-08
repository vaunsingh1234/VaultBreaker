import torch
import torch.nn as nn
from typing import Optional

class SharedStegoHead(nn.Module):
    """
    Shared binary classification head operating over the common latent space Z.
    Architecture: latent_dim -> 64 -> 1 (returns raw logit; apply sigmoid for probability).
    """
    def __init__(self, latent_dim: int = 128, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 1:
            z = z.unsqueeze(0)
            return self.classifier(z).squeeze(0)
        return self.classifier(z)

class AuxiliaryMethodHead(nn.Module):
    """
    Optional auxiliary multi-class head for identifying the steganographic embedding method.
    Provides multi-task regularization during shared representation learning.
    """
    def __init__(self, latent_dim: int = 128, num_methods: int = 8, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_methods)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 1:
            z = z.unsqueeze(0)
            return self.classifier(z).squeeze(0)
        return self.classifier(z)
