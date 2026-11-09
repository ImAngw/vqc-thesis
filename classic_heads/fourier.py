"""Fourier-feature head: encoder features -> learned frequencies -> sin/cos -> action logits."""

import torch
from torch import Tensor, nn


class FourierHead(nn.Module):
    """z = x W^T (W: n_freq x 12, learned) -> [sin z, cos z] -> Linear(2 n_freq -> 8).
    Parameters: 12 n + 2n * 8 + 8 (n=4: 120, about J's 117). The classical counterpart of the periodic structure of a
    re-uploading VQC (its Born probabilities are trigonometric polynomials of learned linear maps of the features),
    without the Born readout. W is initialised randn * freq_std (random Fourier features); the output layer as nn.Linear."""

    def __init__(self, n_freq: int = 4, freq_std: float = 1.0, in_features: int = 12, n_actions: int = 8):
        super().__init__()
        self.freq = nn.Parameter(torch.randn(n_freq, in_features) * freq_std)
        self.out = nn.Linear(2 * n_freq, n_actions)

    def forward(self, features: Tensor) -> Tensor:
        z = features @ self.freq.T
        return self.out(torch.cat([torch.sin(z), torch.cos(z)], dim=-1))
