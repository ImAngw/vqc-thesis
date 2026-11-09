"""Fully connected head: encoder features -> action logits."""

from torch import nn


class MLPHead(nn.Sequential):
    """Linear(12 -> hidden) -> ReLU -> Linear(hidden -> 8). Parameters: 12h + h + 8h + 8 (h=16: 344)."""

    def __init__(self, hidden: int, in_features: int = 12, n_actions: int = 8):
        super().__init__(nn.Linear(in_features, hidden), nn.ReLU(), nn.Linear(hidden, n_actions))
