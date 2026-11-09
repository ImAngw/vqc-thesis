"""96-parameter linear head: encoder features -> action logits."""

from torch import Tensor, nn


class SharedHead(nn.Module):
    """Linear(12 -> 8), no bias. Parameters: 12 * 8 = 96."""

    def __init__(self, in_features: int = 12, n_actions: int = 8):
        super().__init__()
        self.linear = nn.Linear(in_features, n_actions, bias=False)

    def forward(self, features: Tensor) -> Tensor:
        return self.linear(features)
