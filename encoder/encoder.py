"""Encoder shared by all heads: perception linear (4 features) + navigator MLP (4) + memory GRU (4), RMS-normalized."""

import torch
from torch import Tensor, nn

WINDOW = 3  # perception window side of the env (v10: 3x3)
NAV_DIM = 17
MEMORY_DIM = 10  # action one-hot (8) + displacement (dx, dy)
FEATURES = 4  # features per branch


def _norm(features: int, on: bool, affine: bool) -> list[nn.Module]:
    return [nn.RMSNorm(features, elementwise_affine=affine)] if on else []


class Encoder(nn.Module):
    """Three branches concatenated as [perception (4) | navigator (4) | memory (4)] -> 12 features.

    perception (B,1,3,3) -> Flatten -> Linear(9 -> 4) -> [RMSNorm] -> Tanh                     (perception="linear")
               (B,1,w,w) -> conv3x3 (1->c1) -> ReLU -> conv3x3 (c1->c2, stride 2) -> ReLU
                            -> Linear -> [RMSNorm] -> Tanh                                       (perception="cnn")
    navigator  (B,17)    -> Linear(17 -> 4) -> [RMSNorm] -> Tanh
    memory     (B,k,10)  -> GRU(10 -> 4) over the k moves (oldest first, zero initial state), last hidden state
                            -> [RMSNorm -> Tanh]
    perception="cnn" with channels 2->2 is the "XS" encoder of the thesis experiments; it is kept to load
    checkpoints trained on the 5x5 window (env <= v9): Encoder(perception="cnn", window=5). `channels` is
    ignored by the linear branch.

    `norm` (default on) adds a per-branch RMSNorm before the final Tanh, so the pre-activation scale cannot grow
    with the weights and saturate the 4 features (seen in the navigator of 12-feature runs). RMSNorm rather than
    LayerNorm: it removes only the scale, keeping 3 of 4 degrees of freedom per branch instead of 2.
    norm=False is the previous encoder, needed to load checkpoints trained before the change.
    `norm_affine=False` fixes the RMSNorm gains to 1: with learned gains the navigator ones grow to ~1.8 during
    the entropy annealing and the features saturate again (2-4k updates), before shrinking below 1.
    """

    def __init__(self, window: int = WINDOW, channels: tuple[int, int] = (2, 2), features: int = FEATURES,
                 norm: bool = True, norm_affine: bool = True, perception: str = "linear"):
        super().__init__()
        if perception == "linear":
            self.perception = nn.Sequential(
                nn.Flatten(),
                nn.Linear(window * window, features), *_norm(features, norm, norm_affine), nn.Tanh(),
            )
        elif perception == "cnn":
            c1, c2 = channels
            out_side = (window - 1) // 2 + 1  # side of the conv2 output (stride 2, padding 1)
            self.perception = nn.Sequential(
                nn.Conv2d(1, c1, 3, padding=1), nn.ReLU(),
                nn.Conv2d(c1, c2, 3, padding=1, stride=2), nn.ReLU(),
                nn.Flatten(),
                nn.Linear(c2 * out_side * out_side, features), *_norm(features, norm, norm_affine), nn.Tanh(),
            )
        else:
            raise ValueError(f"perception must be 'linear' or 'cnn', got {perception!r}")
        self.navigator = nn.Sequential(nn.Linear(NAV_DIM, features), *_norm(features, norm, norm_affine), nn.Tanh())
        self.memory = nn.GRU(MEMORY_DIM, features, batch_first=True)
        self.memory_out = nn.Sequential(*_norm(features, True, norm_affine), nn.Tanh()) if norm else nn.Identity()
        self.out_features = 3 * features

    def forward(self, perception: Tensor, navigator: Tensor, memory: Tensor) -> Tensor:
        h = self.memory_out(self.memory(memory)[1][0])  # (B, features)
        return torch.cat([self.perception(perception), self.navigator(navigator), h], dim=-1)
