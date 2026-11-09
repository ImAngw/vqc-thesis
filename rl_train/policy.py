"""Actor-critic policy: shared encoder -> head (actor, 8 logits) and critic (state value)."""

from typing import Callable

import torch
from torch import Tensor, nn

from classic_heads import MLPHead
from encoder import Encoder
from encoder.encoder import WINDOW


class Policy(nn.Module):
    """obs {"navigator": (B,17), "perception": (B,1,w,w), "memory": (B,k,10)} -> (logits (B,8), value (B,)).
    `head` is any module (B,12) -> (B,8); defaults to MLPHead(hidden=16). `perception` and `window` select the
    encoder's perception branch (Linear on the 3x3 window by default, "cnn" + window 5 for the env-v8 setup)."""

    def __init__(self, head: nn.Module | None = None, critic_hidden: int = 64, encoder_affine: bool = True,
                 perception: str = "linear", window: int = WINDOW):
        super().__init__()
        self.encoder = Encoder(window=window, norm_affine=encoder_affine, perception=perception)
        self.head = head if head is not None else MLPHead(hidden=16)
        self.critic = nn.Sequential(
            nn.Linear(self.encoder.out_features, critic_hidden), nn.ReLU(), nn.Linear(critic_hidden, 1),
        )
        self.detach_critic = False  # True: the critic reads detached features, the value loss does not train the encoder

    def forward(self, obs: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        features = self.encoder(obs["perception"], obs["navigator"], obs["memory"])
        critic_in = features.detach() if self.detach_critic else features
        return self.head(features), self.critic(critic_in).squeeze(-1)


def build_policy(head_factory: Callable[[], nn.Module], critic_hidden: int = 64, encoder_affine: bool = True,
                 perception: str = "linear", window: int = WINDOW) -> Policy:
    """Policy whose encoder + critic and head are initialised from separate streams, both derived from one draw of
    the global RNG: for a given seed the encoder and critic start identical whatever the head, and the head's
    init does not depend on them. The global RNG ends in the same state whatever the head."""
    base = int(torch.randint(0, 2**31 - 1, (1,)).item())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(base)
        policy = Policy(nn.Identity(), critic_hidden, encoder_affine, perception, window)  # parameter-free placeholder head
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(base + 1)
        policy.head = head_factory()
    return policy


def encoder_kwargs(config) -> dict:
    """Encoder options of an A2CConfig, for Policy / build_policy: the window follows the env's."""
    return {"encoder_affine": bool(config.encoder_affine), "perception": config.perception,
            "window": int(config.env_kwargs.get("window", WINDOW))}
