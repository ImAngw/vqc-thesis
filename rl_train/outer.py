"""Outer step of a single-worker DiLoCo-style schedule: every `inner_steps` updates the weights are moved from the
round's starting point (the "global" weights) towards where the inner optimizer ended, with an SGD step whose
gradient is the pseudo-gradient `global - local` (Nesterov momentum when `momentum` > 0)."""

import torch
from torch import nn


def pack(params: list[nn.Parameter]) -> torch.Tensor:
    """All parameters as one flat vector (a copy)."""
    return torch.cat([p.data.reshape(-1) for p in params])


def unpack(vector: torch.Tensor, params: list[nn.Parameter]) -> None:
    """Copy a flat vector back into the parameters, in place."""
    offset = 0
    for p in params:
        n = p.numel()
        p.data.copy_(vector[offset:offset + n].view_as(p))
        offset += n


class OuterStep:
    """Keeps the global weights and the outer momentum; `step` closes a round."""

    def __init__(self, params: list[nn.Parameter], lr: float, momentum: float, dtype: str = "fp32"):
        self.params = params
        self.global_params = nn.Parameter(pack(params))
        self.opt = torch.optim.SGD([self.global_params], lr=lr, momentum=momentum, nesterov=momentum > 0)
        self.dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[dtype]

    def step(self) -> None:
        """Pseudo-gradient = global - current weights, rounded to `dtype` (bf16 reproduces the multi-worker runtime
        used for the report, which exchanges pseudo-gradients in bf16 even with a single worker); one outer SGD step;
        the model continues from the new global weights."""
        self.global_params.grad = (self.global_params.data - pack(self.params)).to(self.dtype).to(torch.float32)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        unpack(self.global_params.data, self.params)
