"""Bootstrapped n-step returns."""

import torch
from torch import Tensor


def n_step_returns(
    rewards: Tensor,
    terminated: Tensor,
    truncated: Tensor,
    final_values: Tensor,
    last_value: Tensor,
    gamma: float,
) -> Tensor:
    """Bootstrapped n-step returns for a rollout of shape (T, N).

    `rewards[t]`, `terminated[t]`, `truncated[t]` come from the step taken at time t.
    `final_values[t]` is V(terminal obs) and is only read where `truncated[t]` is True.
    `last_value` (N,) is V of the obs after the last step. The return is cut at episode ends:
    bootstrap is 0 on termination, `final_values[t]` on truncation, the next return otherwise.
    """
    returns = torch.zeros_like(rewards)
    next_return = last_value
    for t in reversed(range(rewards.shape[0])):
        bootstrap = torch.where(truncated[t], final_values[t], next_return)
        bootstrap = torch.where(terminated[t], torch.zeros_like(bootstrap), bootstrap)
        next_return = rewards[t] + gamma * bootstrap
        returns[t] = next_return
    return returns
