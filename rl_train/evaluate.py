"""Policy evaluation on fixed seeds."""

from typing import Callable

import numpy as np
import torch
from torch import nn

from rl_train.rollout import obs_to_tensor


@torch.no_grad()
def evaluate(
    model: nn.Module,
    env_fn: Callable,
    episodes: int,
    seed0: int,
    greedy: bool,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """Run `episodes` episodes in parallel, one env per seed `seed0 .. seed0 + episodes - 1`.

    All still-running episodes share one batched forward per step, so the cost is ~max episode length
    forwards instead of the total number of steps (a big win for heads with a high per-call overhead).
    `greedy` takes the argmax action, otherwise actions are sampled from the policy (with a generator
    seeded by `seed0`, so repeated calls give identical results).
    The model's train/eval mode is restored afterwards.
    """
    was_training = model.training
    model.eval()
    envs = [env_fn() for _ in range(episodes)]
    generator = torch.Generator().manual_seed(seed0)
    returns, lengths = np.zeros(episodes), np.zeros(episodes, dtype=int)
    successes, explored = np.zeros(episodes, dtype=bool), np.zeros(episodes)
    try:
        obs = [env.reset(seed=seed0 + i)[0] for i, env in enumerate(envs)]
        active = list(range(episodes))
        while active:
            batch = {k: np.stack([obs[i][k] for i in active]) for k in obs[active[0]]}
            logits, _ = model(obs_to_tensor(batch, device))
            if greedy:
                actions = logits.argmax(dim=-1).cpu()
            else:
                actions = torch.multinomial(torch.softmax(logits, dim=-1).cpu(), 1, generator=generator).squeeze(1)
            still_active = []
            for i, action in zip(active, actions.tolist()):
                obs[i], reward, terminated, truncated, info = envs[i].step(action)
                returns[i] += reward
                lengths[i] += 1
                if terminated or truncated:
                    successes[i] = bool(info["is_success"])
                    explored[i] = float(info["explored_fraction"])
                else:
                    still_active.append(i)
            active = still_active
    finally:
        for env in envs:
            env.close()
        model.train(was_training)
    return {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "length_mean": float(np.mean(lengths)),
        "success_rate": float(np.mean(successes)),
        "explored_fraction_mean": float(np.mean(explored)),
    }
