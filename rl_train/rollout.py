"""Rollout collection from a vector env with SAME_STEP autoreset."""

from dataclasses import dataclass, field

import numpy as np
import torch
from gymnasium.vector import SyncVectorEnv
from torch import Tensor, nn


def obs_to_tensor(obs: dict[str, np.ndarray], device: torch.device | str = "cpu") -> dict[str, Tensor]:
    """Batched numpy dict observation -> float32 tensors on `device`."""
    return {k: torch.as_tensor(np.asarray(v), dtype=torch.float32, device=device) for k, v in obs.items()}


@dataclass
class Rollout:
    """Transitions of shape (T, N, ...). `final_values[t]` is V(terminal obs), meaningful only where truncated."""

    obs: dict[str, Tensor]
    actions: Tensor
    rewards: Tensor
    terminated: Tensor
    truncated: Tensor
    final_values: Tensor
    last_value: Tensor  # (N,) V of the obs after the last step
    # stats of the episodes that finished during this rollout, evaluated at episode end
    ep_returns: list[float] = field(default_factory=list)
    ep_lengths: list[int] = field(default_factory=list)
    ep_explored: list[float] = field(default_factory=list)


class RolloutCollector:
    """Keeps the vector env running across rollouts: envs are never reset between updates."""

    def __init__(self, model: nn.Module, vec_env: SyncVectorEnv, n_steps: int, device: torch.device | str = "cpu", seed: int | None = None):
        self.model = model
        self.env = vec_env
        self.n_steps = n_steps
        self.device = device
        self.n_envs = vec_env.num_envs
        self.obs, _ = vec_env.reset(seed=seed)
        # own generator: action sampling does not depend on (or disturb) the global torch RNG
        self.generator = torch.Generator()
        self.generator.manual_seed(seed if seed is not None else torch.seed())
        self.running_return = np.zeros(self.n_envs)
        self.running_length = np.zeros(self.n_envs, dtype=int)

    @torch.no_grad()
    def collect(self) -> Rollout:
        T, N = self.n_steps, self.n_envs
        obs_buf = {k: [] for k in self.obs}
        actions = torch.zeros(T, N, dtype=torch.long)
        rewards = torch.zeros(T, N)
        terminated = torch.zeros(T, N, dtype=torch.bool)
        truncated = torch.zeros(T, N, dtype=torch.bool)
        final_values = torch.zeros(T, N)
        stats = Rollout({}, actions, rewards, terminated, truncated, final_values, torch.zeros(N))

        for t in range(T):
            obs_t = obs_to_tensor(self.obs, self.device)
            for k, v in obs_t.items():
                obs_buf[k].append(v.cpu())
            logits, _ = self.model(obs_t)
            action = torch.multinomial(torch.softmax(logits, dim=-1).cpu(), 1, generator=self.generator).squeeze(1)
            self.obs, reward, term, trunc, info = self.env.step(action.cpu().numpy())

            actions[t] = action.cpu()
            rewards[t] = torch.as_tensor(reward, dtype=torch.float32)
            terminated[t] = torch.as_tensor(term)
            truncated[t] = torch.as_tensor(trunc)
            self.running_return += reward
            self.running_length += 1

            done = term | trunc
            if done.any():
                self._on_episodes_end(t, done, term, trunc, info, final_values, stats)

        stats.obs = {k: torch.stack(v) for k, v in obs_buf.items()}
        _, stats.last_value = self.model(obs_to_tensor(self.obs, self.device))
        stats.last_value = stats.last_value.cpu()
        return stats

    def _on_episodes_end(self, t, done, term, trunc, info, final_values, stats: Rollout) -> None:
        for i in np.flatnonzero(done):
            stats.ep_returns.append(float(self.running_return[i]))
            stats.ep_lengths.append(int(self.running_length[i]))
            stats.ep_explored.append(float(info["final_info"]["explored_fraction"][i]))
        self.running_return[done] = 0.0
        self.running_length[done] = 0

        timed_out = np.flatnonzero(trunc & ~term)  # terminated wins over truncated: bootstrap is 0 there
        if len(timed_out):
            final_obs = {k: np.stack([info["final_obs"][i][k] for i in timed_out]) for k in self.obs}
            _, values = self.model(obs_to_tensor(final_obs, self.device))
            final_values[t, torch.as_tensor(timed_out)] = values.cpu()
