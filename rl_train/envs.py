"""Environment factories for training and evaluation."""

from typing import Any

from gymnasium.vector import AutoresetMode, SyncVectorEnv

from rl_env.env import CustomEnv


def make_env(**env_kwargs: Any) -> CustomEnv:
    """Single env; seeding is done through `reset(seed=...)`."""
    return CustomEnv(**env_kwargs)


def make_vec_env(n_envs: int, **env_kwargs: Any) -> SyncVectorEnv:
    """`n_envs` envs in one process. With SAME_STEP autoreset, a finished episode returns the
    first obs of the next one, and the terminal obs/info are in `info["final_obs"]`/`info["final_info"]`."""
    return SyncVectorEnv(
        [lambda: make_env(**env_kwargs) for _ in range(n_envs)],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
