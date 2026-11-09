"""Configuration of an A2C run."""

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any


BEST_MODES = ("greedy", "sample")
HEADS = ("mlp5", "mlp16", "mlp32", "mlp64", "mlp128", "shared", "vqc-optim", "vqc-optim-id", "vqc-optim-id-gray", "vqc-nav-id", "vqc-k", "fourier4")  # policy heads selectable from train.py
PERCEPTIONS = ("linear", "cnn")  # encoder perception branch: Linear (3x3 window) or the XS CNN (5x5 window runs)
BEST_METRICS = ("return_mean", "success_rate", "explored_fraction_mean")  # higher is better


@dataclass
class A2CConfig:
    run_name: str = "a2c"
    log_dir: str = "logs"
    seed: int = 0  # initial weights
    env_seed: int = -1  # envs and action sampling; -1 = same as seed
    device: str = "cpu"
    head: str = "mlp16"  # policy head, one of HEADS
    split_init: int = 1  # 1 = encoder + critic and head initialised from separate streams (build_policy), 0 = legacy order
    encoder_affine: int = 1  # 1 = learned RMSNorm gains in the encoder, 0 = fixed to 1 (int, not bool: see norm_advantage)
    detach_critic: int = 0  # 1 = the critic reads detached features: the value loss does not train the shared encoder
    perception: str = "linear"  # encoder perception branch, one of PERCEPTIONS; its window is env_kwargs["window"] (default 3)
    layout_pool: int = 0  # K > 0: the training envs reuse K fixed layouts (CustomEnv layout_pool); eval stays on fresh ones
    layout_pool_seed: int = -1  # seed of the K training layouts; -1 = the env seed

    # training
    total_updates: int = 2000
    n_envs: int = 16
    n_steps: int = 16  # env steps per env in one rollout; one update per rollout
    gamma: float = 0.99
    lr: float = 7e-4
    head_lr: float = 0.0  # learning rate of the policy head's parameters; 0 = same as lr
    scale_lr: float = 0.0  # learning rate of a VQC head's feature scales (s1, s2, ...); 0 = same as the head
    head_momentum: float = 0.0  # RMSprop momentum of the policy head's parameters (the rest of the model has none)
    value_coef: float = 0.5
    entropy_start: float = 0.02  # entropy coefficient at update 0, annealed linearly (as in QML_Thesis)
    entropy_end: float = 0.001  # coefficient from entropy_anneal_updates on; set equal to entropy_start for a constant one
    entropy_anneal_updates: int = 2500
    target_entropy: float = 0.0  # > 0: adaptive coefficient holding the training entropy at this value (replaces the schedule)
    entropy_lr: float = 0.01  # step of log(entropy coefficient) per update, per nat of entropy error
    max_grad_norm: float = 1.0
    norm_advantage: int = 1  # 1 = standardize advantages per rollout (int, not bool: train.py builds argparse flags from field types)

    # outer step (single-worker DiLoCo): every inner_steps updates, SGD(outer_lr, outer_momentum, Nesterov) on the
    # pseudo-gradient "weights at the start of the round - weights now". outer_lr = 0 turns it off.
    outer_lr: float = 0.0
    outer_momentum: float = 0.0
    outer_dtype: str = "fp32"  # precision of the pseudo-gradient, "fp32" or "bf16" (the report's runs used bf16)
    inner_steps: int = 0  # updates per round; required (> 0) when outer_lr > 0, and total_updates must be a multiple

    # evaluation and test
    eval_every: int = 100  # in updates
    eval_episodes: int = 100
    test_episodes: int = 500
    eval_seed: int = 10_000
    test_seed: int = 20_000

    print_every: int = 0  # print a train line every N updates (0 = silent); eval/test lines are always printed when > 0
    best_metric: str = "sample/return_mean"  # eval metric used to pick best.pt, higher is better

    env_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.env_kwargs, str):  # JSON string, e.g. from a command-line override
            self.env_kwargs = json.loads(self.env_kwargs)
        if not isinstance(self.env_kwargs, dict):
            raise ValueError(f"env_kwargs must be a dict, got {self.env_kwargs!r}")
        if "layout_pool" in self.env_kwargs or "layout_pool_seed" in self.env_kwargs:
            raise ValueError("set the layout pool with the layout_pool / layout_pool_seed fields, not env_kwargs")
        if self.perception not in PERCEPTIONS:
            raise ValueError(f"perception must be one of {PERCEPTIONS}, got {self.perception!r}")
        if self.layout_pool < 0:
            raise ValueError("layout_pool must be >= 0")
        if not 0.0 <= self.target_entropy < math.log(8):
            raise ValueError(f"target_entropy must be in [0, ln 8), got {self.target_entropy}")
        if self.entropy_lr <= 0:
            raise ValueError("entropy_lr must be > 0")
        if self.target_entropy > 0 and self.entropy_start <= 0:
            raise ValueError("an adaptive entropy coefficient starts from entropy_start, which must be > 0")
        if self.outer_lr < 0 or self.outer_momentum < 0 or self.inner_steps < 0:
            raise ValueError("outer_lr, outer_momentum and inner_steps must be >= 0")
        if self.outer_dtype not in ("fp32", "bf16"):
            raise ValueError(f"outer_dtype must be fp32 or bf16, got {self.outer_dtype!r}")
        if self.outer_lr > 0 and (self.inner_steps == 0 or self.total_updates % self.inner_steps != 0):
            raise ValueError("with outer_lr > 0, inner_steps must be > 0 and divide total_updates")
        if self.head not in HEADS:
            raise ValueError(f"head must be one of {HEADS}, got {self.head!r}")
        mode, _, metric = self.best_metric.partition("/")
        if mode not in BEST_MODES or metric not in BEST_METRICS:
            raise ValueError(f"best_metric must be <{'|'.join(BEST_MODES)}>/<{'|'.join(BEST_METRICS)}>, got {self.best_metric!r}")
        eval_seeds = range(self.eval_seed, self.eval_seed + self.eval_episodes)
        test_seeds = range(self.test_seed, self.test_seed + self.test_episodes)
        if set(eval_seeds) & set(test_seeds):
            raise ValueError("eval and test seeds overlap: increase the gap between eval_seed and test_seed")

    @property
    def resolved_env_seed(self) -> int:
        """Seed of the envs and of the action sampling: env_seed, or seed when env_seed is -1."""
        return self.seed if self.env_seed < 0 else self.env_seed

    def train_env_kwargs(self) -> dict[str, Any]:
        """Kwargs of the training envs: env_kwargs plus the layout pool when layout_pool > 0.
        Eval and test envs use env_kwargs alone, i.e. layouts never seen in training."""
        if self.layout_pool == 0:
            return dict(self.env_kwargs)
        pool_seed = self.resolved_env_seed if self.layout_pool_seed < 0 else self.layout_pool_seed
        return {**self.env_kwargs, "layout_pool": self.layout_pool, "layout_pool_seed": pool_seed}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
