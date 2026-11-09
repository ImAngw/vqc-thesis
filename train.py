"""CLI entry point: train the encoder + head + critic policy with A2C."""

import argparse
import dataclasses
import json
import math

import torch

from rl_train.a2c import A2CTrainer
from classic_heads import FourierHead, MLPHead, SharedHead
from quantum_heads.vqc_head_i import VQCHeadI
from quantum_heads.vqc_head_j import VQCHeadJ
from quantum_heads.vqc_head_k import VQCHeadK
from rl_train.config import HEADS as HEAD_NAMES, A2CConfig
from rl_train.policy import Policy, build_policy, encoder_kwargs

HEADS = {"mlp5": lambda: MLPHead(hidden=5), "mlp16": lambda: MLPHead(hidden=16), "mlp32": lambda: MLPHead(hidden=32), "mlp64": lambda: MLPHead(hidden=64), "mlp128": lambda: MLPHead(hidden=128), "shared": SharedHead, "vqc-optim": VQCHeadI, "vqc-optim-id": lambda: VQCHeadI(scale_init=0.1, angle_std=0.05, wf_offset=math.pi / 2), "vqc-optim-id-gray": lambda: VQCHeadI(scale_init=0.1, angle_std=0.05, wf_offset=math.pi / 2, action_code="gray"), "vqc-nav-id": lambda: VQCHeadJ(scale_init=0.05, angle_std=0.05, wf_offset=math.pi / 2, action_code="gray"), "vqc-k": lambda: VQCHeadK(scale_init=0.05, angle_std=0.03, wf_offset=math.pi / 2, action_code="gray"), "fourier4": lambda: FourierHead(n_freq=4)}


def parse_args() -> A2CConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = A2CConfig()
    for f in dataclasses.fields(A2CConfig):
        if f.name in ("env_kwargs", "head"):
            continue
        flag = "--" + f.name.replace("_", "-")
        parser.add_argument(flag, type=type(getattr(defaults, f.name)), default=getattr(defaults, f.name))
    parser.add_argument("--env-kwargs", type=json.loads, default={}, help='JSON dict passed to CustomEnv, e.g. \'{"max_steps": 200}\'')
    parser.add_argument("--head", choices=HEAD_NAMES, default=defaults.head, help="policy head")
    args = vars(parser.parse_args())
    return A2CConfig(**args)


def main() -> None:
    cfg = parse_args()
    torch.set_num_threads(1)  # tensors are tiny: extra threads only add overhead (update ~4x faster on CPU)
    torch.manual_seed(cfg.seed)  # before the model is built, so weight init is seeded too
    if cfg.split_init:
        policy = build_policy(HEADS[cfg.head], **encoder_kwargs(cfg))
    else:  # legacy: the head is built first, so the encoder and critic init depend on the head
        policy = Policy(HEADS[cfg.head](), **encoder_kwargs(cfg))
    trainer = A2CTrainer(policy, cfg)
    print(f"run dir: {trainer.logger.run_dir}")
    trainer.train()
    print("done")


if __name__ == "__main__":
    main()
