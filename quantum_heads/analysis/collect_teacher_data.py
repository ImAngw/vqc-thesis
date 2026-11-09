"""
Collect (encoder features, teacher logits) pairs from rollouts of a trained teacher policy.

The encoder is frozen: the 16 features it produces are the input of both the teacher head and the student VQC
heads in the distillation. States come from sampled actions (the metric used for best.pt is sample/return_mean).
Seeds are disjoint from eval_seed (10000) and test_seed (20000) of the training configs.

Run:  .venv/bin/python quantum_heads/analysis/collect_teacher_data.py --out quantum_heads/analysis/data
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)
from classic_heads import MLPHead
from rl_train.envs import make_vec_env
from rl_train.policy import Policy
from rl_train.rollout import RolloutCollector, obs_to_tensor


def collect(policy: Policy, n_samples: int, seed: int, n_envs: int = 16, n_steps: int = 16) -> dict:
    collector = RolloutCollector(policy, make_vec_env(n_envs), n_steps, "cpu", seed=seed)
    feats, logits, returns = [], [], []
    while sum(len(f) for f in feats) < n_samples:
        roll = collector.collect()
        with torch.no_grad():
            flat = {k: v.reshape(-1, *v.shape[2:]) for k, v in roll.obs.items()}
            f = policy.encoder(flat["perception"], flat["navigator"])
            feats.append(f.numpy())
            logits.append(policy.head(f).numpy())
        returns += roll.ep_returns
    return {"features": np.concatenate(feats)[:n_samples], "logits": np.concatenate(logits)[:n_samples],
            "collect_return_mean": np.float64(np.mean(returns)), "collect_episodes": np.int64(len(returns))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=os.path.join(ROOT, "logs/mlp128_s0_v2/best.pt"))
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-samples", type=int, default=20000)
    ap.add_argument("--val-samples", type=int, default=5000)
    ap.add_argument("--train-seed", type=int, default=30000)
    ap.add_argument("--val-seed", type=int, default=40000)
    args = ap.parse_args()

    policy = Policy(MLPHead(hidden=args.hidden))
    policy.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    policy.eval()
    os.makedirs(args.out, exist_ok=True)
    for name, n, seed in (("train", args.train_samples, args.train_seed), ("val", args.val_samples, args.val_seed)):
        d = collect(policy, n, seed)
        np.savez(os.path.join(args.out, f"{name}.npz"), **d)
        p = torch.softmax(torch.tensor(d["logits"]), -1)
        ent = float(-(p * p.clamp_min(1e-12).log()).sum(-1).mean())
        f = d["features"]
        print(f"{name}: n={len(f)} feat[min={f.min():.3f} max={f.max():.3f}] teacher entropy={ent:.3f} nats "
              f"(max {np.log(8):.3f}) max-prob mean={float(p.max(-1).values.mean()):.3f} "
              f"collect return={float(d['collect_return_mean']):.2f} over {int(d['collect_episodes'])} episodes")


if __name__ == "__main__":
    main()
