"""
Evaluate distilled students (saved by distill.py --save-students) in the environment.

Each student (encoder + head) is wrapped in rl_train.policy.Policy (the critic stays at init: evaluation only uses the
logits) and run with rl_train.evaluate.evaluate on the test seeds of the training configs, so the returns compare
directly with test/sample/return_mean of the RL runs. Results are grouped by (variant, head, lr), mean and std over seeds.

Run:  .venv/bin/python quantum_heads/analysis/eval_students.py quantum_heads/analysis/runs/<run>/students
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import os
import sys

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)
from quantum_heads.analysis.distill import ENV_KW, HEADS, POLICY_KW  # noqa: E402
from rl_train.envs import make_env  # noqa: E402
from rl_train.evaluate import evaluate  # noqa: E402
from rl_train.policy import Policy  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("students_dir")
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--seed0", type=int, default=20_000, help="test_seed of A2CConfig")
    args = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.students_dir, "*.pt"))):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        policy = Policy(HEADS[ckpt["head_name"]](), **POLICY_KW)
        policy.encoder.load_state_dict(ckpt["encoder"])
        policy.head.load_state_dict(ckpt["head"])
        name = os.path.basename(path)[:-3]
        variant, head, seed, lr = name.split("_")
        row = {"variant": variant, "head": head, "seed": int(seed[1:]), "lr": lr[2:]}
        for mode, greedy in (("sample", False), ("greedy", True)):
            r = evaluate(policy, lambda: make_env(**ENV_KW), args.episodes, args.seed0, greedy)
            row.update({f"{mode}_{k}": v for k, v in r.items()})
        rows.append(row)
        print(f"{name}: sample return {row['sample_return_mean']:.3f} success {row['sample_success_rate']:.2f} "
              f"| greedy return {row['greedy_return_mean']:.3f}", flush=True)

    with open(os.path.join(args.students_dir, "..", "eval_students.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["variant"], r["head"], r["lr"])].append(r)
    print(f"\n| variant | head | lr | sample return | sample success | greedy return |  ({args.episodes} test episodes, mean±std over seeds)")
    print("|---|---|---|---|---|---|")
    for (variant, head, lr), rs in sorted(groups.items()):
        ms = lambda k: f"{np.mean([r[k] for r in rs]):.3f}±{np.std([r[k] for r in rs]):.3f}"  # noqa: E731
        print(f"| {variant} | {head} | {lr} | {ms('sample_return_mean')} | {ms('sample_success_rate')} | {ms('greedy_return_mean')} |")


if __name__ == "__main__":
    main()
