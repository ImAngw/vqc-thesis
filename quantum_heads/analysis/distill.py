"""
Distillation probe: can the VQC heads I (vqc-optim-id, vqc-optim-id-gray) and J (vqc-nav-id) represent the policy of a
trained MLP teacher (mlp64 or mlp128)?

Supervised, no RL: a student (teacher encoder + fresh head) minimizes KL(teacher || student) on states visited by
the teacher. Two variants separate input scale from circuit capacity:
  frozen    the teacher encoder is frozen, only the head is trained (features precomputed once);
  finetune  the encoder starts from the teacher and is trained together with the head.
Students: the VQC heads and MLPHead(64) (the reference ceiling), several init seeds, a small head-lr grid.

Metric: explained fraction = 1 - KL(teacher || student) / KL(teacher || teacher marginal), on the val split.
Verdict (fixed before looking at the data): capacity limit if, in the same variant, mean+std of a VQC head's fraction is
below 50% of (mean-std) of MLP64's fraction.

RL-like settings (--optimizer rmsprop, --steps, --loss sampled, --max-grad-norm) bring the probe closer to the
A2C update one factor at a time: optimizer and step budget, then a one-sample noisy signal.

Data: rollouts of the teacher with sampled actions, raw observations + teacher logits, seeds disjoint from
eval (10000) and test (20000); cached in --data-dir and reused.

Run:  python quantum_heads/analysis/distill.py --out quantum_heads/analysis/runs/distill_concentration
"""
from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)
from classic_heads import MLPHead  # noqa: E402
from encoder import Encoder  # noqa: E402
from quantum_heads.vqc_head_i import VQCHeadI  # noqa: E402
from quantum_heads.vqc_head_j import VQCHeadJ  # noqa: E402
from rl_train.envs import make_vec_env  # noqa: E402
from rl_train.policy import Policy  # noqa: E402
from rl_train.rollout import RolloutCollector  # noqa: E402

HEADS = {"mlp64": lambda: MLPHead(hidden=64),
         # the RL heads of PLAN-vqc-head.md section 11, with their registered init (as in train.py)
         "vqc-optim-id": lambda: VQCHeadI(scale_init=0.1, angle_std=0.05, wf_offset=math.pi / 2),
         "vqc-optim-id-gray": lambda: VQCHeadI(scale_init=0.1, angle_std=0.05, wf_offset=math.pi / 2, action_code="gray"),
         "vqc-nav-id": lambda: VQCHeadJ(scale_init=0.05, angle_std=0.05, wf_offset=math.pi / 2, action_code="gray"),
         "mlp128": lambda: MLPHead(hidden=128)}
VARIANTS = ("frozen", "finetune")
OBS_KEYS = ("perception", "navigator", "memory")
# the setting of the report's runs (environment v8): CNN perception on a 5x5 window, 30 obstacles. The teacher
# checkpoints and the cached data are in this setting, and so must be the students and their evaluation.
POLICY_KW = {"perception": "cnn", "window": 5}
ENV_KW = {"window": 5, "n_obstacles": 30}


# --- data ---------------------------------------------------------------------------------------
def load_teacher(checkpoint: str, head_name: str) -> Policy:
    teacher = Policy(HEADS[head_name](), **POLICY_KW)
    teacher.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return teacher.eval()


def collect(teacher: Policy, n_samples: int, seed: int) -> dict[str, torch.Tensor]:
    """Raw observations and teacher logits from rollouts of the teacher (sampled actions)."""
    collector = RolloutCollector(teacher, make_vec_env(16, **ENV_KW), n_steps=64, device="cpu", seed=seed)
    chunks, returns = [], []
    while sum(len(c["logits"]) for c in chunks) < n_samples:
        roll = collector.collect()
        obs = {k: v.reshape(-1, *v.shape[2:]) for k, v in roll.obs.items()}
        with torch.no_grad():
            logits, _ = teacher(obs)
        chunks.append({**obs, "logits": logits})
        returns += roll.ep_returns
    data = {k: torch.cat([c[k] for c in chunks])[:n_samples] for k in (*OBS_KEYS, "logits")}
    data["collect_return_mean"] = torch.tensor(float(np.mean(returns)))
    return data


def load_or_collect(args) -> dict[str, dict[str, torch.Tensor]]:
    os.makedirs(args.data_dir, exist_ok=True)
    teacher = None
    splits = {}
    for name, n, seed in (("train", args.train_samples, args.train_seed), ("val", args.val_samples, args.val_seed)):
        path = os.path.join(args.data_dir, f"{name}.pt")
        if not os.path.exists(path):
            teacher = teacher or load_teacher(args.teacher, args.teacher_head)
            torch.save(collect(teacher, n, seed), path)
        splits[name] = torch.load(path, weights_only=True)
    return splits


# --- metrics ------------------------------------------------------------------------------------
def kl_rows(t_logits: torch.Tensor, s_logits: torch.Tensor) -> torch.Tensor:
    """KL(teacher || student) per row, both given as unnormalized logits."""
    lt, ls = F.log_softmax(t_logits, -1), F.log_softmax(s_logits, -1)
    return (lt.exp() * (lt - ls)).sum(-1)


def entropy(logits: torch.Tensor) -> float:
    lp = F.log_softmax(logits, -1)
    return float(-(lp.exp() * lp).sum(-1).mean())


def concentration(logits: torch.Tensor) -> tuple[float, float]:
    """(median max-prob, fraction of states with max-prob > 0.5): how peaked the policy is."""
    p_max = F.softmax(logits, -1).max(-1).values
    return float(p_max.median()), float((p_max > 0.5).float().mean())


def marginal_kl(t_logits: torch.Tensor, marginal: torch.Tensor) -> float:
    """KL(teacher || best state-independent policy), the denominator of the explained fraction."""
    return float(kl_rows(t_logits, marginal.log().expand_as(t_logits)).mean())


# --- one student --------------------------------------------------------------------------------
def train_student(job: dict) -> dict:
    torch.set_num_threads(1)
    args, head_name, variant, seed, lr = job["args"], job["head"], job["variant"], job["seed"], job["lr"]
    splits = torch.load(job["data_path"], weights_only=True)
    encoder_sd = torch.load(args.encoder_checkpoint or args.teacher, map_location="cpu", weights_only=True)

    torch.manual_seed(seed)
    head = HEADS[head_name]()
    encoder = Encoder(**POLICY_KW)
    encoder.load_state_dict({k[len("encoder."):]: v for k, v in encoder_sd.items() if k.startswith("encoder.")})

    def inputs(split):
        obs = tuple(splits[split][k] for k in OBS_KEYS)
        if variant == "frozen":
            with torch.no_grad():
                return encoder(*obs)
        return obs

    x = {s: inputs(s) for s in ("train", "val")}
    t = {s: splits[s]["logits"] for s in ("train", "val")}

    def student(xb):
        return head(xb) if variant == "frozen" else head(encoder(*xb))

    groups = [{"params": head.parameters(), "lr": lr}]
    if variant == "finetune":
        groups.append({"params": encoder.parameters(), "lr": args.encoder_lr})
    if args.optimizer == "rmsprop":  # the optimizer of the A2C trainer
        opt = torch.optim.RMSprop(groups, alpha=0.99, eps=1e-5)
    else:
        opt = torch.optim.Adam(groups)
    params = [p for g in groups for p in g["params"]]

    n = len(t["train"])
    steps_per_epoch = -(-n // args.batch_size)
    total_steps = args.steps or args.epochs * steps_per_epoch
    eval_interval = args.eval_steps or args.eval_every * steps_per_epoch
    marginal = F.softmax(t["train"], -1).mean(0)
    val_marginal_kl = marginal_kl(t["val"], marginal)
    gen = torch.Generator().manual_seed(seed)
    t0 = time.time()
    best_val = float("inf")
    curve = []  # (step, val explained fraction)
    perm, pos = torch.randperm(n, generator=gen), 0
    for step in range(1, total_steps + 1):
        if pos >= n:
            perm, pos = torch.randperm(n, generator=gen), 0
        idx = perm[pos:pos + args.batch_size]
        pos += args.batch_size
        xb = x["train"][idx] if variant == "frozen" else tuple(a[idx] for a in x["train"])
        if args.loss == "sampled":
            # behavior cloning on one action sampled from the teacher: the noisy, one-sample signal of RL
            actions = torch.multinomial(F.softmax(t["train"][idx], -1), 1, generator=gen).squeeze(1)
            loss = F.cross_entropy(student(xb), actions)
        else:
            loss = kl_rows(t["train"][idx], student(xb)).mean()
        opt.zero_grad()
        loss.backward()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        opt.step()
        if step % eval_interval == 0 or step == total_steps:
            with torch.no_grad():
                v = float(kl_rows(t["val"], student(x["val"])).mean())
            best_val = min(best_val, v)
            curve.append((step, 1.0 - v / val_marginal_kl))

    with torch.no_grad():
        s_train, s_val = student(x["train"]), student(x["val"])
    if args.save_students:  # encoder + head, loadable into rl_train.policy.Policy (critic left at init)
        path = os.path.join(args.out, "students", f"{variant}_{head_name}_s{seed}_lr{lr:g}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"head_name": head_name, "encoder": encoder.state_dict(), "head": head.state_dict()}, path)
    val_kl = float(kl_rows(t["val"], s_val).mean())
    return {
        "head": head_name, "variant": variant, "seed": seed, "lr": lr,
        "train_kl": float(kl_rows(t["train"], s_train).mean()), "val_kl": val_kl, "best_val_kl": best_val,
        "val_frac": 1.0 - val_kl / val_marginal_kl,
        "val_entropy": entropy(s_val), "teacher_val_entropy": entropy(t["val"]),
        "val_maxp_median": concentration(s_val)[0], "val_maxp_gt_half": concentration(s_val)[1],
        "n_params": sum(p.numel() for p in head.parameters()), "seconds": time.time() - t0,
        # learned input scales of heads that have them, [M | N | G] order
        "fs": " ".join(f"{v:.3f}" for v in head.fs.tolist()) if hasattr(head, "fs") else "",
        "curve": " ".join(f"{k}:{f:.4f}" for k, f in curve),  # step:val explained fraction
    }


# --- summary ------------------------------------------------------------------------------------
def summarize(rows: list[dict], splits: dict, reference: str) -> str:
    t_val, t_train = splits["val"]["logits"], splits["train"]["logits"]
    marginal = F.softmax(t_train, -1).mean(0)
    lines = [
        f"teacher: val entropy {entropy(t_val):.3f} nats (max {np.log(8):.3f}), "
        f"KL(teacher || marginal) {marginal_kl(t_val, marginal):.3f}, "
        f"collect return {float(splits['train']['collect_return_mean']):.2f}, "
        f"max-prob median {concentration(t_val)[0]:.3f}, states with max-prob > 0.5 {concentration(t_val)[1]:.3f}",
        "",
        "| variant | head | params | best lr | val frac | val KL | train KL | student entropy | max-prob median "
        "| max-prob > 0.5 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    best = {}  # (variant, head) -> (lr, list of rows); lr chosen by mean val_frac over seeds
    for (variant, head), group in itertools.groupby(sorted(rows, key=lambda r: (r["variant"], r["head"], r["lr"])),
                                                    key=lambda r: (r["variant"], r["head"])):
        by_lr = {}
        for r in group:
            by_lr.setdefault(r["lr"], []).append(r)
        lr, rs = max(by_lr.items(), key=lambda kv: np.mean([r["val_frac"] for r in kv[1]]))
        best[(variant, head)] = rs

        def ms(key):
            v = np.array([r[key] for r in rs])
            return f"{v.mean():.3f}±{v.std():.3f}"
        lines.append(f"| {variant} | {head} | {rs[0]['n_params']} | {lr:g} | {ms('val_frac')} | {ms('val_kl')} | "
                     f"{ms('train_kl')} | {ms('val_entropy')} | {ms('val_maxp_median')} | {ms('val_maxp_gt_half')} |")

    lines += ["", f"Verdict (capacity limit if mean+std of the head < 0.5 * (mean-std) of {reference}, same variant):"]
    for (variant, head), rs in best.items():
        if head == reference or (variant, reference) not in best:
            continue
        g = np.array([r["val_frac"] for r in rs])
        m = np.array([r["val_frac"] for r in best[(variant, reference)]])
        limit = g.mean() + g.std() < 0.5 * (m.mean() - m.std())
        ratio = g.mean() / m.mean() if m.mean() > 0 else float("nan")
        lines.append(f"- {variant} {head}: head/{reference} = {ratio:.2f} -> {'CAPACITY LIMIT' if limit else 'no capacity limit'}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", default=os.path.join(ROOT, "logs/mlp64_s0_v8b_n64_15k_b/best.pt"))
    ap.add_argument("--teacher-head", default="mlp64", choices=list(HEADS), help="head of the teacher checkpoint")
    ap.add_argument("--reference", default="mlp64", choices=list(HEADS), help="ceiling head of the verdict")
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "quantum_heads/analysis/data/mlp64_v8"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-samples", type=int, default=20000)
    ap.add_argument("--val-samples", type=int, default=5000)
    ap.add_argument("--train-seed", type=int, default=30000)
    ap.add_argument("--val-seed", type=int, default=40000)
    ap.add_argument("--heads", nargs="+", default=list(HEADS), choices=list(HEADS))
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-3, 3e-3, 1e-2], help="head lr grid (Adam)")
    ap.add_argument("--encoder-lr", type=float, default=1e-3, help="encoder lr in the finetune variant")
    ap.add_argument("--encoder-checkpoint", default=None,
                    help="Policy state_dict to take the student encoder from (default: the teacher)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--eval-every", type=int, default=10, help="in epochs")
    ap.add_argument("--steps", type=int, default=0, help="train for this many steps instead of --epochs")
    ap.add_argument("--eval-steps", type=int, default=0, help="eval interval in steps (overrides --eval-every)")
    ap.add_argument("--optimizer", choices=("adam", "rmsprop"), default="adam", help="rmsprop = the A2C optimizer")
    ap.add_argument("--loss", choices=("kl", "sampled"), default="kl",
                    help="kl = full KL(teacher || student); sampled = cross-entropy on one action sampled from the teacher")
    ap.add_argument("--max-grad-norm", type=float, default=0.0, help="clip the student gradient norm (0 = off)")
    ap.add_argument("--save-students", action="store_true", help="save each student to <out>/students/")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = ap.parse_args()

    splits = load_or_collect(args)
    os.makedirs(args.out, exist_ok=True)
    data_path = os.path.join(args.out, "_splits.pt")
    torch.save(splits, data_path)

    jobs = [{"args": args, "data_path": data_path, "head": h, "variant": v, "seed": s, "lr": lr}
            for v in args.variants for h in args.heads for s in args.seeds for lr in args.lrs]
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for r in pool.map(train_student, jobs):
            rows.append(r)
            print(f"{r['variant']:8s} {r['head']:8s} seed {r['seed']} lr {r['lr']:g}: val frac {r['val_frac']:.3f} "
                  f"val KL {r['val_kl']:.4f} entropy {r['val_entropy']:.3f} ({r['seconds']:.0f}s)", flush=True)
    os.remove(data_path)

    with open(os.path.join(args.out, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = summarize(rows, splits, args.reference)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(summary + "\n")
    print("\n" + summary)


if __name__ == "__main__":
    main()
