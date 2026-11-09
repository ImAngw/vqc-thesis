"""
A3.3 of docs/design/SPEC-generalization-full.md: cross-distillation between J (vqc-nav-id) and the Fourier head.

Can a J head represent the policy of a trained Fourier run, and vice versa? Per teacher run (J or Fourier, K=25 or
200, seeds 0-9): 20000 states from its rollouts with sampled actions on fresh layouts (env seeds 30000+, disjoint from
eval 10000-10099 and from the training pools), split 80/20 train/val; the teacher encoder is frozen and its features
precomputed; a fresh student head is trained on KL(teacher || student) (Adam, batch 512, 100 epochs). Students: J,
Fourier and MLP64 (a generic ceiling), 3 init seeds, head lr in {1e-3, 3e-3}. The metrics and the loss are those of
distill.py: explained fraction = 1 - KL(teacher || student) / KL(teacher || teacher marginal) on val. For the lr
chosen on val (per K, teacher type, student type), the student's test return with the teacher encoder (100 fresh
layouts, sampled actions).

    python quantum_heads/analysis/cross_distill.py --k 25 200 --workers 8
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import policy_equivalence as pe  # noqa: E402  (also puts ROOT, quantum_heads and scripts on the path)
from distill import kl_rows, marginal_kl  # noqa: E402
from rl_train.envs import make_vec_env  # noqa: E402
from rl_train.rollout import RolloutCollector  # noqa: E402
from train import HEADS  # noqa: E402

STUDENTS = {"J": "vqc-nav-id", "F": "fourier4", "MLP64": "mlp64"}
TEACHER_TYPE = {"nav-id": "J", "four": "F"}
OBS_KEYS = ("perception", "navigator", "memory")


def collect_features(teacher, n: int, seed: int) -> dict[str, torch.Tensor]:
    """Frozen-encoder features and teacher logits on n states of the teacher's own sampled rollouts."""
    env = make_vec_env(16, **pe.RUN_CONFIG["env_kwargs"])
    collector = RolloutCollector(teacher, env, n_steps=64, device="cpu", seed=seed)
    feats, logits, total = [], [], 0
    while total < n:
        roll = collector.collect()
        obs = {k: v.reshape(-1, *v.shape[2:]) for k, v in roll.obs.items()}
        with torch.no_grad():
            f = pe.features_of(teacher, obs)
            feats.append(f)
            logits.append(teacher.head(f))
        total += len(f)
    env.close()
    return {"features": torch.cat(feats)[:n], "logits": torch.cat(logits)[:n]}


def data_path(args, arm: str, k: str, seed: int) -> str:
    return os.path.join(args.data_dir, f"{pe.run_name(arm, k, seed)}.pt")


def prepare(args, arm: str, k: str, seed: int) -> None:
    path = data_path(args, arm, k, seed)
    if os.path.exists(path):
        return
    teacher = pe.load_run(arm, k, seed, args.checkpoints)
    data = collect_features(teacher, args.samples, args.collect_seed)
    n_train = int(0.8 * args.samples)
    perm = torch.randperm(args.samples, generator=torch.Generator().manual_seed(0))
    split = {"train": perm[:n_train], "val": perm[n_train:]}
    torch.save({s: {key: v[idx] for key, v in data.items()} for s, idx in split.items()}, path)


def train_student(job: dict) -> dict:
    torch.set_num_threads(1)
    d = torch.load(job["data"], weights_only=True)
    x, t = {s: d[s]["features"] for s in d}, {s: d[s]["logits"] for s in d}
    torch.manual_seed(job["seed"])
    head = HEADS[STUDENTS[job["student"]]]()
    opt = torch.optim.Adam(head.parameters(), lr=job["lr"])
    n, gen = len(t["train"]), torch.Generator().manual_seed(job["seed"])
    steps = job["epochs"] * -(-n // job["batch"])
    perm, pos, t0 = torch.randperm(n, generator=gen), 0, time.time()
    for _ in range(steps):
        if pos >= n:
            perm, pos = torch.randperm(n, generator=gen), 0
        idx = perm[pos:pos + job["batch"]]
        pos += job["batch"]
        loss = kl_rows(t["train"][idx], head(x["train"][idx])).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        val_kl = float(kl_rows(t["val"], head(x["val"])).mean())
    denom = marginal_kl(t["val"], F.softmax(t["train"], -1).mean(0))
    if job["save"]:
        torch.save(head.state_dict(), job["save"])
    return {**{k: job[k] for k in ("k", "teacher", "teacher_seed", "student", "seed", "lr")},
            "val_kl": val_kl, "val_frac": 1.0 - val_kl / denom, "seconds": time.time() - t0}


def student_return(job: dict) -> dict:
    """Test return of the teacher run with its head replaced by the trained student head."""
    torch.set_num_threads(1)
    model = pe.load_run(job["arm"], job["k"], job["teacher_seed"], job["checkpoints"])
    head = HEADS[STUDENTS[job["student"]]]()
    head.load_state_dict(torch.load(job["head"], weights_only=True))
    model.head = head
    metrics, _ = pe.test_eval(model)
    return {**{k: job[k] for k in ("k", "teacher", "teacher_seed", "student", "seed")},
            "return": metrics["return_mean"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", nargs="+", default=["25", "200"])
    ap.add_argument("--seeds", type=int, default=10, help="teacher seeds")
    ap.add_argument("--init-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-3, 3e-3])
    ap.add_argument("--samples", type=int, default=20000)
    ap.add_argument("--collect-seed", type=int, default=30000)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--reuse", default=None, help="json of an earlier call: its students are not trained again")
    ap.add_argument("--checkpoints", default=os.path.join(pe.ROOT, "checkpoints"))
    ap.add_argument("--data-dir", default=os.path.join(pe.ROOT, "quantum_heads", "analysis", "data", "a3"))
    ap.add_argument("--out", default=os.path.join(pe.ROOT, "quantum_heads", "analysis", "runs", "a3"))
    args = ap.parse_args()
    os.makedirs(args.data_dir, exist_ok=True)
    students_dir = os.path.join(args.out, "students")
    os.makedirs(students_dir, exist_ok=True)

    teachers = [(arm, k, s) for k in args.k for arm in TEACHER_TYPE for s in range(args.seeds)]
    for arm, k, s in teachers:
        prepare(args, arm, k, s)
    print(f"data ready: {len(teachers)} teachers")

    # students already trained by an earlier call (--reuse: its json) are not trained again
    cached = {}
    if args.reuse:
        with open(args.reuse) as f:
            for r in json.load(f)["students"]:
                cached[(r["k"], r["teacher"], r["teacher_seed"], r["student"], r["seed"], r["lr"])] = r
    jobs = []
    for (arm, k, s), student, seed, lr in itertools.product(teachers, STUDENTS, args.init_seeds, args.lrs):
        name = f"{pe.run_name(arm, k, s)}__{student}_s{seed}_lr{lr:g}.pt"
        jobs.append({"data": data_path(args, arm, k, s), "k": k, "teacher": TEACHER_TYPE[arm], "teacher_seed": s,
                     "student": student, "seed": seed, "lr": lr, "epochs": args.epochs, "batch": args.batch,
                     "save": os.path.join(students_dir, name)})
    key = lambda j: (j["k"], j["teacher"], j["teacher_seed"], j["student"], j["seed"], j["lr"])  # noqa: E731
    todo = [j for j in jobs if key(j) not in cached]
    with ProcessPoolExecutor(args.workers) as pool:
        trained = dict(zip(map(key, todo), pool.map(train_student, todo)))
    rows = [cached.get(key(j)) or trained[key(j)] for j in jobs]
    print(f"trained {len(todo)} students, reused {len(jobs) - len(todo)}")

    # lr per (K, teacher type, student type): best mean val fraction over teachers and init seeds
    best_lr = {}
    for key, group in itertools.groupby(sorted(rows, key=lambda r: (r["k"], r["teacher"], r["student"])),
                                        key=lambda r: (r["k"], r["teacher"], r["student"])):
        group = list(group)
        best_lr[key] = max(args.lrs, key=lambda lr: np.mean([r["val_frac"] for r in group if r["lr"] == lr]))

    ret_jobs = [{"arm": {v: a for a, v in TEACHER_TYPE.items()}[r["teacher"]], "k": r["k"],
                 "teacher": r["teacher"], "teacher_seed": r["teacher_seed"], "student": r["student"],
                 "seed": r["seed"], "checkpoints": args.checkpoints, "head": j["save"]}
                for r, j in zip(rows, jobs) if r["lr"] == best_lr[(r["k"], r["teacher"], r["student"])]]
    with ProcessPoolExecutor(args.workers) as pool:
        returns = list(pool.map(student_return, ret_jobs))
    teacher_ret = {}
    for arm, k, s in teachers:
        teacher_ret[(k, TEACHER_TYPE[arm], s)] = pe.test_eval(pe.load_run(arm, k, s, args.checkpoints))[0]["return_mean"]

    summary = []
    print("\nA3.3 cross-distillation (frozen teacher encoder): explained fraction on val and test return, "
          "mean +- SE over the 10 teachers (each teacher: mean of 3 init seeds at the chosen lr)")
    for (k, teacher, student), lr in sorted(best_lr.items()):
        per_t = {}
        for r in rows:
            if (r["k"], r["teacher"], r["student"], r["lr"]) == (k, teacher, student, lr):
                per_t.setdefault(r["teacher_seed"], []).append(r["val_frac"])
        rets = {}
        for r in returns:
            if (r["k"], r["teacher"], r["student"]) == (k, teacher, student):
                rets.setdefault(r["teacher_seed"], []).append(r["return"])
        frac = [np.mean(v) for v in per_t.values()]
        ret = [np.mean(v) for v in rets.values()]
        tret = [teacher_ret[(k, teacher, s)] for s in rets]
        row = {"k": k, "teacher": teacher, "student": student, "lr": lr,
               "frac_mean": float(np.mean(frac)), "frac_se": float(np.std(frac, ddof=1) / np.sqrt(len(frac))),
               "return_mean": float(np.mean(ret)), "return_se": float(np.std(ret, ddof=1) / np.sqrt(len(ret))),
               "teacher_return_mean": float(np.mean(tret))}
        summary.append(row)
        print(f"  K={k:>3s} teacher {teacher:2s} <- student {student:5s} (lr {lr:g}): frac {row['frac_mean']:.3f} +- "
              f"{row['frac_se']:.3f}, test {row['return_mean']:.2f} +- {row['return_se']:.2f} "
              f"(teacher {row['teacher_return_mean']:.2f})")
    with open(os.path.join(args.out, "cross_distill.json"), "w") as f:
        json.dump({"summary": summary, "students": rows, "returns": returns}, f, indent=1)
    print(f"saved {os.path.join(args.out, 'cross_distill.json')}")


if __name__ == "__main__":
    main()
