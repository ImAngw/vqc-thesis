"""
A3 of docs/design/SPEC-generalization-full.md: do J (vqc-nav-id) and the Fourier head compute the same kind of policy?
Plus the eval-time temperature control of the K=5 entropy question (same spec, section on the K=5 control).

Three commands, all offline on trained runs (weights model_latest.pt, round 50):
  distance     A3.1: TV distance (and both KLs) between the action distributions of every pair of J / Fourier runs
               at one K, each pair on the visited test states of both runs; reported as a 2x2 table
               (same / different head x same / different pool: seed s = pool s).
  periods      A3.2: how many periods every J data-gate angle and every Fourier projection sweeps on the run's own
               encoder features, rho = (q99 - q1) / (2 pi); plus turning points of the action probabilities along a
               sweep of each encoder feature.
  temperature  K=5 control: logits / T, T per run so that the entropy on its own visited test states matches J's;
               test return at that T, paired with J per seed.

States = encoder inputs visited by the run's own sampled test eval (100 fresh layouts, eval seeds 10000-10099).

    python quantum_heads/analysis/policy_equivalence.py distance --k 25
    python quantum_heads/analysis/policy_equivalence.py periods --k 25
    python quantum_heads/analysis/policy_equivalence.py temperature --k 5 --arms four,h5,h128
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from typing import Callable

import numpy as np
import torch
from torch import nn

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "quantum_heads"), os.path.join(ROOT, "scripts")]

from rl_train.config import A2CConfig  # noqa: E402
from rl_train.envs import make_env  # noqa: E402
from rl_train.evaluate import evaluate  # noqa: E402
from rl_train.policy import build_policy, encoder_kwargs  # noqa: E402
from summarize_gen_pilot import ci95, mean_se  # noqa: E402
from train import HEADS  # noqa: E402

# run-name arm -> registered head; every generalization run shares the rest of the configuration
ARM_HEAD = {"nav-id": "vqc-nav-id", "four": "fourier4", "h5": "mlp5", "h128": "mlp128"}
RUN_CONFIG = dict(perception="cnn", detach_critic=1, env_kwargs={"window": 5, "n_obstacles": 30})
RERUN = {("nav-id", "5", 6), ("nav-id", "5", 9)}  # J K=5 s6 / s9: the -r reruns replace the stopped runs
PERIODIC_RHO = 0.5  # the angle sweeps >= pi on the data: the response can turn back


def run_name(arm: str, k: str, seed: int) -> str:
    return f"gen-{arm}-k{k}-s{seed}-7.5k-cpu-outer" + ("-r" if (arm, k, seed) in RERUN else "")


def load_run(arm: str, k: str, seed: int, checkpoints: str) -> nn.Module:
    cfg = A2CConfig(head=ARM_HEAD[arm], **RUN_CONFIG)
    model = build_policy(HEADS[cfg.head], **encoder_kwargs(cfg))
    path = os.path.join(checkpoints, run_name(arm, k, seed), "model_latest.pt")
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.detach_critic = True
    return model.eval()


class Tempered(nn.Module):
    """A policy whose logits are divided by a temperature (T = 1: the policy itself)."""

    def __init__(self, policy: nn.Module, temperature: float = 1.0):
        super().__init__()
        self.policy, self.temperature = policy, temperature

    def forward(self, obs):
        logits, value = self.policy(obs)
        return logits / self.temperature, value


def test_eval(model: nn.Module, temperature: float = 1.0) -> tuple[dict, dict[str, torch.Tensor]]:
    """Sampled test eval (100 fresh layouts) of `model` at a temperature: its metrics and every visited observation."""
    cfg = A2CConfig(**RUN_CONFIG)
    seen = []
    hook = model.register_forward_pre_hook(lambda mod, args: seen.append({k: v.clone() for k, v in args[0].items()}))
    try:
        with torch.no_grad():
            metrics = evaluate(Tempered(model, temperature).eval(), lambda: make_env(**cfg.env_kwargs), cfg.eval_episodes,
                               cfg.eval_seed, greedy=False)
    finally:
        hook.remove()
    return metrics, {k: torch.cat([s[k] for s in seen]) for k in seen[0]}


def subsample(obs: dict[str, torch.Tensor], n: int, seed: int = 0) -> dict[str, torch.Tensor]:
    total = len(next(iter(obs.values())))
    idx = torch.randperm(total, generator=torch.Generator().manual_seed(seed))[:n]
    return {k: v[idx] for k, v in obs.items()}


def action_probs(model: nn.Module, obs: dict[str, torch.Tensor], temperature: float = 1.0) -> torch.Tensor:
    with torch.no_grad():
        return torch.softmax(model(obs)[0] / temperature, dim=-1)


def features_of(model: nn.Module, obs: dict[str, torch.Tensor]) -> torch.Tensor:
    with torch.no_grad():
        return model.encoder(obs["perception"], obs["navigator"], obs["memory"])


def tv(p: torch.Tensor, q: torch.Tensor) -> float:
    """Mean total variation distance between rows of two (N, A) probability tables."""
    return 0.5 * (p - q).abs().sum(-1).mean().item()


def kl(p: torch.Tensor, q: torch.Tensor, floor: float = 1e-12) -> float:
    """Mean KL(p || q) over rows, probabilities floored at `floor`."""
    p, q = p.clamp_min(floor), q.clamp_min(floor)
    return (p * (p.log() - q.log())).sum(-1).mean().item()


def entropy(p: torch.Tensor, floor: float = 1e-12) -> float:
    return -(p * p.clamp_min(floor).log()).sum(-1).mean().item()


# ---------------------------------------------------------------- A3.1 distance

def pair_classes(seeds: list[int]) -> dict[str, list[tuple[tuple[str, int], tuple[str, int]]]]:
    """The four classes of run pairs of the 2x2 table (J = nav-id, F = four; seed = pool)."""
    return {
        "cross_same_pool": [(("nav-id", s), ("four", s)) for s in seeds],
        "within_J": [(("nav-id", a), ("nav-id", b)) for a, b in itertools.combinations(seeds, 2)],
        "within_F": [(("four", a), ("four", b)) for a, b in itertools.combinations(seeds, 2)],
        "cross_diff_pool": [(("nav-id", a), ("four", b)) for a in seeds for b in seeds if a != b],
    }


def pair_distances(probs: dict, a: tuple[str, int], b: tuple[str, int]) -> dict[str, float]:
    """probs[(run, states_of)] = (N, 8) action probabilities of `run` on the states visited by `states_of`.
    Distance on the states of both runs; KL oriented from the first run of the pair."""
    pa = torch.cat([probs[(a, a)], probs[(a, b)]])
    pb = torch.cat([probs[(b, a)], probs[(b, b)]])
    return {"tv": tv(pa, pb), "kl_ab": kl(pa, pb), "kl_ba": kl(pb, pa)}


def summarize_classes(dist: dict, classes: dict) -> dict[str, dict[str, float]]:
    out = {}
    for name, pairs in classes.items():
        values = [dist[(a, b)]["tv"] for a, b in pairs]
        out[name] = {"pairs": len(values), "tv_mean": float(np.mean(values)), "tv_min": float(np.min(values)),
                     "tv_max": float(np.max(values))}
    within = (out["within_J"]["tv_mean"] + out["within_F"]["tv_mean"]) / 2
    out["head_effect"] = {"tv": out["cross_diff_pool"]["tv_mean"] - within, "seed_spread": within}
    return out


def cmd_distance(args) -> dict:
    seeds = list(range(args.seeds))
    runs = [(arm, s) for arm in ("nav-id", "four") for s in seeds]
    models = {r: load_run(r[0], args.k, r[1], args.checkpoints) for r in runs}
    states = {}
    for r in runs:
        _, obs = test_eval(models[r])
        states[r] = subsample(obs, args.states)
    probs = {(r, o): action_probs(models[r], states[o]) for r in runs for o in runs}
    classes = pair_classes(seeds)
    dist = {pair: pair_distances(probs, *pair) for pairs in classes.values() for pair in pairs}
    summary = summarize_classes(dist, classes)
    kl_cross = [dist[p] for p in classes["cross_same_pool"]]
    summary["cross_same_pool"]["kl_J_F"] = float(np.mean([d["kl_ab"] for d in kl_cross]))
    summary["cross_same_pool"]["kl_F_J"] = float(np.mean([d["kl_ba"] for d in kl_cross]))
    summary["per_seed_cross_same_pool_tv"] = {s: dist[(("nav-id", s), ("four", s))]["tv"] for s in seeds}
    print(f"A3.1 K={args.k}: TV distance between action distributions (mean [min, max] over pairs)")
    for name in ("within_J", "within_F", "cross_same_pool", "cross_diff_pool"):
        c = summary[name]
        print(f"  {name:16s} {c['tv_mean']:.3f} [{c['tv_min']:.3f}, {c['tv_max']:.3f}]  ({c['pairs']} pairs)")
    h = summary["head_effect"]
    print(f"  head effect = cross (different pool) - seed spread = {h['tv']:+.3f} (seed spread {h['seed_spread']:.3f})")
    print(f"  KL(J||F) {summary['cross_same_pool']['kl_J_F']:.3f}, KL(F||J) {summary['cross_same_pool']['kl_F_J']:.3f}"
          " (same pool)")
    return summary


# ---------------------------------------------------------------- A3.2 periods

def angle_range_periods(angles: torch.Tensor) -> torch.Tensor:
    """(N, G) angles over N states -> (G,) rho = (q99 - q1) / (2 pi)."""
    q = torch.quantile(angles, torch.tensor([0.01, 0.99], dtype=angles.dtype), dim=0)
    return (q[1] - q[0]) / (2 * math.pi)


def head_angles(head: nn.Module, features: torch.Tensor) -> torch.Tensor:
    """(N, G): the data-dependent angles of a head on its input features. J: the 36 data-gate angles (3 blocks x 3
    qubits x 4 gates); Fourier: the n_freq projections W_k . x."""
    with torch.no_grad():
        if hasattr(head, "data_angles"):
            return torch.cat([head.data_angles(features, r).flatten(1) for r in (1, 2, 3)], dim=1)
        if hasattr(head, "freq"):
            return features @ head.freq.T
    raise TypeError(f"no data angles for {type(head).__name__}")


def turning_points(curves: torch.Tensor, tol: float = 1e-4) -> torch.Tensor:
    """(..., L) curves -> (...) number of local extrema; steps smaller than `tol` count as flat."""
    d = curves.diff(dim=-1)
    sign = torch.where(d.abs() < tol, torch.zeros_like(d), d.sign())
    rows = sign.reshape(-1, sign.shape[-1])
    counts = [int((nz[1:] != nz[:-1]).sum()) for nz in (row[row != 0] for row in rows)]  # sign changes, flats skipped
    return torch.tensor(counts, dtype=torch.long).reshape(curves.shape[:-1])


def nonmonotone_amplitude(curves: torch.Tensor) -> torch.Tensor:
    """(..., L) -> (...): sum |steps| - |end - start|, 0 for a monotone curve (the part that goes back and forth)."""
    d = curves.diff(dim=-1)
    return d.abs().sum(-1) - (curves[..., -1] - curves[..., 0]).abs()


def significant_turning_points(curves: torch.Tensor, min_change: float = 0.02) -> torch.Tensor:
    """(..., L) -> (...): reversals into a monotone stretch whose net change is >= min_change."""
    d = curves.diff(dim=-1)
    rows = d.reshape(-1, d.shape[-1])
    counts = []
    for row in rows:
        stretches, current = [], 0.0  # net change of each maximal monotone stretch (flat steps join the current one)
        for step in row.tolist():
            if current * step < 0:
                stretches.append(current)
                current = 0.0
            current += step
        stretches.append(current)
        counts.append(sum(1 for c in stretches[1:] if abs(c) >= min_change))
    return torch.tensor(counts, dtype=torch.long).reshape(curves.shape[:-1])


def sweep_turning_points(head: nn.Module, features: torch.Tensor, n_states: int = 200, grid: int = 64) -> list[dict]:
    """Per encoder feature: mean over the 8 action probabilities and the states of the turning points, the significant
    turning points and the non-monotone amplitude when that feature sweeps its q1-q99 range (64 points) with the
    others fixed at a visited state."""
    base = features[:n_states]
    q = torch.quantile(features, torch.tensor([0.01, 0.99], dtype=features.dtype), dim=0)
    out = []
    with torch.no_grad():
        for f in range(features.shape[1]):
            values = torch.linspace(q[0, f].item(), q[1, f].item(), grid)
            x = base[:, None, :].repeat(1, grid, 1)
            x[:, :, f] = values
            p = torch.softmax(head(x.reshape(-1, x.shape[-1])), dim=-1).reshape(len(base), grid, -1)
            curves = p.transpose(1, 2)
            out.append({"turning": turning_points(curves).float().mean().item(),
                        "significant": significant_turning_points(curves).float().mean().item(),
                        "amplitude": nonmonotone_amplitude(curves).mean().item()})
    return out


def cmd_periods(args) -> dict:
    result = {}
    for arm in ("nav-id", "four"):
        per_seed = {}
        for s in range(args.seeds):
            model = load_run(arm, args.k, s, args.checkpoints)
            _, obs = test_eval(model)
            feats = features_of(model, subsample(obs, args.states))
            rho = angle_range_periods(head_angles(model.head, feats))
            per_seed[s] = {"rho_max": rho.max().item(), "rho_median": rho.median().item(),
                           "periodic": int((rho >= PERIODIC_RHO).sum()), "angles": len(rho),
                           "turning_points": sweep_turning_points(model.head, feats)}
        n_periodic = sum(1 for v in per_seed.values() if v["periodic"] > 0)
        result[arm] = {"per_seed": per_seed, "seeds_with_periodic_angle": n_periodic}
        print(f"A3.2 K={args.k} {arm}: seeds with >= 1 angle sweeping >= half a period: {n_periodic}/{args.seeds}")
        for s, v in per_seed.items():
            tp = {key: np.mean([f[key] for f in v["turning_points"]]) for key in ("turning", "significant", "amplitude")}
            print(f"  s{s}: rho max {v['rho_max']:.2f} median {v['rho_median']:.2f}, periodic angles "
                  f"{v['periodic']}/{v['angles']}, per sweep: turning points {tp['turning']:.2f}, significant "
                  f"{tp['significant']:.2f}, non-monotone amplitude {tp['amplitude']:.3f}")
        for key in ("turning", "significant", "amplitude"):
            result[arm][f"mean_{key}"] = float(np.mean([np.mean([f[key] for f in v["turning_points"]])
                                                        for v in per_seed.values()]))
        print(f"  mean over seeds: turning {result[arm]['mean_turning']:.2f}, significant "
              f"{result[arm]['mean_significant']:.2f}, amplitude {result[arm]['mean_amplitude']:.3f}")
    return result


# ---------------------------------------------------------------- temperature control

def find_temperature(entropy_at: Callable[[float], float], target: float, tol: float = 0.02,
                     lo: float = 0.25, hi: float = 20.0, iters: int = 20) -> tuple[float, float]:
    """Bisection on log T for entropy_at(T) = target (entropy grows with T). Returns (T, entropy at T); clips to
    the bracket when the target is outside it."""
    log_lo, log_hi = math.log(lo), math.log(hi)
    h_lo, h_hi = entropy_at(lo), entropy_at(hi)
    if target <= h_lo:
        return lo, h_lo
    if target >= h_hi:
        return hi, h_hi
    t, h = 1.0, entropy_at(1.0)
    for _ in range(iters):
        if abs(h - target) <= tol:
            break
        if h < target:
            log_lo = math.log(t)
        else:
            log_hi = math.log(t)
        t = math.exp((log_lo + log_hi) / 2)
        h = entropy_at(t)
    return t, h


def visited_entropy(model: nn.Module, temperature: float) -> tuple[float, float]:
    """(entropy of the tempered policy on the states it visits in the test eval, test return)."""
    metrics, obs = test_eval(model, temperature)
    return entropy(action_probs(model, obs, temperature)), metrics["return_mean"]


def cmd_temperature(args) -> dict:
    seeds = list(range(args.seeds))
    j = {s: visited_entropy(load_run("nav-id", args.k, s, args.checkpoints), 1.0) for s in seeds}
    target = float(np.mean([h for h, _ in j.values()]))
    print(f"K={args.k}: J entropy on its visited test states (T=1): mean {target:.3f} "
          f"(per seed {', '.join(f'{h:.2f}' for h, _ in j.values())}); J test return "
          f"{np.mean([r for _, r in j.values()]):.2f}")
    result = {"target_entropy": target, "J": {s: {"entropy": h, "return": r} for s, (h, r) in j.items()}}
    for arm in args.arms.split(","):
        rows = {}
        for s in seeds:
            model = load_run(arm, args.k, s, args.checkpoints)
            h1, r1 = visited_entropy(model, 1.0)
            cache = {}

            def entropy_at(t, model=model, cache=cache):
                cache[t] = visited_entropy(model, t)
                return cache[t][0]

            t, h = find_temperature(entropy_at, target)
            rows[s] = {"T": t, "entropy": h, "return": cache[t][1], "entropy_T1": h1, "return_T1": r1}
        diffs = [j[s][1] - rows[s]["return"] for s in seeds]
        m, se = mean_se(diffs)
        lo, hi = ci95(m, se, len(diffs))
        result[arm] = {"per_seed": rows, "J_minus_X": {"mean": m, "se": se, "ci95": [lo, hi]}}
        print(f"  {arm}: T {np.mean([v['T'] for v in rows.values()]):.2f} (per seed "
              f"{', '.join(f'{v['T']:.2f}' for v in rows.values())}), entropy "
              f"{np.mean([v['entropy'] for v in rows.values()]):.3f}, test {np.mean([v['return'] for v in rows.values()]):.2f}"
              f" (T=1: {np.mean([v['return_T1'] for v in rows.values()]):.2f}); J - {arm}(T) = {m:+.2f} +- {se:.2f}, "
              f"CI95 [{lo:+.2f}, {hi:+.2f}]")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("distance", "periods", "temperature"))
    parser.add_argument("--k", default="25", help="K of the runs: 5, 25, 200 or inf")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--states", type=int, default=2000, help="visited states per run")
    parser.add_argument("--arms", default="four,h5,h128", help="temperature: arms tempered to J's entropy")
    parser.add_argument("--checkpoints", default=os.path.join(ROOT, "checkpoints"))
    parser.add_argument("--out", default=os.path.join(ROOT, "quantum_heads", "analysis", "runs", "a3"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    result = {"distance": cmd_distance, "periods": cmd_periods, "temperature": cmd_temperature}[args.command](args)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.command}_k{args.k}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=1, default=str)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
