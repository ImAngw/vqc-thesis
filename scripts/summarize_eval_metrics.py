"""Greedy and sampled eval returns of the generalization runs, from checkpoints/<run>/metrics.jsonl.

The round logs only hold the --show columns; metrics.jsonl has every eval metric of every round, including
greedy/* (argmax action) and pool/greedy/*. Same cells and window as summarize_gen_pilot.py:

    python scripts/summarize_eval_metrics.py 'gen-*' 'ent-*' --ref 'nav-id-hlr2e-3-detc-s[12]-7.5k-cpu-outer=nav-id:inf'
"""

import argparse
import fnmatch
import json
import math
from pathlib import Path

from summarize_gen_pilot import GEN_NAME, SEED, fmt, k_order, mean_se

METRICS = ("sample/return_mean", "greedy/return_mean", "pool/sample/return_mean", "pool/greedy/return_mean")
SHORT = {"sample/return_mean": "test", "greedy/return_mean": "test greedy",
         "pool/sample/return_mean": "pool", "pool/greedy/return_mean": "pool greedy"}


def read_eval(path: Path) -> dict[int, dict[str, float]]:
    """{round: eval metrics}; a resumed run keeps the last occurrence of a round."""
    rounds = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("eval"):
                rounds[int(row["round"])] = row["eval"]
    return rounds


def window(rounds: dict, key: str, lo: int, hi: int) -> float:
    values = [rounds[r][key] for r in range(lo, hi + 1) if r in rounds and key in rounds[r]]
    return sum(values) / len(values) if len(values) == hi - lo + 1 else math.nan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", default=["gen-*", "ent-*"], help="run name globs")
    parser.add_argument("--ref", action="append", default=[], help="GLOB=HEAD:K, older runs joining a cell")
    parser.add_argument("--checkpoints", default="checkpoints")
    parser.add_argument("--rounds", default="41-50")
    parser.add_argument("--pairs", default="nav-id:h5,nav-id:h128,nav-id:h5-ent,nav-id:h128-ent")
    args = parser.parse_args()
    lo, hi = map(int, args.rounds.split("-"))

    names = sorted(p.name for p in Path(args.checkpoints).iterdir() if (p / "metrics.jsonl").exists())
    runs = {}
    for name in names:
        m = GEN_NAME.match(name)
        if m and any(fnmatch.fnmatch(name, g) for g in args.runs):
            head = m["head"] if m["sweep"] == "gen" else f"{m['head']}-{m['sweep']}"
            runs[name] = (head, m["k"], int(m["seed"]))
    for spec in args.ref:
        pattern, _, cell = spec.partition("=")
        head, _, k = cell.partition(":")
        for name in names:
            if fnmatch.fnmatch(name, pattern) and SEED.search(name):
                runs[name] = (head, k or "inf", int(SEED.search(name)[1]))
    if not runs:
        raise SystemExit(f"no run matches {args.runs} in {args.checkpoints}/")

    cells = {}  # (head, K) -> {seed: {metric: window mean}}
    for name, (head, k, seed) in runs.items():
        rounds = read_eval(Path(args.checkpoints) / name / "metrics.jsonl")
        cells.setdefault((head, k), {})[seed] = {m: window(rounds, m, lo, hi) for m in METRICS}

    print(f"Per cell: mean +- SE over seeds of rounds {lo}-{hi} (per seed in brackets); '-' = absent")
    for (head, k), seeds in sorted(cells.items(), key=lambda kv: (kv[0][0], k_order(kv[0][1]))):
        parts = []
        for m in METRICS:
            values = [seeds[s][m] for s in sorted(seeds)]
            mean, se = mean_se(values)
            if not math.isnan(mean):
                parts.append(f"{SHORT[m]} {fmt(mean)} +- {fmt(se, 4)} [{' '.join(fmt(v, 5) for v in values)}]")
        print(f"{head:>10s} K={k:>4s} n={len(seeds)}  " + "  |  ".join(parts))

    print("\nPaired differences per seed (A - B, same seed = same pool), test greedy and test sample")
    for pair in args.pairs.split(","):
        a, _, b = pair.partition(":")
        for k in sorted({k for h, k in cells if h == a} & {k for h, k in cells if h == b}, key=k_order):
            common = sorted(set(cells[(a, k)]) & set(cells[(b, k)]))
            for m in ("greedy/return_mean", "sample/return_mean"):
                diffs = [cells[(a, k)][s][m] - cells[(b, k)][s][m] for s in common]
                mean, se = mean_se(diffs)
                if not math.isnan(mean):
                    per_seed = " ".join(f"s{s}:{d:+.2f}" for s, d in zip(common, diffs) if not math.isnan(d))
                    print(f"  {a} - {b:9s} K={k:>4s} {SHORT[m]:11s}: {fmt(mean)} +- {fmt(se, 4)}  [{per_seed}]")


if __name__ == "__main__":
    main()
