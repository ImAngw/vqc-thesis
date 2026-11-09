"""Summary of the generalization pilot (docs/design/SPEC-generalization-pilot.md) from the round logs (`node_0.log`, written by train.py with --inner-steps).

Reads logs/<run>/node_0.log, takes the TOTAL row of every round and averages a window of rounds (41-50 by default)
of the test return (`sample/return_mean`, fresh layouts) and of the pool return (`pool/sample/return_mean`,
training layouts). Pilot runs are named gen-<head>-k<K|inf>-s<seed>-...; the entropy-matched control ent-<head>-... becomes head
<head>-ent. Older runs can join a (head, K) cell with --ref 'GLOB=HEAD:K'; --extra adds per-cell means of other
logged metrics (e.g. entropy,greedy/return_mean).

    python scripts/summarize_gen_pilot.py 'gen-*' --ref 'nav-id-hlr2e-3-detc-s[012]-7.5k-cpu-outer=nav-id:inf'
"""

import argparse
import fnmatch
import math
import re
from pathlib import Path

TEST, POOL = "sample/return_mean", "pool/sample/return_mean"
# ent = entropy-matched; the runs of the entropy-target sweep (ent-sweep-*, scripts/sweep_entropy.py) are not cells here
GEN_NAME = re.compile(r"^(?P<sweep>gen|ent)-(?!sweep-)(?P<head>.+?)-k(?P<k>\d+|inf)-s(?P<seed>\d+)-")
SEED = re.compile(r"-s(\d+)(?=-|$)")
OVERFIT_GAP = 1.0  # spec: pool - test >= 1.0 on at least 2 of 3 seeds
# two-sided 95% Student t quantiles t_{0.975}(df) for df = 1..30 (scipy.stats.t.ppf, 3 decimals)
T975 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
        2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042)


def read_rounds(log: Path) -> dict[int, dict[str, float]]:
    """{round: {column: value}} from the TOTAL rows; a resumed run keeps the last occurrence of a round."""
    rounds, current, header = {}, None, None
    for line in log.read_text().splitlines():
        if line.startswith("round "):
            current = int(line.split()[1].split("/")[0])
        elif line.strip().startswith("node "):
            header = line
        elif line.strip().startswith("TOTAL") and header and current is not None:
            # match by order, not by position: a value wider than its column (e.g. loss -0.0006851) shifts the
            # rest of the row. The TOTAL row has every header column but `state`, which it leaves empty.
            names, values = header.split()[1:], line.split()[1:]
            if len(values) == len(names) - 1 and "state" in names:
                names = [n for n in names if n != "state"]
            if len(values) != len(names):
                raise ValueError(f"{log}: round {current}: {len(values)} values for {len(names)} columns")
            row = {}
            for name, value in zip(names, values):
                try:
                    row[name] = float(value.rstrip("s"))
                except ValueError:
                    pass  # "-": not available in this round
            rounds[current] = row
    return rounds


def window_mean(rounds: dict[int, dict[str, float]], key: str, lo: int, hi: int) -> float:
    values = [rounds[r][key] for r in range(lo, hi + 1) if r in rounds and key in rounds[r]]
    return sum(values) / len(values) if len(values) == hi - lo + 1 else math.nan


def mean_se(values: list[float]) -> tuple[float, float]:
    values = [v for v in values if not math.isnan(v)]
    if not values:
        return math.nan, math.nan
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, math.nan
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var / len(values))


def ci95(mean: float, se: float, n: int) -> tuple[float, float]:
    """95% CI of a mean from n values: mean +- t_{0.975}(n - 1) * se (nan below 2 values or beyond the table)."""
    if n < 2 or n - 1 > len(T975) or math.isnan(se):
        return math.nan, math.nan
    half = T975[n - 2] * se
    return mean - half, mean + half


def fmt(x: float, width: int = 6) -> str:
    return f"{'-':>{width}}" if math.isnan(x) else f"{x:{width}.2f}"


def k_order(k: str) -> float:
    return math.inf if k in ("inf", "-") else float(k)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", default=["gen-*"], help="run name globs (default: gen-*)")
    parser.add_argument("--ref", action="append", default=[], help="GLOB=HEAD:K, older runs joining a cell")
    parser.add_argument("--logs", default="logs")
    parser.add_argument("--rounds", default="41-50", help="window of rounds to average (default 41-50)")
    parser.add_argument("--pairs", default="nav-id:h5,nav-id:h128", help="paired differences A:B (A - B)")
    parser.add_argument("--extra", default="", help="comma-separated logged metrics: per-cell mean of the window")
    args = parser.parse_args()
    lo, hi = map(int, args.rounds.split("-"))

    names = sorted(p.name for p in Path(args.logs).iterdir() if (p / "node_0.log").exists())
    runs = {}  # name -> (head, K, seed)
    for name in names:
        m = GEN_NAME.match(name)
        if m and any(fnmatch.fnmatch(name, g) for g in args.runs):
            head = m["head"] if m["sweep"] == "gen" else f"{m['head']}-{m['sweep']}"
            runs[name] = (head, m["k"], int(m["seed"]))
        elif any(fnmatch.fnmatch(name, g) for g in args.runs) and SEED.search(name):
            runs[name] = (SEED.sub("-", name, count=1), "-", int(SEED.search(name)[1]))
    for spec in args.ref:
        pattern, _, cell = spec.partition("=")
        head, _, k = cell.partition(":")
        for name in names:
            if fnmatch.fnmatch(name, pattern) and SEED.search(name):
                runs[name] = (head, k or "inf", int(SEED.search(name)[1]))
    if not runs:
        raise SystemExit(f"no run matches {args.runs} in {args.logs}/")

    extra = [m for m in args.extra.split(",") if m]
    cells, extras = {}, {}  # (head, K) -> {seed: (test, pool)}; (head, K, metric) -> [window mean per seed]
    print(f"Per run (mean of rounds {lo}-{hi}; '-' = window incomplete or metric absent)")
    print(f"{'run':52s} {'head':>14s} {'K':>4s} {'seed':>4s} {'rounds':>6s} {'test':>6s} {'pool':>6s} {'gap':>6s}")
    for name, (head, k, seed) in sorted(runs.items(), key=lambda kv: (kv[1][0], k_order(kv[1][1]), kv[1][2])):
        rounds = read_rounds(Path(args.logs) / name / "node_0.log")
        test, pool = window_mean(rounds, TEST, lo, hi), window_mean(rounds, POOL, lo, hi)
        cells.setdefault((head, k), {})[seed] = (test, pool)
        for metric in extra:
            extras.setdefault((head, k, metric), []).append(window_mean(rounds, metric, lo, hi))
        print(f"{name:52s} {head:>14s} {k:>4s} {seed:>4d} {len(rounds):>6d} {fmt(test)} {fmt(pool)} {fmt(pool - test)}")

    print(f"\nPer cell (mean +- SE over seeds; overfit = seeds with pool - test >= {OVERFIT_GAP})")
    print(f"{'head':>14s} {'K':>4s} {'n':>2s} {'test':>14s} {'pool':>14s} {'gap':>14s} {'overfit':>8s}")
    for (head, k), seeds in sorted(cells.items(), key=lambda kv: (kv[0][0], k_order(kv[0][1]))):
        tests = [t for t, _ in seeds.values()]
        pools = [p for _, p in seeds.values()]
        gaps = [p - t for t, p in seeds.values()]
        (tm, ts), (pm, ps), (gm, gs) = mean_se(tests), mean_se(pools), mean_se(gaps)
        n_gap = sum(1 for g in gaps if not math.isnan(g))
        overfit = f"{sum(1 for g in gaps if g >= OVERFIT_GAP)}/{n_gap}" if n_gap else "-"
        more = "".join(f"  {metric} {fmt(mean_se(extras[(head, k, metric)])[0])}" for metric in extra)
        print(f"{head:>14s} {k:>4s} {len(seeds):>2d} {fmt(tm)} +- {fmt(ts, 4)} {fmt(pm)} +- {fmt(ps, 4)} "
              f"{fmt(gm)} +- {fmt(gs, 4)} {overfit:>8s}{more}")

    print("\nPaired test differences per seed (A - B, same seed = same pool; CI95 = mean +- t_0.975(n-1) SE)")
    for pair in args.pairs.split(","):
        a, _, b = pair.partition(":")
        for k in sorted({k for h, k in cells if h == a} & {k for h, k in cells if h == b}, key=k_order):
            common = sorted(set(cells[(a, k)]) & set(cells[(b, k)]))
            diffs = [cells[(a, k)][s][0] - cells[(b, k)][s][0] for s in common]
            mean, se = mean_se(diffs)
            per_seed = " ".join(f"s{s}:{d:+.2f}" for s, d in zip(common, diffs) if not math.isnan(d))
            t = mean / se if se and not math.isnan(se) else math.nan  # paired t, n - 1 degrees of freedom
            n = sum(1 for d in diffs if not math.isnan(d))
            lo_ci, hi_ci = ci95(mean, se, n)
            print(f"  {a} - {b}  K={k:>4s}: mean {fmt(mean)} +- {fmt(se, 4)}  t {fmt(t, 5)} (n={n})  "
                  f"CI95 [{fmt(lo_ci, 5)}, {fmt(hi_ci, 5)}]  [{per_seed}]")


if __name__ == "__main__":
    main()
