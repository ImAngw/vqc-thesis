"""Tables 11-12 of the report (part B): test return of J without noise (NR), with the effective depolarizing model
(MDE, scripts/eval_noise.py) and with every action from a full Aer simulation (AER, scripts/eval_aer.py), per IBM
backend, and the paired per-seed differences AER - NR and AER - MDE with their 95% CI.

Reads quantum_heads/analysis/runs/noise/<run>.json (clean and calibrated rows) and .../aer/<run>.json.

    python scripts/summarize_noise.py
"""

import argparse
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize_gen_pilot import ci95, mean_se  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKENDS = ("fake_torino", "fake_brisbane")


def load(noise_dir: str, pattern: str) -> dict[str, dict]:
    """{run: {"NR": {backend: return}, "MDE": {...}, "AER": {...}}} for the runs that have both files."""
    rows = {}
    for path in sorted(glob.glob(os.path.join(noise_dir, pattern))):
        run = os.path.basename(path)[:-len(".json")]
        aer_path = os.path.join(noise_dir, "aer", run + ".json")
        if not os.path.exists(aer_path):
            continue
        base = json.load(open(path))["results"]
        aer = json.load(open(aer_path))["results"]
        rows[run] = {"NR": {b: base["clean"]["return_mean"] for b in BACKENDS},
                     "MDE": {b: base[b]["return_mean"] for b in BACKENDS},
                     "AER": {b: aer[b]["return_mean"] for b in BACKENDS}}
    return rows


def fmt_ci(values: list[float]) -> str:
    mean, se = mean_se(values)
    lo, hi = ci95(mean, se, len(values))
    return f"{mean:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--noise-dir", default=os.path.join(ROOT, "quantum_heads", "analysis", "runs", "noise"))
    ap.add_argument("--runs", default="gen-nav-id-k25-s*-7.5k-cpu-outer.json")
    args = ap.parse_args()
    rows = load(args.noise_dir, args.runs)
    n = len(rows)
    print(f"Table 11: test return of J, mean +- SE over {n} seeds (same weights and layouts in every row)")
    print(f"{'simulation':34s}" + "".join(f"{b:>20s}" for b in BACKENDS))
    for key, label in (("NR", "NR  no noise"), ("MDE", "MDE effective depolarizing model"), ("AER", "AER full Aer simulation")):
        cells = []
        for b in BACKENDS:
            mean, se = mean_se([r[key][b] for r in rows.values()])
            cells.append(f"{mean:.2f} +- {se:.2f}")
        print(f"{label:34s}" + "".join(f"{c:>20s}" for c in cells))
    print(f"\nTable 12: paired differences per seed, mean [95% CI], n = {n}")
    print(f"{'comparison':34s}" + "".join(f"{b:>26s}" for b in BACKENDS))
    for a, b_, label in (("AER", "NR", "AER - NR  cost of the noise"), ("AER", "MDE", "AER - MDE  error of the model")):
        cells = [fmt_ci([r[a][b] - r[b_][b] for r in rows.values()]) for b in BACKENDS]
        print(f"{label:34s}" + "".join(f"{c:>26s}" for c in cells))
    print("\nPer seed AER return: " + "  ".join(
        f"s{run.split('-s')[-1].split('-')[0]} " + "/".join(f"{r['AER'][b]:.2f}" for b in BACKENDS) for run, r in rows.items()))
    if n < 2 or math.isnan(mean_se([r["AER"][BACKENDS[0]] for r in rows.values()])[1]):
        sys.exit("fewer than 2 complete seeds")


if __name__ == "__main__":
    main()
