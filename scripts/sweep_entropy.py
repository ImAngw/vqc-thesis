"""Entropy-target sweep for the MLP heads (thesis chapter on the entropy control).

The entropy-matched controls of the report (ent-h5-k25, ent-h128-k25) hold the training entropy at 1.425 nats, the
entropy of J, chosen after the fact. The sweep makes the target the variable of the comparison: the same two heads at
K=25, the same adaptive entropy coefficient, with four targets fixed before looking at any result.

    python scripts/sweep_entropy.py --list                      # the 80 runs and the machine of each
    python scripts/sweep_entropy.py --write                     # add them to configs/runs.json
    python scripts/sweep_entropy.py --commands mac              # train.py commands of the runs of one machine
    python scripts/sweep_entropy.py --commands nodo2 --log-dir logs
    python scripts/sweep_entropy.py --run mac --jobs 4          # train the runs of one machine, 4 at a time

The runs are single-threaded in practice: the result of a run is the same with 1, 2 or the default number of torch
threads (checked on 4 rounds), so several can share a machine without changing their numbers.

Every (head, target) cell has seeds 0-9 with the pool of seed s, as in the report. All 80 runs were trained on the Mac:
the plan was to split each cell between the Mac (seeds 0-4) and nodo2 (seeds 5-9), but nodo2 became unreachable after
its first runs, so the ones of nodo2 were rerun on the Mac. Runs are bit-identical only on the same machine. The machine
of each run is recorded in configs/runs.json.
"""

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cell_command  # noqa: E402

CONFIG = cell_command.ROOT / "configs" / "runs.json"
HEADS = {"h5": "mlp5", "h128": "mlp128"}
TARGETS = (0.4, 0.8, 1.2, 1.8)  # nats, fixed before any result; the maximum is ln 8 = 2.08
SEEDS = tuple(range(10))
MAC_SEEDS = SEEDS  # planned: range(5), the rest on nodo2; nodo2 became unreachable and all runs went to the Mac
STARTED = "2026-11-09"
SUFFIX = "7.5k-cpu-outer"


def machine(seed: int) -> str:
    return "mac" if seed in MAC_SEEDS else "nodo2"


def run_name(head: str, target: float, seed: int) -> str:
    return f"ent-sweep-{head}-t{target}-k25-s{seed}-{SUFFIX}"


def records() -> dict[str, dict]:
    """{run name: record} of the 80 runs, with the same parameters as the ent-h5-k25 / ent-h128-k25 cells."""
    out = {}
    for head, code_name in HEADS.items():
        for target in TARGETS:
            if not 0 < target < math.log(8):
                raise ValueError(f"target {target} outside (0, ln 8)")
            for seed in SEEDS:
                out[run_name(head, target, seed)] = {
                    "params": {"anneal_by": "steps", "detach_critic": 1,
                               "env_kwargs": '{"window": 5, "n_obstacles": 30}', "env_seed": seed,
                               "head": code_name, "layout_pool": 25, "perception": "cnn", "target_entropy": target},
                    "seed": seed, "started": STARTED, "machine": machine(seed)}
    return out


def merge(path: Path = CONFIG) -> int:
    """Add the sweep runs to the run table; returns how many were new. Only the machine of existing sweep entries is updated."""
    runs = json.loads(path.read_text())
    new = {k: v for k, v in records().items() if k not in runs}
    for k, v in records().items():
        if k in runs:
            runs[k]["machine"] = v["machine"]
    runs.update(new)
    path.write_text(json.dumps(runs, indent=1, sort_keys=True))
    return len(new)


def is_done(log_dir: Path, name: str, rounds: int = 50) -> bool:
    """True when the run wrote the log of all its rounds (one TOTAL line per round)."""
    log = Path(log_dir) / name / "node_0.log"
    if not log.exists():
        return False
    return sum(line.lstrip().startswith("TOTAL") for line in log.read_text().splitlines()) >= rounds


def set_aside_partial(log_dir: Path, name: str) -> Path | None:
    """Rename the directory of an interrupted run (the logger refuses an existing one); nothing is deleted."""
    run_dir = Path(log_dir) / name
    if not run_dir.exists():
        return None
    target = run_dir.with_name(f"{name}.partial")
    n = 1
    while target.exists():
        n += 1
        target = run_dir.with_name(f"{name}.partial{n}")
    return run_dir.rename(target)


def run_machine(which: str, jobs: int, log_dir: str) -> int:
    """Train the runs of one machine, `jobs` at a time, skipping the ones already complete. Returns the failures."""
    runs = {**cell_command.load_runs(), **records()}
    todo = [n for n, r in records().items() if r["machine"] == which and not is_done(Path(log_dir), n)]
    print(f"{len(todo)} runs to do on {which}, {jobs} at a time", flush=True)
    Path(log_dir).mkdir(exist_ok=True)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

    def one(name: str) -> int:
        set_aside_partial(Path(log_dir), name)
        cmd = [sys.executable, str(cell_command.ROOT / "train.py")] + cell_command.command(name, runs, log_dir)
        with open(Path(log_dir) / f"{name}.out", "w") as out:
            code = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, env=env, cwd=cell_command.ROOT).returncode
        print(f"{'done' if code == 0 else 'FAILED'} {name}", flush=True)
        return code

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        return sum(code != 0 for code in pool.map(one, todo))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--write", action="store_true", help="add the runs to configs/runs.json")
    ap.add_argument("--commands", choices=("mac", "nodo2"), help="print the train.py commands of one machine")
    ap.add_argument("--run", choices=("mac", "nodo2"), help="train the runs of one machine")
    ap.add_argument("--jobs", type=int, default=4, help="runs at a time with --run")
    ap.add_argument("--log-dir", default="logs")
    args = ap.parse_args()
    if args.write:
        print(f"{merge()} runs added to {CONFIG}")
    if args.list:
        for name, rec in records().items():
            print(f"{name}  {rec['machine']}")
    if args.run:
        sys.exit(1 if run_machine(args.run, args.jobs, args.log_dir) else 0)
    if args.commands:
        runs = {**cell_command.load_runs(), **records()}
        for name, rec in records().items():
            if rec["machine"] == args.commands:
                print("python train.py " + " ".join(shlex.quote(a) for a in cell_command.command(name, runs, args.log_dir)))


if __name__ == "__main__":
    main()
