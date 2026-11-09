"""The train.py command that reproduces a run of the report.

configs/runs.json holds, for each of the runs in data/logs, the parameters it was launched with (taken from the run
records of the original runner). All of them share the same schedule: 7500 updates in 50 rounds of 150, outer step
(lr 0.7, Nesterov momentum 0.9, pseudo-gradient in bf16), evaluation every round, CPU.

    python scripts/cell_command.py gen-h5-k25-s3-7.5k-cpu-outer            # one run
    python scripts/cell_command.py 'gen-*-k25-s*' --log-dir logs           # every run matching a glob
    python scripts/cell_command.py --list                                  # cells (runs without the seed)

Seeds: the original runner drew the seed of the environments and of the action sampling from `env_seed` (or from
`seed` when there was none), train.py takes it directly: the command passes the drawn value (docs/REPRODUCE.md).
Runs launched before the environment defaults changed (2026-10-14) have no perception / env_kwargs in their record:
they ran on environment v8, i.e. perception cnn, window 5, 30 obstacles, which the command passes explicitly.
Warm-started runs (ws-*) used options that train.py does not have and are not reproducible from this repository.
"""

import argparse
import fnmatch
import json
import re
import shlex
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SCHEDULE = ["--n-envs", "16", "--n-steps", "64", "--total-updates", "7500", "--inner-steps", "150",
            "--outer-lr", "0.7", "--outer-momentum", "0.9", "--outer-dtype", "bf16", "--eval-every", "150"]
UNSUPPORTED = ("init_from", "critic_warmup")
ENV_V8_UNTIL = "2026-10-14"  # env defaults changed (window 3, 60 obstacles) in the commits of this day
ENV_V8 = ["--perception", "cnn", "--env-kwargs", '{"window": 5, "n_obstacles": 30}']


def drawn_env_seed(e: int) -> int:
    """Seed that the original runner derived from env_seed (or seed) for the vector environment."""
    return int(torch.randint(0, 2**31 - 1, (1,), generator=torch.Generator().manual_seed(e)).item())


def load_runs(path: Path = ROOT / "configs" / "runs.json") -> dict:
    return json.loads(path.read_text())


def command(run: str, runs: dict, log_dir: str = "logs") -> list[str]:
    """Arguments of `python train.py` for `run`; ValueError when train.py cannot reproduce it."""
    rec = runs[run]
    params = dict(rec["params"])
    bad = [k for k in UNSUPPORTED if k in params]
    if bad:
        raise ValueError(f"{run}: uses {', '.join(bad)}, which train.py does not have (warm-started run)")
    params.pop("anneal_by", None)  # train.py anneals the entropy on the update count
    seed = rec["seed"]
    e = int(params.pop("env_seed", seed))
    args = ["--run-name", run, "--log-dir", log_dir, "--seed", str(seed), "--env-seed", str(drawn_env_seed(e))]
    if "layout_pool" in params:
        args += ["--layout-pool-seed", str(e)]
    args += SCHEDULE
    if "env_kwargs" not in params and rec["started"] < ENV_V8_UNTIL:
        args += ENV_V8
    for key, value in sorted(params.items()):
        args += ["--" + key.replace("_", "-"), str(value)]
    return args


def cells(runs: dict) -> list[str]:
    return sorted({re.sub(r"-s\d+(?=-|$)", "-s*", name) for name in runs})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run name or glob")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--list", action="store_true", help="list the cells")
    args = ap.parse_args()
    runs = load_runs()
    if args.list or not args.runs:
        print("\n".join(cells(runs)))
        return
    names = sorted(n for n in runs if any(fnmatch.fnmatch(n, g) or fnmatch.fnmatch(n, g + "*") for g in args.runs))
    if not names:
        sys.exit(f"no run matches {args.runs}")
    for name in names:
        try:
            print("python train.py " + " ".join(shlex.quote(a) for a in command(name, runs, args.log_dir)))
        except ValueError as err:
            print(f"# {err}", file=sys.stderr)


if __name__ == "__main__":
    main()
