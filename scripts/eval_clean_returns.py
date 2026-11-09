"""Clean test return of trained runs from their final weights (model_latest.pt), as eval_noise.py's "clean" row.

The reference lines of the noise figure: the same evaluation (100 fresh layouts, eval seeds 10000-10099, sampled
actions) for heads other than J, so they compare with J's noisy returns weight for weight.

    python scripts/eval_clean_returns.py 'gen-four-k25-s*-7.5k-cpu-outer=fourier4' \
        'ent-h5-k25-s*-7.5k-cpu-outer=mlp5' --out quantum_heads/analysis/runs/noise/references_k25.json
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "quantum_heads", "analysis")]

import policy_equivalence as pe  # noqa: E402
import torch  # noqa: E402
from rl_train.config import A2CConfig  # noqa: E402
from rl_train.policy import build_policy, encoder_kwargs  # noqa: E402
from train import HEADS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("specs", nargs="+", help="GLOB=HEAD: run name glob under --checkpoints and its registered head")
    ap.add_argument("--checkpoints", default=os.path.join(ROOT, "checkpoints"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    out = {}
    for spec in args.specs:
        pattern, _, head = spec.partition("=")
        runs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(args.checkpoints, pattern)))
        for run in runs:
            cfg = A2CConfig(head=head, **pe.RUN_CONFIG)
            model = build_policy(HEADS[head], **encoder_kwargs(cfg))
            state = torch.load(os.path.join(args.checkpoints, run, "model_latest.pt"), map_location="cpu",
                               weights_only=True)
            model.load_state_dict(state)
            model.detach_critic = True
            out[run] = pe.test_eval(model.eval())[0]
            print(f"{run:40s} {head:10s} return {out[run]['return_mean']:.2f}")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
