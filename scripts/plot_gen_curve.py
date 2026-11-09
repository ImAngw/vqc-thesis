"""Curve in K of the generalization study (docs/design/SPEC-generalization-full.md, analysis "Curve").

Left: test return (sample/return_mean on fresh layouts, mean of rounds 41-50) vs K per head, mean +- SE over seeds;
entropy-matched cells (ent-*) as hollow markers of their head's color, drawn only when all their seeds are complete.
Right: paired difference J - X vs K (same seed = same pool) with its 95% CI, for X in h5, h128, Fourier.

Reads logs/ like summarize_gen_pilot.py (same name parsing, --ref cells, a later run of a cell replaces an earlier
one, e.g. a -r rerun). Needs matplotlib, which is not in requirements.txt: run it with an interpreter that has it.

    python scripts/plot_gen_curve.py --out docs/figures/partA_curve_k.png
"""

import argparse
import fnmatch
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_gen_pilot import GEN_NAME, SEED, TEST, ci95, mean_se, read_rounds, window_mean  # noqa: E402

KS = ["5", "25", "200", "inf"]
# head -> (label, color, marker); colors = categorical slots 1-4 of the dataviz reference palette (validated, light)
HEADS = {
    "nav-id": ("J (VQC)", "#2a78d6", "o"),
    "four": ("Fourier", "#eb6834", "s"),
    "h5": ("MLP h5", "#1baf7a", "^"),
    "h128": ("MLP h128", "#eda100", "D"),
}
REF_DEFAULT = "nav-id-hlr2e-3-detc-s*-7.5k-cpu-outer=nav-id:inf"
# cells never drawn: the entropy-matched K=5 control failed (entropy off target, pool not fitted; see the spec)
SKIP_DEFAULT = "four-ent:5,h5-ent:5"


TEXT = {
    "en": {"x": "K = training layouts", "y1": "test return (fresh layouts, rounds {lo}-{hi})",
           "t1": "Test return vs K (mean ± SE; hollow = entropy-matched)", "y2": "paired difference J − X (test return)",
           "t2": "J − X per seed (mean, 95% CI)"},
    "it": {"x": "K = layout di training", "y1": "ritorno di test (layout nuovi, round {lo}-{hi})",
           "t1": "Ritorno di test in funzione di K (media ± ES; vuoti = entropia di J)",
           "y2": "differenza appaiata J − X (ritorno di test)", "t2": "J − X per seed (media, IC 95%)"},
}


def collect(logs: Path, globs: list[str], refs: list[str], lo: int, hi: int) -> dict:
    """{(head, K): {seed: test}} with the same cell rules as summarize_gen_pilot.py."""
    names = sorted(p.name for p in logs.iterdir() if (p / "node_0.log").exists())
    runs = {}
    for name in names:
        m = GEN_NAME.match(name)
        if m and any(fnmatch.fnmatch(name, g) for g in globs):
            head = m["head"] if m["sweep"] == "gen" else f"{m['head']}-{m['sweep']}"
            runs[name] = (head, m["k"], int(m["seed"]))
    for spec in refs:
        pattern, _, cell = spec.partition("=")
        head, _, k = cell.partition(":")
        for name in names:
            if fnmatch.fnmatch(name, pattern) and SEED.search(name):
                runs[name] = (head, k or "inf", int(SEED.search(name)[1]))
    cells = {}
    for name, (head, k, seed) in sorted(runs.items(), key=lambda kv: (kv[1][0], kv[1][1], kv[1][2])):
        cells.setdefault((head, k), {})[seed] = window_mean(read_rounds(logs / name / "node_0.log"), TEST, lo, hi)
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs", default="logs")
    parser.add_argument("--ref", action="append", default=None, help=f"GLOB=HEAD:K (default {REF_DEFAULT})")
    parser.add_argument("--rounds", default="41-50")
    parser.add_argument("--seeds", type=int, default=10, help="seeds a cell needs to be drawn (default 10)")
    parser.add_argument("--out", default="docs/figures/partA_curve_k.png")
    parser.add_argument("--skip", default=SKIP_DEFAULT, help=f"HEAD:K cells not drawn (default {SKIP_DEFAULT})")
    parser.add_argument("--lang", choices=("en", "it"), default="en", help="language of the labels (default en)")
    parser.add_argument("--size", default="11,4.4", help="figure size in inches W,H (default 11,4.4)")
    parser.add_argument("--font-size", type=float, default=10, help="base font size (default 10)")
    args = parser.parse_args()
    t = TEXT[args.lang]
    lo, hi = map(int, args.rounds.split("-"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = collect(Path(args.logs), ["gen-*", "ent-*"], args.ref or [REF_DEFAULT], lo, hi)
    skip = {tuple(c.split(":")) for c in args.skip.split(",") if c}
    complete = lambda c: (c in cells and c not in skip  # noqa: E731
                          and sum(not math.isnan(v) for v in cells[c].values()) >= args.seeds)

    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    plt.rcParams.update({"font.size": args.font_size, "pdf.fonttype": 42, "axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted,
                         "ytick.color": muted, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=tuple(float(v) for v in args.size.split(",")), facecolor=surface)
    x = {k: i for i, k in enumerate(KS)}
    xt = ["5", "25", "200", "∞"]

    # left: test return vs K
    ends = []  # (y, label, x) of each curve's last point, for direct labels spread apart below
    for head, (label, color, marker) in HEADS.items():
        pts = [(x[k], *mean_se(list(cells[(head, k)].values()))) for k in KS if complete((head, k))]
        xs, ms, ses = zip(*pts)
        ax1.errorbar(xs, ms, yerr=ses, color=color, marker=marker, ms=7, lw=2, capsize=3, mec=surface, mew=1.5,
                     label=label, zorder=3)
        ends.append((ms[-1], label, xs[-1]))
        for k in KS:
            if complete((f"{head}-ent", k)):
                m, se = mean_se(list(cells[(f"{head}-ent", k)].values()))
                ax1.errorbar(x[k] + 0.12, m, yerr=se, color=color, marker=marker, ms=7, mfc=surface, mew=1.5,
                             capsize=3, lw=0, elinewidth=1.5, zorder=3)
                ax1.annotate("ent", (x[k] + 0.12, m), xytext=(9, 0), textcoords="offset points", va="center",
                             color=muted, fontsize=args.font_size * 0.8)
    # direct labels at the right end, pushed apart to a minimum gap so close curves (J, Fourier at K=inf) stay legible
    y0, y1 = ax1.get_ylim()
    gap, placed = 0.055 * (y1 - y0), []
    for y, label, xe in sorted(ends):
        y_lab = max(y, placed[-1] + gap) if placed else y
        placed.append(y_lab)
        ax1.annotate(label, (xe, y), xytext=(xe + 0.12, y_lab), textcoords="data", va="center", color=ink, fontsize=args.font_size * 0.9)
    ax1.axhline(0, color=muted, lw=0.8, zorder=1)
    ax1.set_xticks(range(len(KS)), xt)
    ax1.set_xlim(-0.3, len(KS) - 0.4)
    ax1.set_xlabel(t["x"])
    ax1.set_ylabel(t["y1"].format(lo=lo, hi=hi))
    ax1.set_title(t["t1"], fontsize=args.font_size, loc="left", color=ink)
    ax1.grid(axis="y", color=grid, lw=0.8)
    ax1.legend(frameon=False, fontsize=args.font_size * 0.9, loc="lower right")

    # right: J - X with CI95
    ax2.axhline(0, color=muted, lw=0.8, zorder=1)
    others = [h for h in HEADS if h != "nav-id"]
    for j, head in enumerate(others):
        label, color, marker = HEADS[head]
        pts = []
        for k in KS:
            if not (complete(("nav-id", k)) and complete((head, k))):
                continue
            common = sorted(set(cells[("nav-id", k)]) & set(cells[(head, k)]))
            diffs = [cells[("nav-id", k)][s] - cells[(head, k)][s] for s in common]
            diffs = [d for d in diffs if not math.isnan(d)]
            m, se = mean_se(diffs)
            lo_ci, hi_ci = ci95(m, se, len(diffs))
            pts.append((x[k] + (j - 1) * 0.12, m, m - lo_ci, hi_ci - m))
        xs, ms, el, eh = zip(*pts)
        ax2.errorbar(xs, ms, yerr=[el, eh], color=color, marker=marker, ms=7, lw=2, capsize=3, mec=surface, mew=1.5,
                     label=f"J − {label}", zorder=3)
    ax2.set_xticks(range(len(KS)), xt)
    ax2.set_xlim(-0.4, len(KS) - 0.6)
    ax2.set_xlabel(t["x"])
    ax2.set_ylabel(t["y2"])
    ax2.set_title(t["t2"], fontsize=args.font_size, loc="left", color=ink)
    ax2.grid(axis="y", color=grid, lw=0.8)
    ax2.legend(frameon=False, fontsize=args.font_size * 0.9, loc="upper right")

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor=surface)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
