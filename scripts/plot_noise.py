"""Noise robustness figure of part B: J's test return vs gate noise, with the calibrated IBM backends on the curve.

Reads the per-run JSONs of scripts/eval_noise.py (J, K=25, final weights) and the clean references of
scripts/eval_clean_returns.py (Fourier and h5-ent at K=25, same weights and evaluation). x = depolarizing p2 after
every CX (p1 = p2 / 10 after every rotation), log scale, p2 = 0 drawn at the left edge; the calibrated backends sit
at their effective p2 (mean over the runs; their p1 and readout error are the calibrated ones, so the x position is
approximate): filled = effective noise model (B2), hollow = full Aer simulation of every action (B3, aer/*.json of
scripts/eval_aer.py). y = test return, mean +- SE over the seeds. Needs matplotlib (not in requirements.txt).

    python scripts/plot_noise.py --out docs/figures/partB_noise.png
"""
import argparse
import glob
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P2 = (0.0, 0.001, 0.003, 0.01, 0.03)
ZERO_AT = 3e-4  # where p2 = 0 is drawn on the log axis
BACKENDS = {"fake_torino": ("IBM Torino (calibrated)", "#eda100", "D"),
            "fake_brisbane": ("IBM Brisbane (calibrated)", "#e87ba4", "s")}
REFERENCES = {"gen-four-k25": ("Fourier, no noise", "#eb6834"), "ent-h5-k25": ("MLP h5 at J's entropy, no noise", "#1baf7a")}


TEXT = {
    "en": {"torino": "IBM Torino (calibrated)", "brisbane": "IBM Brisbane (calibrated)", "four": "Fourier, no noise",
           "h5": "MLP h5 at J's entropy, no noise", "j": "J (VQC), depolarizing noise, {n} seeds",
           "eff": "{label}, effective model, p2 {p2:.1f}%", "aer": "{label}, full Aer simulation",
           "x": "two-qubit gate error p2 (depolarizing after every CX; p1 = p2 / 10)",
           "y": "test return (100 fresh layouts)", "title": "J at K=25 under gate noise (mean ± SE over seeds)"},
    "it": {"torino": "IBM Torino (calibrato)", "brisbane": "IBM Brisbane (calibrato)", "four": "Fourier, senza rumore",
           "h5": "MLP h5 all'entropia di J, senza rumore", "j": "J (VQC), rumore depolarizzante, {n} seed",
           "eff": "{label}, modello effettivo, p2 {p2:.1f}%", "aer": "{label}, simulazione Aer completa",
           "x": "errore delle porte a due qubit p2 (depolarizzante dopo ogni CX; p1 = p2 / 10)",
           "y": "ritorno di test (100 layout nuovi)", "title": "J a K=25 sotto rumore delle porte (media ± ES sui seed)"},
}


def mean_se(values):
    m = sum(values) / len(values)
    se = math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1) / len(values)) if len(values) > 1 else 0.0
    return m, se


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--noise-dir", default=os.path.join(ROOT, "quantum_heads", "analysis", "runs", "noise"))
    ap.add_argument("--runs", default="gen-nav-id-k25-s*-7.5k-cpu-outer.json")
    ap.add_argument("--references", default="references_k25.json")
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "figures", "partB_noise.png"))
    ap.add_argument("--lang", choices=("en", "it"), default="en", help="language of the labels (default en)")
    ap.add_argument("--size", default="7.5,4.4", help="figure size in inches W,H (default 7.5,4.4)")
    ap.add_argument("--font-size", type=float, default=10, help="base font size (default 10)")
    args = ap.parse_args()
    t = TEXT[args.lang]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(args.noise_dir, args.runs)))]
    refs = json.load(open(os.path.join(args.noise_dir, args.references)))
    ink, muted, grid, surface, blue = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#2a78d6"
    plt.rcParams.update({"font.size": args.font_size, "pdf.fonttype": 42, "axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted,
                         "ytick.color": muted, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=tuple(float(v) for v in args.size.split(",")), facecolor=surface)

    for prefix, (_, color) in REFERENCES.items():
        label = t["four"] if "four" in prefix else t["h5"]
        m, _ = mean_se([v["return_mean"] for k, v in refs.items() if k.startswith(prefix)])
        ax.axhline(m, color=color, lw=1.5, ls="--", zorder=1)
        # labels between the 0.1% and 0.3% points, under the line: no marker or error bar there
        ax.annotate(f"{label} ({m:.2f})", (0.0017, m), xytext=(0, -5), textcoords="offset points", ha="center",
                    va="top", color=ink, fontsize=args.font_size * 0.85)

    xs = [ZERO_AT if p == 0 else p for p in P2]
    stats = [mean_se([r["results"][f"p2={p:g}" if p else "p2=0"]["return_mean"] for r in rows]) for p in P2]
    ax.errorbar(xs, [m for m, _ in stats], yerr=[s for _, s in stats], color=blue, marker="o", ms=7, lw=2,
                capsize=3, mec=surface, mew=1.5, zorder=3, label=t["j"].format(n=len(rows)))

    aer = {}
    for path in glob.glob(os.path.join(args.noise_dir, "aer", args.runs)):
        row = json.load(open(path))
        aer[row["run"]] = row["results"]
    for name, (_, color, marker) in BACKENDS.items():
        label = t["torino"] if name == "fake_torino" else t["brisbane"]
        p2 = sum(r["calibration"][name]["p2"] for r in rows) / len(rows)
        m, se = mean_se([r["results"][name]["return_mean"] for r in rows])
        ax.errorbar([p2], [m], yerr=[se], color=color, marker=marker, ms=9, lw=0, elinewidth=1.5, capsize=3,
                    mec=surface, mew=1.5, zorder=4, label=t["eff"].format(label=label, p2=100 * p2))
        full = [aer[r["run"]][name]["return_mean"] for r in rows if name in aer.get(r["run"], {})]
        if len(full) == len(rows):  # full Aer (B3), drawn only when every seed has it, hollow, just left of B2
            m, se = mean_se(full)
            ax.errorbar([p2 / 1.15], [m], yerr=[se], color=color, marker=marker, ms=9, mfc=surface, mew=1.8,
                        lw=0, elinewidth=1.5, capsize=3, zorder=4, label=t["aer"].format(label=label))

    ax.set_xscale("log")
    ax.set_xticks(xs, ["0", "0.1%", "0.3%", "1%", "3%"])
    ax.minorticks_off()
    ax.set_xlim(ZERO_AT / 1.6, 0.05)
    ax.set_xlabel(t["x"])
    ax.set_ylabel(t["y"])
    ax.set_title(t["title"], fontsize=args.font_size, loc="left", color=ink)
    ax.grid(axis="y", color=grid, lw=0.8)
    ax.set_ylim(1.7, 4.8)
    ax.legend(frameon=False, fontsize=args.font_size * 0.85, loc="center left", bbox_to_anchor=(0.0, 0.42))
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor=surface)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
