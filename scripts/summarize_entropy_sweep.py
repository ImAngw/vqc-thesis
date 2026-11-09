"""Results of the entropy-target sweep (scripts/sweep_entropy.py): test return of h5 and h128 at K=25 as a function of
the entropy they were held at, with J and Fourier as references.

Points of each MLP curve: free (the schedule of the main experiment, gen-<head>-k25), the four sweep targets
(ent-sweep-<head>-t<target>-k25) and the target of the report's matched control, 1.425 nats (ent-<head>-k25).
All numbers are means over the rounds 41-50 of each run, then over seeds. The paired differences J - X use the same
seed (same pool, same initial encoder) and a 95% interval mean +- t_0.975(n-1) SE, as in the report.

    python scripts/summarize_entropy_sweep.py --logs data/logs --logs logs
    python scripts/summarize_entropy_sweep.py --logs data/logs --logs logs --tex thesis/generated --fig thesis/figures/entropy_sweep.pdf

The figure needs matplotlib. "Entropy reached" is the training entropy of the policy (column `entropy` of the log), the
quantity the adaptive coefficient controls and the one reported for J (1.425 nats).
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_gen_pilot import POOL, TEST, ci95, mean_se, read_rounds, window_mean  # noqa: E402

SUCCESS, ENTROPY = "sample/success_rate", "entropy"
SWEEP = re.compile(r"^ent-sweep-(?P<head>h5|h128)-t(?P<target>[0-9.]+)-k25-s(?P<seed>\d+)-")
OFF_TARGET = 0.1  # nats: a seed is "off target" when its training entropy differs from the target by more
MATCHED = 1.425  # target of the matched controls of the report (entropy of J)
HEAD_LABEL = {"h5": "MLP h5", "h128": "MLP h128"}
REFERENCES = {"J": "gen-nav-id-k25", "Fourier": "gen-four-k25"}
COLOR = {"h5": "#1baf7a", "h128": "#eda100", "J": "#2a78d6", "Fourier": "#eb6834"}  # palette of plot_gen_curve.py


def run_metrics(log: Path, lo: int, hi: int) -> dict[str, float]:
    rounds = read_rounds(log)
    return {"test": window_mean(rounds, TEST, lo, hi), "pool": window_mean(rounds, POOL, lo, hi),
            "success": window_mean(rounds, SUCCESS, lo, hi), "entropy": window_mean(rounds, ENTROPY, lo, hi)}


def collect(log_dirs: list[Path], lo: int, hi: int) -> dict[tuple[str, str], dict[int, dict[str, float]]]:
    """{(curve, point): {seed: metrics}}; curve in h5, h128, J, Fourier; point is "free", "1.425", a sweep target or "-"."""
    cells: dict = {}
    patterns = {f"gen-{h}-k25": (h, "free") for h in HEAD_LABEL} | {f"ent-{h}-k25": (h, str(MATCHED)) for h in HEAD_LABEL}
    patterns |= {prefix: (name, "-") for name, prefix in REFERENCES.items()}
    for logs in log_dirs:
        if not logs.exists():
            continue
        for path in sorted(logs.iterdir()):
            log = path / "node_0.log"
            if not log.exists():
                continue
            m = SWEEP.match(path.name)
            if m:
                key, seed = (m["head"], m["target"]), int(m["seed"])
            else:
                hit = re.match(r"^((?:gen|ent)-[a-z0-9-]+?-k25)-s(\d+)-7\.5k-cpu-outer$", path.name)
                if not hit or hit[1] not in patterns:
                    continue
                key, seed = patterns[hit[1]], int(hit[2])
            metrics = run_metrics(log, lo, hi)
            if not math.isnan(metrics["test"]):  # incomplete window: the run has not reached round `hi`
                cells.setdefault(key, {})[seed] = metrics
    return cells


def point_order(point: str) -> float:
    return -1.0 if point == "free" else 0.0 if point == "-" else float(point)


def summary(seeds: dict[int, dict[str, float]], key: str) -> tuple[float, float]:
    return mean_se([m[key] for m in seeds.values()])


def paired(j: dict, other: dict) -> tuple[float, tuple[float, float], int]:
    common = sorted(set(j) & set(other))
    diffs = [j[s]["test"] - other[s]["test"] for s in common]
    mean, se = mean_se(diffs)
    return mean, ci95(mean, se, len(diffs)), len(diffs)


def machine_check(cells: dict, runs_path: Path) -> dict[str, tuple[float, int]]:
    """Mean residual (return - mean of its cell) per machine, over the sweep runs: should be near 0 for both."""
    runs = json.loads(runs_path.read_text())
    resid: dict[str, list[float]] = {}
    for (curve, point), seeds in cells.items():
        if curve not in HEAD_LABEL or point in ("free", str(MATCHED)):
            continue
        mean = summary(seeds, "test")[0]
        for seed, m in seeds.items():
            name = f"ent-sweep-{curve}-t{point}-k25-s{seed}-7.5k-cpu-outer"
            if name in runs:
                resid.setdefault(runs[name]["machine"], []).append(m["test"] - mean)
    return {k: (sum(v) / len(v), len(v)) for k, v in resid.items()}


def fmt_pm(mean: float, se: float, digits: int = 2) -> str:
    return "-" if math.isnan(mean) else f"{mean:.{digits}f} ± {se:.{digits}f}" if not math.isnan(se) else f"{mean:.{digits}f}"


def print_tables(cells: dict, machines: dict | None) -> None:
    print("Table A: entropy sweep at K=25 (mean ± SE over seeds, rounds 41-50)")
    print(f"{'curve':10s} {'target':>7s} {'n':>2s} {'test return':>14s} {'success %':>11s} {'entropy reached':>16s} "
          f"{'off target':>10s} {'gap':>13s}")
    order = [(c, p) for c in (*HEAD_LABEL, *REFERENCES) for p in sorted({p for (cc, p) in cells if cc == c}, key=point_order)]
    for curve, point in order:
        seeds = cells[(curve, point)]
        gap = [m["pool"] - m["test"] for m in seeds.values()]
        off = "-"
        if point not in ("free", "-"):
            off = f"{sum(abs(m['entropy'] - float(point)) > OFF_TARGET for m in seeds.values())}/{len(seeds)}"
        print(f"{curve:10s} {point:>7s} {len(seeds):>2d} {fmt_pm(*summary(seeds, 'test')):>14s} "
              f"{fmt_pm(summary(seeds, 'success')[0] * 100, summary(seeds, 'success')[1] * 100, 1):>11s} "
              f"{fmt_pm(*summary(seeds, 'entropy')):>16s} {off:>10s} {fmt_pm(*mean_se(gap)):>13s}")
    print("\nTable B: paired difference J - MLP per seed (mean [95% CI], n pairs)")
    j = cells.get(("J", "-"), {})
    for head in HEAD_LABEL:
        for point in sorted({p for (c, p) in cells if c == head}, key=point_order):
            mean, (lo, hi), n = paired(j, cells[(head, point)])
            marker = "*" if n >= 2 and (lo > 0 or hi < 0) else " "
            print(f"{head:6s} {point:>7s} n={n:<2d} {mean:+.2f} [{lo:+.2f}, {hi:+.2f}] {marker}")
    print("(* = interval excludes 0; otherwise: no difference detected, not equivalence)")
    if machines:
        print("\nMachine check: mean residual of the sweep runs (return - cell mean), per machine")
        for name, (value, n) in sorted(machines.items()):
            print(f"  {name:6s} n={n:<3d} {value:+.2f}")


def write_tex(cells: dict, out_dir: Path) -> None:
    """LaTeX tables for the thesis (booktabs); decimal point as in the report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    head = "% generated by scripts/summarize_entropy_sweep.py -- do not edit\n"
    rows = [head + "\\begin{tabular}{lrrrrrr}\n\\toprule\n testa & target & seed & ritorno di test & successo (\\%) & entropia raggiunta & fuori obiettivo \\\\\n\\midrule"]
    for curve in (*HEAD_LABEL, *REFERENCES):
        for point in sorted({p for (c, p) in cells if c == curve}, key=point_order):
            seeds = cells[(curve, point)]
            target = {"free": "libero", "-": "--"}.get(point, point)
            name = HEAD_LABEL.get(curve, curve)
            succ = summary(seeds, "success")
            off = "--" if point in ("free", "-") else f"{sum(abs(m['entropy'] - float(point)) > OFF_TARGET for m in seeds.values())}/{len(seeds)}"
            rows.append(f" {name} & {target} & {len(seeds)} & ${fmt_pm(*summary(seeds, 'test'))}$ & "
                        f"${fmt_pm(succ[0] * 100, succ[1] * 100, 1)}$ & ${fmt_pm(*summary(seeds, 'entropy'))}$ & {off} \\\\")
    rows.append("\\bottomrule\n\\end{tabular}\n")
    (out_dir / "sweep_table.tex").write_text("\n".join(rows).replace("±", "\\pm"))
    j = cells.get(("J", "-"), {})
    rows = [head + "\\begin{tabular}{lrrr}\n\\toprule\n testa & target & coppie & differenza J $-$ MLP [IC 95\\%] \\\\\n\\midrule"]
    for h in HEAD_LABEL:
        for point in sorted({p for (c, p) in cells if c == h}, key=point_order):
            mean, (lo, hi), n = paired(j, cells[(h, point)])
            target = "libero" if point == "free" else point
            bold = lo > 0 or hi < 0
            cell = f"{mean:+.2f} [{lo:+.2f}, {hi:+.2f}]"
            rows.append(f" {HEAD_LABEL[h]} & {target} & {n} & {'\\textbf{' + cell + '}' if bold else cell} \\\\")
    rows.append("\\bottomrule\n\\end{tabular}\n")
    (out_dir / "sweep_paired.tex").write_text("\n".join(rows))


def draw(cells: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("pdf")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["pdf.fonttype"] = 42  # embedded TrueType, no transparency: PDF/A friendly
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for head in HEAD_LABEL:
        pts = [(summary(cells[(head, p)], "entropy"), summary(cells[(head, p)], "test"), p)
               for p in sorted({p for (c, p) in cells if c == head}, key=point_order)]
        pts.sort(key=lambda p: p[0][0])
        xs, ys = [e[0] for e, _, _ in pts], [t[0] for _, t, _ in pts]
        ax.errorbar(xs, ys, xerr=[e[1] for e, _, _ in pts], yerr=[t[1] for _, t, _ in pts], color=COLOR[head], marker="o",
                    markersize=4, linewidth=0, elinewidth=0.8, capsize=2, label=HEAD_LABEL[head])
        for (e, t, p) in pts:  # each point is labelled with its target
            ax.annotate("lib." if p == "free" else p, (e[0], t[0]), xytext=(4, 5), textcoords="offset points",
                        fontsize=6, color=COLOR[head])
    for name, marker in (("J", "D"), ("Fourier", "s")):
        if (name, "-") in cells:
            (em, es), (tm, ts) = summary(cells[(name, "-")], "entropy"), summary(cells[(name, "-")], "test")
            ax.errorbar([em], [tm], xerr=[es], yerr=[ts], color=COLOR[name], marker=marker, markersize=6, linestyle="none",
                        capsize=2, label=name)
    ax.set_xlabel("entropia di training raggiunta (nat)")
    ax.set_ylabel("ritorno di test, K = 25")
    ax.set_title("etichette: entropia obiettivo (lib. = senza controllo)", fontsize=7, loc="left")
    ax.axvline(math.log(8), color="#999999", linewidth=0.8, linestyle=":")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", action="append", default=None, help="log directory (repeatable; default data/logs)")
    ap.add_argument("--rounds", default="41-50")
    ap.add_argument("--runs", default=str(Path(__file__).resolve().parents[1] / "configs" / "runs.json"))
    ap.add_argument("--tex", help="directory for the LaTeX tables")
    ap.add_argument("--fig", help="output PDF of the figure")
    args = ap.parse_args()
    lo, hi = map(int, args.rounds.split("-"))
    cells = collect([Path(d) for d in (args.logs or ["data/logs"])], lo, hi)
    if not cells:
        sys.exit("no complete run found")
    print_tables(cells, machine_check(cells, Path(args.runs)))
    if args.tex:
        write_tex(cells, Path(args.tex))
    if args.fig:
        draw(cells, Path(args.fig))


if __name__ == "__main__":
    main()
