import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("summarize_entropy_sweep", ROOT / "scripts" / "summarize_entropy_sweep.py")
ses = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ses)

HEADER = "  node    steps  steps/s       loss  sync/round  state  sample/return_mean  pool/sample/return_mean  sample/success_rate  sample/length_mean       entropy    value_loss"


def write_run(logs, name, test, entropy, rounds=50):
    lines = []
    for r in range(1, rounds + 1):
        lines += [f"round {r}/50", HEADER,
                  f"  local     150     17.8     0.5000        0.0s  active        {test}       {test + 1}          0.50         200.0         {entropy}         1.000",
                  f"  TOTAL     150     17.8     0.5000        0.0s                {test}       {test + 1}          0.50         200.0         {entropy}         1.000"]
    path = logs / name
    path.mkdir(parents=True)
    (path / "node_0.log").write_text("\n".join(lines) + "\n")


def test_collect_paired_difference_and_incomplete_runs(tmp_path):
    for seed in range(3):
        write_run(tmp_path, f"gen-nav-id-k25-s{seed}-7.5k-cpu-outer", 4.0 + seed, 1.4)
        write_run(tmp_path, f"ent-sweep-h5-t0.8-k25-s{seed}-7.5k-cpu-outer", 2.0 + seed, 0.9)
    write_run(tmp_path, "ent-sweep-h5-t0.8-k25-s3-7.5k-cpu-outer", 1.0, 0.9, rounds=40)  # window 41-50 incomplete
    cells = ses.collect([tmp_path], 41, 50)
    assert set(cells[("h5", "0.8")]) == {0, 1, 2}
    mean, (lo, hi), n = ses.paired(cells[("J", "-")], cells[("h5", "0.8")])
    assert n == 3 and abs(mean - 2.0) < 1e-9 and lo == hi == 2.0  # constant difference: zero-width interval
    assert abs(ses.summary(cells[("h5", "0.8")], "entropy")[0] - 0.9) < 1e-9


def test_tex_tables_are_written(tmp_path):
    for seed in range(3):
        write_run(tmp_path / "logs", f"gen-nav-id-k25-s{seed}-7.5k-cpu-outer", 4.0 + seed, 1.4)
        write_run(tmp_path / "logs", f"ent-sweep-h5-t0.8-k25-s{seed}-7.5k-cpu-outer", 0.1 * seed, 0.9)
    cells = ses.collect([tmp_path / "logs"], 41, 50)
    ses.write_tex(cells, tmp_path / "out")
    assert "\\toprule" in (tmp_path / "out" / "sweep_table.tex").read_text()
    assert "\\textbf{+" in (tmp_path / "out" / "sweep_paired.tex").read_text()  # J - h5 about +5, CI excludes 0
