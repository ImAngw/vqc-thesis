"""Tests of scripts/summarize_gen_pilot.py: the 95% CI of paired differences and the cell of a rerun (-r) run."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import summarize_gen_pilot as sgp  # noqa: E402

HEADER = "  node    steps  steps/s       loss  sync/round  state       sample/return_mean  pool/sample/return_mean"


def write_log(logs: Path, name: str, test: float, pool: float, rounds: int = 50) -> None:
    lines = []
    for r in range(1, rounds + 1):
        lines += [f"round {r}/50", HEADER, f"  TOTAL     150      6.9     0.6243       21.6s  {test}  {pool}"]
    (logs / name).mkdir(parents=True)
    (logs / name / "node_0.log").write_text("\n".join(lines) + "\n")


def run(logs: Path, *args: str) -> str:
    cmd = [sys.executable, str(ROOT / "scripts" / "summarize_gen_pilot.py"), "--logs", str(logs), *args]
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


@pytest.mark.parametrize("n", [2, 3, 8, 10, 31])
def test_ci95_matches_scipy(n):
    stats = pytest.importorskip("scipy.stats")
    lo, hi = sgp.ci95(1.0, 0.5, n)
    half = stats.t.ppf(0.975, n - 1) * 0.5
    assert lo == pytest.approx(1.0 - half, abs=1e-3)
    assert hi == pytest.approx(1.0 + half, abs=1e-3)


def test_ci95_undefined_cases():
    assert all(map(lambda x: x != x, sgp.ci95(1.0, 0.5, 1)))  # one value: no CI
    assert all(map(lambda x: x != x, sgp.ci95(1.0, 0.5, 32)))  # beyond the table
    assert all(map(lambda x: x != x, sgp.ci95(1.0, float("nan"), 10)))


def test_paired_line_has_ci(tmp_path):
    # diffs a - b = 1, 2, ..., 10: mean 5.5, SE sqrt(var / 10) = 0.957, half-width 2.262 * 0.957 = 2.166
    for s in range(10):
        write_log(tmp_path, f"gen-a-k5-s{s}-x", test=float(s + 1), pool=0.0)
        write_log(tmp_path, f"gen-b-k5-s{s}-x", test=0.0, pool=0.0)
    out = run(tmp_path, "gen-*", "--pairs", "a:b")
    line = next(l for l in out.splitlines() if "a - b" in l)
    assert "(n=10)" in line
    assert "CI95 [ 3.33,  7.67]" in line


def test_rerun_replaces_stopped_run_in_cell(tmp_path):
    # a stopped run (8 rounds, window incomplete) and its full rerun with the -r suffix: the cell takes the rerun
    write_log(tmp_path, "gen-a-k5-s6-x", test=9.0, pool=9.0, rounds=8)
    write_log(tmp_path, "gen-a-k5-s6-x-r", test=2.0, pool=3.0)
    out = run(tmp_path, "gen-*", "--pairs", "a:a")
    cell = next(l for l in out.splitlines() if l.split()[:3] == ["a", "5", "1"])
    assert "2.00" in cell and "3.00" in cell
