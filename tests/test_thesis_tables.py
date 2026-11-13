import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("thesis_tables", ROOT / "scripts" / "thesis_tables.py")
tt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tt)
LOGS = ROOT / "data" / "logs"


@pytest.fixture(scope="module")
def cells():
    return tt.collect(LOGS)


def test_table_4_values_of_the_report(cells):
    """J, K=5 -1.13 +- 0.37 (47%); K=25 4.05 +- 0.47 (92%); h128-ent K=25 4.29 +- 0.23 (93%); J K=inf 4.02 +- 0.28 (94%)."""
    text = tt.table_returns(cells)
    for expected in ("$-1.13\\pm0.37$ (47\\%)", "$4.05\\pm0.47$ (92\\%)", "$4.29\\pm0.23$ (93\\%)", "$4.02\\pm0.28$ (94\\%)"):
        assert expected in text, expected


def test_table_5_values_of_the_report(cells):
    """J - h5-ent at K=25: +2.13 [0.82, 3.44], excludes 0 (bold); J - Fourier at K=25: +0.37 [-0.82, 1.56], includes 0."""
    assert tt.paired(cells, "h5-ent", "25") == "\\textbf{\\boldmath $+2.13$ $[0.82,\\,3.44]$}"
    assert tt.paired(cells, "four", "25") == "$+0.37$ $[-0.82,\\,1.56]$"


def test_table_6_values_of_the_report(cells):
    """Gap of J at K=5: 6.63 +- 0.39; h128 at K=25: 4.35 +- 0.47."""
    text = tt.table_gap(cells)
    assert "$6.63\\pm0.39$" in text and "$4.35\\pm0.47$" in text


def test_tables_11_and_12_of_the_report():
    noise = ROOT / "quantum_heads" / "analysis" / "runs" / "noise"
    rows = tt.summarize_noise.load(str(noise), "gen-nav-id-k25-s*-7.5k-cpu-outer.json")
    t11, t12 = tt.table_noise(rows)
    assert tt.noise_reference(rows) == "$4.18\\pm0.44$"  # no-noise return, reported in the caption
    assert "NR" not in t11 and "$4.18" not in t11
    for expected in ("$4.09\\pm0.38$", "$3.80\\pm0.41$"):
        assert expected in t11, expected
    assert "\\textbf{\\boldmath $-0.38$ $[-0.62,\\,-0.14]$}" in t12 and "$-0.10$ $[-0.28,\\,0.09]$" in t12


def test_table_4_bold_marks_column_maxima_and_j_at_k25(cells):
    """Bold: the highest mean of each K column (K=5 J, K=25 h128-ent, K=200 and inf h128) and J at K=25."""
    rows = {line.split(" & ")[0]: line for line in tt.table_returns(cells).splitlines() if " & " in line}
    bold_cells = {name: [c.strip().startswith("\\textbf") for c in line.rstrip("\\ ").split(" & ")[1:]] for name, line in rows.items()}
    assert bold_cells["J"][:2] == [True, True] and not any(bold_cells["J"][2:])
    assert bold_cells["h128-ent"][1] is True
    assert bold_cells["h128"][2:] == [True, True]
    assert not any(bold_cells["Fourier"]) and not any(bold_cells["h5"]) and not any(bold_cells["h5-ent"])
