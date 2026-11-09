import importlib.util
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("sweep_entropy", ROOT / "scripts" / "sweep_entropy.py")
sw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sw)
cc = sw.cell_command
BASE = {k: v for k, v in json.loads((ROOT / "configs" / "runs.json").read_text()).items() if not k.startswith("ent-sweep-")}


def flags(args):
    return dict(zip(args[::2], args[1::2]))


def test_eighty_runs_all_on_the_mac():
    recs = sw.records()
    assert len(recs) == 80
    assert {r["machine"] for r in recs.values()} == {"mac"}
    assert {r["seed"] for r in recs.values()} == set(range(10))


def test_targets_are_valid_for_train_py_and_distinct_from_the_matched_control():
    assert len(set(sw.TARGETS)) == 4 and all(0 < t < math.log(8) for t in sw.TARGETS)
    assert 1.425 not in sw.TARGETS


def test_sweep_run_differs_from_the_matched_control_only_in_the_target():
    """Same seeds, pool, environment and schedule as ent-h5-k25 / ent-h128-k25: only --target-entropy and the name change."""
    runs = {**BASE, **sw.records()}
    for head, ctrl in (("h5", "ent-h5-k25"), ("h128", "ent-h128-k25")):
        for seed in (0, 7):
            new = flags(cc.command(sw.run_name(head, 0.8, seed), runs))
            old = flags(cc.command(f"{ctrl}-s{seed}-{sw.SUFFIX}", runs))
            diff = {k for k in set(new) | set(old) if new.get(k) != old.get(k)}
            assert diff == {"--run-name", "--target-entropy"}
            assert new["--target-entropy"] == "0.8"


def test_merge_adds_80_runs_once_and_keeps_the_others(tmp_path):
    path = tmp_path / "runs.json"
    path.write_text(json.dumps(BASE, indent=1, sort_keys=True))
    assert sw.merge(path) == 80
    assert sw.merge(path) == 0
    merged = json.loads(path.read_text())
    assert len(merged) == len(BASE) + 80
    assert all(merged[k] == v for k, v in BASE.items())


def test_is_done_needs_all_the_rounds(tmp_path):
    name = sw.run_name("h5", 0.4, 0)
    assert not sw.is_done(tmp_path, name)
    log = tmp_path / name / "node_0.log"
    log.parent.mkdir()
    log.write_text("  local 150\n  TOTAL 150\n" * 49)
    assert not sw.is_done(tmp_path, name)
    log.write_text("  local 150\n  TOTAL 150\n" * 50)
    assert sw.is_done(tmp_path, name)


def test_interrupted_run_dir_is_renamed_not_deleted(tmp_path):
    name = sw.run_name("h5", 0.4, 6)
    assert sw.set_aside_partial(tmp_path, name) is None
    (tmp_path / name).mkdir()
    (tmp_path / name / "node_0.log").write_text("x")
    first = sw.set_aside_partial(tmp_path, name)
    assert first.name == name + ".partial" and (first / "node_0.log").read_text() == "x"
    (tmp_path / name).mkdir()
    assert sw.set_aside_partial(tmp_path, name).name == name + ".partial2"
    assert not (tmp_path / name).exists()


def test_sweep_runs_are_not_cells_of_the_k_summaries():
    spec2 = importlib.util.spec_from_file_location("summarize_gen_pilot", ROOT / "scripts" / "summarize_gen_pilot.py")
    sg = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(sg)
    assert sg.GEN_NAME.match(sw.run_name("h5", 0.4, 0)) is None
    assert sg.GEN_NAME.match("ent-h5-k25-s0-7.5k-cpu-outer")["head"] == "h5"
