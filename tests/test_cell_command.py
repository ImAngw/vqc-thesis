import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("cell_command", ROOT / "scripts" / "cell_command.py")
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)
RUNS = cc.load_runs()


def test_h5_k25_seed0_is_the_command_checked_against_the_original_log():
    args = cc.command("gen-h5-k25-s0-7.5k-cpu-outer", RUNS)
    flags = dict(zip(args[::2], args[1::2]))
    assert flags["--env-seed"] == "818069386" and flags["--layout-pool-seed"] == "0" and flags["--seed"] == "0"
    assert flags["--head"] == "mlp5" and flags["--layout-pool"] == "25" and flags["--perception"] == "cnn"
    assert flags["--outer-dtype"] == "bf16" and flags["--total-updates"] == "7500"


def test_old_runs_get_the_v8_environment_new_ones_keep_their_own():
    old = dict(zip(*[iter(cc.command("nav-id-hlr2e-3-detc-s0-7.5k-cpu-outer", RUNS))] * 2))
    assert old["--perception"] == "cnn" and old["--env-kwargs"] == '{"window": 5, "n_obstacles": 30}'
    new = cc.command("gen-four-k25-s1-7.5k-cpu-outer", RUNS)
    assert new.count("--perception") == 1 and new.count("--env-kwargs") == 1


def test_run_without_env_seed_uses_its_seed():
    args = cc.command("h128-rms-detc-s3-7.5k-cpu-outer", RUNS)
    assert args[args.index("--env-seed") + 1] == str(cc.drawn_env_seed(3))


def test_warm_started_runs_are_refused():
    name = next(n for n in RUNS if n.startswith("ws-") and "init_from" in RUNS[n]["params"])
    with pytest.raises(ValueError):
        cc.command(name, RUNS)


def test_every_supported_run_is_accepted_by_train_py(monkeypatch):
    import train
    checked = 0
    for name in RUNS:
        try:
            args = cc.command(name, RUNS)
        except ValueError:
            continue
        monkeypatch.setattr(sys, "argv", ["train.py"] + args)
        cfg = train.parse_args()
        assert cfg.run_name == name and cfg.outer_lr == 0.7 and cfg.total_updates == 7500
        checked += 1
    assert checked > 250
