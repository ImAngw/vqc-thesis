import json
import math

import pytest

from rl_train.logger import MetricsLogger


def test_rows_grow_equally_and_roundtrip_with_nan(tmp_path):
    log = MetricsLogger(tmp_path, "run")
    log.log("train", {"update": 1, "loss": 0.5, "ret": float("nan")})
    log.log("train", {"update": 2, "loss": 0.25, "ret": 1.0})
    log.log("eval", {"update": 2, "ret": 3.0})
    log.flush()

    data = json.loads(log.path.read_text())
    assert set(data) == {"train/update", "train/loss", "train/ret", "eval/update", "eval/ret"}
    assert len({len(v) for k, v in data.items() if k.startswith("train/")}) == 1
    assert data["train/update"] == [1.0, 2.0] and data["eval/ret"] == [3.0]
    assert math.isnan(data["train/ret"][0]) and data["train/ret"][1] == 1.0


def test_inconsistent_keys_in_a_group_are_rejected(tmp_path):
    log = MetricsLogger(tmp_path, "run")
    log.log("train", {"a": 1, "b": 2})
    with pytest.raises(ValueError):
        log.log("train", {"a": 1})
    assert log.metrics["train/a"] == [1.0]  # the rejected row left no trace


def test_flush_is_atomic_and_leaves_no_temp_file(tmp_path):
    log = MetricsLogger(tmp_path, "run")
    log.log("train", {"a": 1})
    log.flush()
    log.log("train", {"a": 2})
    log.flush()
    assert json.loads(log.path.read_text()) == {"train/a": [1.0, 2.0]}
    assert [p.name for p in log.run_dir.iterdir()] == ["metrics.json"]


def test_existing_run_dir_is_an_error(tmp_path):
    MetricsLogger(tmp_path, "run")
    with pytest.raises(FileExistsError):
        MetricsLogger(tmp_path, "run")
