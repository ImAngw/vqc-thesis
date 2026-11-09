import importlib.util
import math
from pathlib import Path

import torch

from rl_train.a2c import A2CTrainer
from rl_train.config import A2CConfig
from rl_train.policy import Policy

SUMMARIZE = Path(__file__).resolve().parents[2] / "scripts" / "summarize_gen_pilot.py"


def _read_rounds(path):
    spec = importlib.util.spec_from_file_location("summarize_gen_pilot", SUMMARIZE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.read_rounds(path)


def _train(tmp_path, **kw):
    cfg = A2CConfig(run_name="r", log_dir=str(tmp_path), n_envs=4, n_steps=8, total_updates=6, eval_episodes=1,
                    test_episodes=1, env_kwargs={"max_steps": 10}, **kw)
    torch.manual_seed(cfg.seed)
    trainer = A2CTrainer(Policy(), cfg)
    trainer.train()
    return trainer


def test_round_log_is_readable_by_the_summary_scripts(tmp_path):
    trainer = _train(tmp_path, inner_steps=3, eval_every=3, outer_lr=0.7, outer_momentum=0.9, layout_pool=2, env_seed=0)
    rounds = _read_rounds(trainer.logger.run_dir / "node_0.log")
    assert sorted(rounds) == [1, 2]
    m = trainer.logger.metrics
    for rnd in (1, 2):
        row = rounds[rnd]
        window = slice((rnd - 1) * 3, rnd * 3)
        assert math.isclose(row["entropy"], sum(m["train/entropy"][window]) / 3, rel_tol=1e-3)
        assert math.isclose(row["sample/return_mean"], m["eval/sample/return_mean"][rnd - 1], rel_tol=1e-3, abs_tol=1e-3)
        assert "pool/sample/return_mean" in row


def test_round_log_marks_rounds_without_eval(tmp_path):
    trainer = _train(tmp_path, inner_steps=3, eval_every=6)  # evaluated at update 6 only
    rounds = _read_rounds(trainer.logger.run_dir / "node_0.log")
    assert "sample/return_mean" not in rounds[1] and "sample/return_mean" in rounds[2]


def test_no_round_log_without_inner_steps(tmp_path):
    trainer = _train(tmp_path, eval_every=3)
    assert not (trainer.logger.run_dir / "node_0.log").exists()
