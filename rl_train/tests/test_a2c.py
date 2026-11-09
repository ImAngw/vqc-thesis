import math

import torch

from rl_train.a2c import A2CTrainer
from rl_train.config import A2CConfig
from rl_train.policy import Policy

TRAIN_KEYS = {
    "train/update", "train/step", "train/episode", "train/policy_loss", "train/value_loss",
    "train/entropy", "train/return_mean", "train/length_mean", "train/explored_fraction_mean",
    "train/adv_std", "train/explained_var", "train/grad_norm", "train/entropy_coef",
}


def _cfg(tmp_path, name="run", **kw):
    base = dict(run_name=name, log_dir=str(tmp_path), n_envs=4, n_steps=8, total_updates=3, eval_every=3,
                eval_episodes=1, test_episodes=1, env_kwargs={"max_steps": 10})
    return A2CConfig(**{**base, **kw})


def _trainer(tmp_path, **kw):
    cfg = _cfg(tmp_path, **kw)
    torch.manual_seed(cfg.seed)
    return A2CTrainer(Policy(), cfg)


def test_short_run_logs_expected_rows(tmp_path):
    t = _trainer(tmp_path)
    t.train()
    m = {k: v for k, v in t.logger.metrics.items() if k.startswith("train/")}
    assert set(m) == TRAIN_KEYS
    assert all(len(v) == 3 for v in m.values())
    assert m["train/update"] == [1.0, 2.0, 3.0]
    assert m["train/step"] == [32.0, 64.0, 96.0]  # n_envs * n_steps per update
    for key in ("policy_loss", "value_loss", "entropy"):
        assert all(math.isfinite(x) for x in m[f"train/{key}"])
    assert all(0 < e <= math.log(8) + 1e-4 for e in m["train/entropy"])
    assert m["train/episode"][-1] > 0  # max_steps=10 and 96 steps per env-batch: episodes finish
    assert (t.logger.run_dir / "metrics.json").exists()


def test_nan_when_no_episode_finishes(tmp_path):
    t = _trainer(tmp_path, total_updates=1, n_steps=4, env_kwargs={"max_steps": 300})
    t.train()
    m = t.logger.metrics
    assert math.isnan(m["train/return_mean"][0]) and math.isnan(m["train/length_mean"][0])
    assert m["train/episode"] == [0.0]


def test_parameters_change_after_an_update(tmp_path):
    t = _trainer(tmp_path, total_updates=1)
    before = [p.detach().clone() for p in t.model.parameters()]
    t.train()
    assert any(not torch.equal(b, p) for b, p in zip(before, t.model.parameters()))


def test_same_seed_same_train_metrics(tmp_path):
    a = _trainer(tmp_path, name="a", seed=3)
    b = _trainer(tmp_path, name="b", seed=3)
    a.train()
    b.train()
    for key in TRAIN_KEYS:
        x, y = torch.tensor(a.logger.metrics[key]), torch.tensor(b.logger.metrics[key])
        assert torch.allclose(x, y, equal_nan=True), key


def test_different_seed_different_metrics(tmp_path):
    a = _trainer(tmp_path, name="a", seed=1)
    b = _trainer(tmp_path, name="b", seed=2)
    a.train()
    b.train()
    assert a.logger.metrics["train/policy_loss"] != b.logger.metrics["train/policy_loss"]


def test_entropy_coef_anneals_linearly_then_stays_constant(tmp_path):
    t = _trainer(tmp_path, entropy_start=0.02, entropy_end=0.001, entropy_anneal_updates=11)
    assert t.entropy_coef() == 0.02
    t.n_updates = 5
    assert abs(t.entropy_coef() - 0.0105) < 1e-9  # halfway: 5 / (11 - 1)
    for n in (10, 11, 1000):
        t.n_updates = n
        assert abs(t.entropy_coef() - 0.001) < 1e-12


def test_defaults_match_qml_thesis():
    cfg = A2CConfig()
    assert cfg.norm_advantage == 1
    assert (cfg.entropy_start, cfg.entropy_end, cfg.entropy_anneal_updates) == (0.02, 0.001, 2500)


def test_adaptive_entropy_coef_moves_toward_the_target(tmp_path):
    t = _trainer(tmp_path, target_entropy=1.5, entropy_lr=0.1, entropy_start=0.02)
    assert math.isclose(t.entropy_coef(), 0.02)
    t.n_updates = 1000  # the schedule no longer applies
    assert math.isclose(t.entropy_coef(), 0.02)
    rollout = t.collector.collect()
    coef_before = t.entropy_coef()
    losses = t.update(rollout)
    assert losses["entropy_coef"] == coef_before  # the coefficient used in this update is logged
    expected = coef_before * math.exp(0.1 * (1.5 - losses["entropy"]))  # entropy below / above target -> up / down
    assert math.isclose(t.entropy_coef(), expected, rel_tol=1e-9)
    assert (t.entropy_coef() > coef_before) == (losses["entropy"] < 1.5)


def test_adaptive_entropy_coef_is_clamped(tmp_path):
    # the untrained policy is near uniform (entropy ~2.06 < ln 8 ~ 2.079): a target just below ln 8 is above it,
    # and a huge step pushes the coefficient to the upper bound
    t = _trainer(tmp_path, target_entropy=2.0794, entropy_lr=1e5)
    losses = t.update(t.collector.collect())
    assert losses["entropy"] < 2.0794
    assert math.isclose(t.entropy_coef(), 1.0)
    t = _trainer(tmp_path, name="run2", target_entropy=0.01, entropy_lr=100.0)  # far above target -> lower bound
    t.update(t.collector.collect())
    assert math.isclose(t.entropy_coef(), 1e-4)


def test_adaptive_entropy_run_logs_a_finite_coefficient(tmp_path):
    t = _trainer(tmp_path, target_entropy=1.5)
    t.train()
    coefs = t.logger.metrics["train/entropy_coef"]
    assert len(coefs) == 3 and all(math.isfinite(c) and 1e-4 <= c <= 1.0 for c in coefs)
    assert len(set(coefs)) > 1  # it adapts


def test_target_entropy_off_keeps_the_schedule(tmp_path):
    t = _trainer(tmp_path, entropy_start=0.02, entropy_end=0.001, entropy_anneal_updates=11)
    t.update(t.collector.collect())
    t.n_updates = 5
    assert abs(t.entropy_coef() - 0.0105) < 1e-9  # an update does not touch the scheduled coefficient


def test_entropy_config_is_validated(tmp_path):
    import pytest

    for kw in ({"target_entropy": -0.1}, {"target_entropy": math.log(8)}, {"entropy_lr": 0.0},
               {"target_entropy": 1.0, "entropy_start": 0.0}):
        with pytest.raises(ValueError):
            _cfg(tmp_path, **kw)
