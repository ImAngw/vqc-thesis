import copy
import json
import math

import numpy as np
import pytest
import torch

import rl_train.a2c as a2c_module
from rl_train.a2c import A2CTrainer
from rl_train.config import A2CConfig
from rl_train.policy import Policy

METRICS = {"return_mean", "return_std", "length_mean", "success_rate", "explored_fraction_mean"}
INDEX = {"update", "step", "episode"}


def _trainer(tmp_path, name="run", **kw):
    base = dict(
        run_name=name, log_dir=str(tmp_path), n_envs=2, n_steps=4, total_updates=4, eval_every=2,
        eval_episodes=2, test_episodes=3, env_kwargs={"max_steps": 6},
    )
    cfg = A2CConfig(**{**base, **kw})
    torch.manual_seed(cfg.seed)
    return A2CTrainer(Policy(), cfg)


def test_smoke_run_produces_all_outputs(tmp_path):
    t = _trainer(tmp_path)
    t.train()
    run = tmp_path / "run"
    assert {p.name for p in run.iterdir()} == {"metrics.json", "best.pt", "last.pt", "config.json"}
    assert json.loads((run / "config.json").read_text())["eval_every"] == 2

    m = json.loads((run / "metrics.json").read_text())
    expected_eval = {f"{g}/{x}" for g in ("eval", "test") for x in INDEX} | {
        f"{g}/{mode}/{k}" for g in ("eval", "test") for mode in ("greedy", "sample") for k in METRICS
    }
    assert expected_eval <= set(m)
    assert {k for k in m if k.startswith("train/")} >= {f"train/{x}" for x in INDEX}
    for group, rows in (("train", 4), ("eval", 2), ("test", 1)):
        lengths = {len(v) for k, v in m.items() if k.startswith(group + "/")}
        assert lengths == {rows}, group
    assert m["eval/update"] == [2.0, 4.0]
    assert m["test/update"] == [4.0]


def test_final_weights_are_evaluated_when_not_a_multiple_of_eval_every(tmp_path):
    t = _trainer(tmp_path, total_updates=3, eval_every=2)
    t.train()
    assert t.logger.metrics["eval/update"] == [2.0, 3.0]


def test_run_shorter_than_eval_every_still_has_best_model(tmp_path):
    t = _trainer(tmp_path, total_updates=1, eval_every=50)
    t.train()
    assert (tmp_path / "run" / "best.pt").exists()
    assert t.logger.metrics["eval/update"] == [1.0]


def test_best_model_only_on_strictly_better_best_metric_and_test_uses_it(tmp_path, monkeypatch):
    t = _trainer(tmp_path, total_updates=4, eval_every=1, eval_episodes=1, test_episodes=1)
    eval_returns = iter([1.0, 3.0, 3.0, 2.0])  # best at eval 2 only; the tie at eval 3 keeps it
    snapshots, calls = [], []

    def fake_evaluate(model, env_fn, episodes, seed0, greedy, device="cpu"):
        if not greedy and seed0 == t.cfg.eval_seed:  # default best_metric: sample/return_mean
            snapshots.append(copy.deepcopy(model.state_dict()))
            value = next(eval_returns)
        else:
            value = 0.0
        calls.append((seed0, greedy, copy.deepcopy(model.state_dict())))
        return {k: value if k == "return_mean" else 0.0 for k in METRICS}

    monkeypatch.setattr(a2c_module, "evaluate", fake_evaluate)
    t.train()

    best = torch.load(tmp_path / "run" / "best.pt", weights_only=True)
    assert all(torch.equal(best[k], snapshots[1][k]) for k in best)  # weights of eval 2
    assert not all(torch.equal(best[k], snapshots[2][k]) for k in best)  # not the tied eval 3
    test_calls = [c for c in calls if c[0] == t.cfg.test_seed]
    assert len(test_calls) == 2 and {c[1] for c in test_calls} == {True, False}
    assert all(all(torch.equal(best[k], c[2][k]) for k in best) for c in test_calls)
    last = torch.load(tmp_path / "run" / "last.pt", weights_only=True)
    assert not all(torch.equal(best[k], last[k]) for k in best)  # last.pt is the final training state


def test_existing_run_directory_is_an_error(tmp_path):
    _trainer(tmp_path)
    with pytest.raises(FileExistsError):
        _trainer(tmp_path)


def test_eval_values_are_finite(tmp_path):
    t = _trainer(tmp_path)
    t.train()
    for k, v in t.logger.metrics.items():
        if k.startswith(("eval/", "test/")):
            assert all(math.isfinite(x) for x in v), k


def test_best_metric_is_configurable_and_validated(tmp_path):
    assert A2CConfig().best_metric == "sample/return_mean"
    A2CConfig(best_metric="greedy/success_rate")
    for bad in ("return_mean", "sample/length_mean", "median/return_mean"):
        with pytest.raises(ValueError):
            A2CConfig(best_metric=bad)


def test_selection_follows_the_configured_metric(tmp_path, monkeypatch):
    """Greedy keeps improving while sample peaks early: best.pt must follow best_metric."""
    t = _trainer(tmp_path, total_updates=3, eval_every=1, eval_episodes=1, test_episodes=1, best_metric="greedy/return_mean")
    greedy_values = iter([1.0, 2.0, 3.0])
    snapshots = []

    def fake_evaluate(model, env_fn, episodes, seed0, greedy, device="cpu"):
        value = 0.0
        if seed0 == t.cfg.eval_seed:
            if greedy:
                snapshots.append(copy.deepcopy(model.state_dict()))
                value = next(greedy_values)
            else:
                value = 10.0 if len(snapshots) == 1 else 0.0  # sample is best at eval 1 only
        return {k: value if k == "return_mean" else 0.0 for k in METRICS}

    monkeypatch.setattr(a2c_module, "evaluate", fake_evaluate)
    t.train()
    best = torch.load(tmp_path / "run" / "best.pt", weights_only=True)
    assert all(torch.equal(best[k], snapshots[2][k]) for k in best)  # eval 3, best greedy


def test_silent_by_default(tmp_path, capsys):
    t = _trainer(tmp_path)
    capsys.readouterr()  # discard the one-off GradLogger banner printed at construction
    t.train()
    assert capsys.readouterr().out == ""


def test_progress_is_printed_when_requested(tmp_path, capsys):
    t = _trainer(tmp_path, print_every=2)
    t.train()
    out = capsys.readouterr().out.splitlines()
    assert [l.split()[2] for l in out if l.startswith("[train]")] == ["2", "4"]  # updates 2 and 4
    assert sum(l.startswith("[eval ]") for l in out) == 2 and sum(l.startswith("[test ]") for l in out) == 1
    assert sum("sample" in l and "ret" in l for l in out) == 3 and sum("greedy" in l and "ret" in l for l in out) == 3
    assert any("(new best)" in l for l in out)  # the first eval is always a new best
    # printing does not change what is logged
    assert len(t.logger.metrics["train/update"]) == 4


def test_head_lr_puts_the_head_in_its_own_param_group(tmp_path):
    t = _trainer(tmp_path, head_lr=7e-3)
    rest, head = t.optimizer.param_groups
    assert head["lr"] == 7e-3 and rest["lr"] == t.cfg.lr
    assert {id(p) for p in head["params"]} == {id(p) for p in t.model.head.parameters()}
    assert len(rest["params"]) + len(head["params"]) == len(list(t.model.parameters()))

    single = _trainer(tmp_path, name="run2")  # head_lr = 0: one group, everything at lr
    assert len(single.optimizer.param_groups) == 1 and single.optimizer.param_groups[0]["lr"] == single.cfg.lr


def test_scale_lr_and_head_momentum_split_the_vqc_head(tmp_path):
    from quantum_heads.vqc_head_i import VQCHeadI

    cfg = A2CConfig(run_name="run", log_dir=str(tmp_path), n_envs=2, n_steps=4, head_lr=5e-4, scale_lr=1e-2,
                    head_momentum=0.9, env_kwargs={"max_steps": 6})
    t = A2CTrainer(Policy(VQCHeadI()), cfg)
    rest, head, scales = t.optimizer.param_groups
    head_named = dict(t.model.head.named_parameters())
    assert {id(p) for p in scales["params"]} == {id(head_named[n]) for n in ("s1", "s2", "s3")}
    assert {id(p) for p in head["params"]} == {id(p) for n, p in head_named.items() if n not in ("s1", "s2", "s3")}
    assert (rest["lr"], head["lr"], scales["lr"]) == (cfg.lr, 5e-4, 1e-2)
    assert rest["momentum"] == 0 and head["momentum"] == scales["momentum"] == 0.9
    assert sum(len(g["params"]) for g in t.optimizer.param_groups) == len(list(t.model.parameters()))

    # momentum alone: the head at lr in its own group; scale_lr on a head without scales is an error
    only_momentum = _trainer(tmp_path, name="run2", head_momentum=0.9)
    rest, head = only_momentum.optimizer.param_groups
    assert head["lr"] == rest["lr"] and head["momentum"] == 0.9 and rest["momentum"] == 0
    with pytest.raises(ValueError, match="scale_lr"):
        _trainer(tmp_path, name="run3", scale_lr=1e-2)


def _same_obs(a, b):
    return all(np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in a)


def test_env_seed_sets_the_envs_and_leaves_the_weights_to_seed(tmp_path):
    default = _trainer(tmp_path, seed=1)
    explicit = _trainer(tmp_path, name="run2", seed=1, env_seed=1)  # env_seed = seed: the default stream
    other = _trainer(tmp_path, name="run3", seed=1, env_seed=2)
    assert _same_obs(default.collector.obs, explicit.collector.obs)
    assert torch.equal(default.collector.generator.get_state(), explicit.collector.generator.get_state())
    assert not _same_obs(default.collector.obs, other.collector.obs)
    for p, q in zip(default.model.parameters(), other.model.parameters()):
        assert torch.equal(p, q)


def test_detach_critic_config_reaches_the_model(tmp_path):
    assert _trainer(tmp_path).model.detach_critic is False
    assert _trainer(tmp_path, name="run2", detach_critic=1).model.detach_critic is True


def _layout(env):
    obstacles = tuple(sorted((int(x), int(y)) for y, x in zip(*np.nonzero(env._obstacles))))
    return obstacles, env._agent, env._goal


def test_layout_pool_reaches_the_training_envs_only(tmp_path):
    trainer = _trainer(tmp_path, layout_pool=3, env_kwargs={"max_steps": 6})
    train_envs = trainer.collector.env.envs
    pool = {(tuple(sorted(o)), a, g) for o, a, g in train_envs[0]._layout_pool}
    assert len(pool) == 3
    assert all(_layout(env) in pool for env in train_envs)
    assert "layout_pool" not in trainer.cfg.env_kwargs  # eval / test envs: fresh layouts
    trainer.train()  # resets inside the rollouts keep drawing from the pool (the agent has moved: compare the rest)
    assert all((_layout(env)[0], _layout(env)[2]) in {(o, g) for o, _, g in pool} for env in train_envs)


def test_layout_pool_seed_defaults_to_the_env_seed():
    def pool_seed(**kw):
        return A2CConfig(layout_pool=5, **kw).train_env_kwargs()["layout_pool_seed"]

    assert pool_seed(seed=4) == 4
    assert pool_seed(seed=4, env_seed=2) == 2
    assert pool_seed(seed=4, env_seed=2, layout_pool_seed=7) == 7
    assert A2CConfig(env_kwargs={"window": 5}).train_env_kwargs() == {"window": 5}  # no pool: env_kwargs as is


def test_config_validates_perception_pool_and_env_kwargs():
    assert A2CConfig(env_kwargs='{"window": 5, "n_obstacles": 30}').env_kwargs == {"window": 5, "n_obstacles": 30}
    for kw in ({"perception": "mlp"}, {"layout_pool": -1}, {"env_kwargs": {"layout_pool": 3}}, {"env_kwargs": "[5]"}):
        with pytest.raises(ValueError):
            A2CConfig(**kw)


def test_cnn_perception_with_window_5_runs_the_env_v8_setup(tmp_path):
    from encoder import Encoder
    from rl_train.policy import build_policy, encoder_kwargs

    cfg = A2CConfig(run_name="cnn", log_dir=str(tmp_path), n_envs=2, n_steps=4, total_updates=4, eval_every=2,
                    eval_episodes=2, test_episodes=3, perception="cnn", layout_pool=2,
                    env_kwargs={"max_steps": 6, "window": 5, "n_obstacles": 30})
    torch.manual_seed(cfg.seed)
    policy = build_policy(lambda: torch.nn.Linear(12, 8), **encoder_kwargs(cfg))
    reference = Encoder(window=5, perception="cnn")  # the XS encoder of the 5x5-window runs (section 11)
    assert {k: v.shape for k, v in policy.encoder.state_dict().items()} == \
           {k: v.shape for k, v in reference.state_dict().items()}
    trainer = A2CTrainer(policy, cfg)
    assert trainer.collector.obs["perception"].shape[1:] == (1, 5, 5)
    trainer.train()


def test_pool_metrics_are_logged_only_with_a_layout_pool(tmp_path):
    with_pool = _trainer(tmp_path, layout_pool=2)
    with_pool.train()
    m = with_pool.logger.metrics
    pool_keys = {f"{g}/pool/{mode}/{k}" for g in ("eval", "test") for mode in ("greedy", "sample") for k in METRICS}
    assert pool_keys <= set(m)
    assert all(len(m[k]) == len(m["eval/update"]) for k in pool_keys if k.startswith("eval/"))
    without = _trainer(tmp_path, name="run2")
    without.train()
    assert not any("/pool/" in k for k in without.logger.metrics)


@pytest.mark.parametrize("pool_seed", [0, 1, 2])
def test_eval_layouts_are_never_in_the_training_pool(pool_seed):
    """The pilot setup: K = 200 training layouts vs the 100 eval layouts (seeds 10000..10099)."""
    from rl_env.env import CustomEnv

    env_kwargs = {"window": 5, "n_obstacles": 30}
    cfg = A2CConfig(layout_pool=200, env_seed=pool_seed, env_kwargs=env_kwargs)
    pool = {(tuple(sorted(o)), a, g) for o, a, g in CustomEnv(**cfg.train_env_kwargs())._layout_pool}
    assert len(pool) == 200
    fresh = CustomEnv(**cfg.env_kwargs)
    for i in range(cfg.eval_episodes):
        fresh.reset(seed=cfg.eval_seed + i)
        assert _layout(fresh) not in pool


def test_fourier_head_runs_in_the_pilot_setup(tmp_path):
    from rl_train.policy import build_policy, encoder_kwargs
    from train import HEADS

    cfg = A2CConfig(run_name="four", log_dir=str(tmp_path), n_envs=2, n_steps=4, total_updates=2, eval_every=2,
                    eval_episodes=2, test_episodes=2, head="fourier4", perception="cnn", layout_pool=3,
                    target_entropy=1.425, env_kwargs={"max_steps": 6, "window": 5, "n_obstacles": 30})
    torch.manual_seed(cfg.seed)
    trainer = A2CTrainer(build_policy(HEADS["fourier4"], **encoder_kwargs(cfg)), cfg)
    trainer.train()
    assert all(math.isfinite(v) for v in trainer.logger.metrics["eval/pool/sample/return_mean"])
