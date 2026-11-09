import pytest
import torch
from torch import nn

from rl_train.a2c import A2CTrainer
from rl_train.config import A2CConfig
from rl_train.outer import OuterStep, pack, unpack
from rl_train.policy import Policy


def test_pack_unpack_roundtrip():
    net = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    params = list(net.parameters())
    flat = pack(params)
    assert flat.numel() == sum(p.numel() for p in params)
    for p in params:
        p.data.zero_()
    unpack(flat, params)
    assert torch.equal(pack(params), flat)


def test_outer_step_matches_hand_computed_nesterov():
    lr, mom = 0.7, 0.9
    p = nn.Parameter(torch.tensor([1.0, -2.0]))
    outer = OuterStep([p], lr, mom)
    g0 = torch.tensor([1.0, -2.0])  # starting point of round 1
    p.data.add_(torch.tensor([-0.5, 0.25]))  # inner optimizer moved the weights
    outer.step()
    delta1 = torch.tensor([0.5, -0.25])  # global - local = [1, -2] - [0.5, -1.75]
    buf = delta1.clone()  # torch SGD: first step buffer = grad
    expected = g0 - lr * (delta1 + mom * buf)  # Nesterov: grad + momentum * buffer
    assert torch.allclose(p.data, expected)

    g1 = expected.clone()  # round 2 starts from the new global weights
    p.data.add_(torch.tensor([0.1, 0.1]))
    local = p.data.clone()
    outer.step()
    delta2 = g1 - local
    buf = mom * buf + delta2
    expected = g1 - lr * (delta2 + mom * buf)
    assert torch.allclose(p.data, expected)


def test_outer_lr_one_no_momentum_keeps_local_weights():
    p = nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
    outer = OuterStep([p], 1.0, 0.0)
    p.data.add_(1.0)
    local = p.data.clone()
    outer.step()
    assert torch.allclose(p.data, local)


def test_config_validation():
    with pytest.raises(ValueError):
        A2CConfig(outer_lr=0.7)  # inner_steps missing
    with pytest.raises(ValueError):
        A2CConfig(outer_lr=0.7, inner_steps=150, total_updates=1000)  # not a multiple
    A2CConfig(outer_lr=0.7, outer_momentum=0.9, inner_steps=150, total_updates=1500)


def _run(tmp_path, name, **kw):
    cfg = A2CConfig(run_name=name, log_dir=str(tmp_path), n_envs=4, n_steps=8, total_updates=6, eval_every=3,
                    eval_episodes=1, test_episodes=1, env_kwargs={"max_steps": 10}, **kw)
    torch.manual_seed(cfg.seed)
    trainer = A2CTrainer(Policy(), cfg)
    trainer.train()
    return torch.cat([p.data.reshape(-1) for p in trainer.model.parameters()])


def test_outer_step_changes_training_only_when_on(tmp_path):
    """The default (outer_lr = 0) is deterministic and untouched; an active outer step changes the weights."""
    base = _run(tmp_path, "base")
    again = _run(tmp_path, "again")
    assert torch.equal(base, again)  # the run is deterministic on this machine
    outer = _run(tmp_path, "outer", outer_lr=0.7, outer_momentum=0.9, inner_steps=3)
    assert not torch.equal(base, outer)
