import pytest
import torch

from rl_train.policy import Policy, build_policy
from train import HEADS

HEAD_NAMES = ("mlp64", "mlp128", "vqc-optim")


def _build(name: str, seed: int = 3) -> Policy:
    torch.manual_seed(seed)
    return build_policy(HEADS[name])


def _same(a: torch.nn.Module, b: torch.nn.Module) -> bool:
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def test_encoder_and_critic_start_identical_whatever_the_head():
    ref = _build(HEAD_NAMES[0])
    for name in HEAD_NAMES[1:]:
        other = _build(name)
        assert _same(ref.encoder, other.encoder) and _same(ref.critic, other.critic), name


@pytest.mark.parametrize("name", HEAD_NAMES)
def test_build_policy_is_deterministic_and_follows_the_seed(name):
    assert _same(_build(name), _build(name))
    assert not _same(_build(name, seed=3).encoder, _build(name, seed=4).encoder)


def test_global_rng_ends_in_the_same_state_whatever_the_head():
    states = []
    for name in HEAD_NAMES:
        _build(name)
        states.append(torch.get_rng_state())
    assert all(torch.equal(states[0], s) for s in states[1:])



def test_detach_critic_keeps_the_value_loss_out_of_the_encoder():
    obs = {"navigator": torch.randn(6, 17), "perception": torch.randn(6, 1, 3, 3), "memory": torch.randn(6, 4, 10)}
    policy = _build("mlp64")
    logits, value = policy(obs)
    value.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.encoder.parameters())

    policy.zero_grad(set_to_none=True)
    policy.detach_critic = True
    logits_d, value_d = policy(obs)
    assert torch.equal(logits, logits_d) and torch.equal(value, value_d)  # same outputs, only the gradient changes
    value_d.sum().backward()
    assert all(p.grad is None for p in policy.encoder.parameters())
    assert all(p.grad is not None for p in policy.critic.parameters())
    logits_d.sum().backward()  # the policy still trains the encoder
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.encoder.parameters())
