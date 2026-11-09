import pytest
import torch
from torch import nn

from rl_train.config import A2CConfig
from rl_train.envs import make_env
from rl_train.evaluate import evaluate
from rl_train.policy import Policy

KEYS = {"return_mean", "return_std", "length_mean", "success_rate", "explored_fraction_mean"}


class Uniform(nn.Module):
    """All-zero logits: argmax is always action 0 (north), sampling is uniform."""

    def forward(self, obs):
        b = obs["navigator"].shape[0]
        return torch.zeros(b, 8), torch.zeros(b)


def env_fn():
    return make_env(max_steps=20)


def test_metric_keys_and_ranges():
    out = evaluate(Policy(), env_fn, episodes=3, seed0=0, greedy=True)
    assert set(out) == KEYS
    assert out["length_mean"] <= 20 and 0 <= out["success_rate"] <= 1 and 0 < out["explored_fraction_mean"] <= 1


@pytest.mark.parametrize("greedy", [True, False])
def test_fixed_seeds_give_identical_results(greedy):
    torch.manual_seed(0)
    model = Policy()
    a = evaluate(model, env_fn, episodes=4, seed0=100, greedy=greedy)
    b = evaluate(model, env_fn, episodes=4, seed0=100, greedy=greedy)
    assert a == b


def test_greedy_is_argmax_and_sample_is_not():
    greedy = evaluate(Uniform(), env_fn, episodes=5, seed0=7, greedy=True)
    sample = evaluate(Uniform(), env_fn, episodes=5, seed0=7, greedy=False)

    # manual replay of "always action 0" on the same seeds
    env = env_fn()
    expected = []
    for i in range(5):
        env.reset(seed=7 + i)
        total, done = 0.0, False
        while not done:
            _, r, term, trunc, _ = env.step(0)
            total += r
            done = term or trunc
        expected.append(total)
    assert greedy["return_mean"] == pytest.approx(sum(expected) / 5, abs=1e-6)
    assert sample != greedy


def test_greedy_matches_sequential_one_episode_at_a_time():
    """The batched rollout gives the same greedy episodes as playing each seed alone with batch 1."""
    torch.manual_seed(0)
    model = Policy().eval()
    env = env_fn()
    returns, lengths = [], []
    with torch.no_grad():
        for i in range(6):
            obs, _ = env.reset(seed=50 + i)
            total, n, done = 0.0, 0, False
            while not done:
                logits, _ = model({k: torch.as_tensor(v[None], dtype=torch.float32) for k, v in obs.items()})
                obs, r, term, trunc, _ = env.step(int(logits.argmax(dim=-1).item()))
                total, n, done = total + r, n + 1, term or trunc
            returns.append(total)
            lengths.append(n)
    out = evaluate(model, env_fn, episodes=6, seed0=50, greedy=True)
    assert out["return_mean"] == pytest.approx(sum(returns) / 6, abs=1e-6)
    assert out["length_mean"] == pytest.approx(sum(lengths) / 6)


def test_model_mode_is_restored_and_no_grad_left_behind():
    model = Policy()
    model.train()
    evaluate(model, env_fn, episodes=1, seed0=0, greedy=True)
    assert model.training
    model.eval()
    evaluate(model, env_fn, episodes=1, seed0=0, greedy=True)
    assert not model.training
    assert all(p.grad is None for p in model.parameters())


def test_overlapping_eval_and_test_seeds_are_rejected():
    A2CConfig()  # defaults are disjoint
    with pytest.raises(ValueError):
        A2CConfig(eval_seed=0, eval_episodes=20, test_seed=10, test_episodes=100)
