import torch

from rl_train.returns import n_step_returns

GAMMA = 0.5


def _run(rewards, terminated, truncated, final_values, last_value):
    t = lambda x, dtype=torch.float32: torch.tensor(x, dtype=dtype).reshape(len(x), -1)
    return n_step_returns(
        t(rewards),
        t(terminated, torch.bool),
        t(truncated, torch.bool),
        t(final_values),
        torch.tensor([last_value]),
        GAMMA,
    ).flatten()


def test_no_episode_end_bootstraps_last_value():
    # R2 = 1 + .5*8 = 5, R1 = 1 + .5*5 = 3.5, R0 = 1 + .5*3.5 = 2.75
    out = _run([1, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0], 8.0)
    assert torch.allclose(out, torch.tensor([2.75, 3.5, 5.0]))


def test_termination_cuts_with_zero_bootstrap():
    # step 1 terminates: R1 = 2; R0 = 1 + .5*2 = 2; R2 starts a new episode: 3 + .5*4 = 5
    out = _run([1, 2, 3], [0, 1, 0], [0, 0, 0], [99, 99, 99], 4.0)
    assert torch.allclose(out, torch.tensor([2.0, 2.0, 5.0]))


def test_truncation_bootstraps_final_value():
    # step 1 truncated with V(final)=6: R1 = 2 + .5*6 = 5; R0 = 1 + .5*5 = 3.5
    out = _run([1, 2, 3], [0, 0, 0], [0, 1, 0], [99, 6, 99], 4.0)
    assert torch.allclose(out, torch.tensor([3.5, 5.0, 5.0]))


def test_mixed_termination_and_truncation():
    # t0 truncated (V=10): 1 + 5 = 6; t1 terminated: 2; t2 free: 3 + .5*4 = 5
    out = _run([1, 2, 3], [0, 1, 0], [1, 0, 0], [10, 99, 99], 4.0)
    assert torch.allclose(out, torch.tensor([6.0, 2.0, 5.0]))


def test_terminated_wins_over_truncated():
    out = _run([1], [1], [1], [10], 4.0)
    assert torch.allclose(out, torch.tensor([1.0]))


def test_envs_are_independent():
    rewards = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    terminated = torch.tensor([[True, False], [False, False]])
    out = n_step_returns(rewards, terminated, torch.zeros_like(terminated), torch.zeros(2, 2), torch.tensor([2.0, 2.0]), GAMMA)
    # env0: R1 = 1 + .5*2 = 2, R0 = 1 (terminated); env1: R1 = 2, R0 = 1 + .5*2 = 2
    assert torch.allclose(out, torch.tensor([[1.0, 2.0], [2.0, 2.0]]))
