import numpy as np
import torch
from torch import nn

from rl_train.envs import make_vec_env
from rl_train.policy import Policy
from rl_train.rollout import RolloutCollector, obs_to_tensor


class AgentXValue(nn.Module):
    """Uniform policy; value = agent x / n (navigator[1]), so V identifies which obs was evaluated."""

    def forward(self, obs):
        nav = obs["navigator"]
        return torch.zeros(nav.shape[0], 8), nav[:, 1]


def test_reference_model_contract():
    model = Policy()
    vec = make_vec_env(4)
    obs, _ = vec.reset(seed=0)
    logits, value = model(obs_to_tensor(obs))
    assert logits.shape == (4, 8) and value.shape == (4,)


def test_rollout_shapes_and_episode_stats():
    torch.manual_seed(0)
    vec = make_vec_env(2, max_steps=3)
    ro = RolloutCollector(AgentXValue(), vec, n_steps=7, seed=0).collect()
    assert ro.obs["navigator"].shape == (7, 2, 17) and ro.obs["perception"].shape == (7, 2, 1, 3, 3)
    assert ro.actions.shape == ro.rewards.shape == ro.final_values.shape == (7, 2)
    assert ro.last_value.shape == (2,)
    # max_steps=3: episodes end at steps 3 and 6 in both envs, all by truncation
    assert ro.truncated[[2, 5]].all() and ro.truncated.sum() == 4 and not ro.terminated.any()
    assert ro.ep_lengths == [3, 3, 3, 3]
    expected = [ro.rewards[0:3, 0].sum(), ro.rewards[0:3, 1].sum(), ro.rewards[3:6, 0].sum(), ro.rewards[3:6, 1].sum()]
    assert np.allclose(ro.ep_returns, [float(x) for x in expected], atol=1e-5)
    assert len(ro.ep_explored) == 4 and all(0.0 < e <= 1.0 for e in ro.ep_explored)


def test_truncation_bootstraps_terminal_obs_not_next_episode_obs():
    torch.manual_seed(0)
    n_steps, max_steps = 3, 3
    ro = RolloutCollector(AgentXValue(), make_vec_env(2, max_steps=max_steps), n_steps=n_steps, seed=0).collect()

    # replay the same actions in an identical env and capture the true terminal / next-episode obs
    replay = make_vec_env(2, max_steps=max_steps)
    replay.reset(seed=0)
    for t in range(n_steps):
        obs, _, _, _, info = replay.step(ro.actions[t].numpy())
    terminal_x = np.array([info["final_obs"][i]["navigator"][1] for i in range(2)], dtype=np.float32)
    next_x = obs["navigator"][:, 1]

    assert torch.allclose(ro.final_values[2], torch.tensor(terminal_x))
    assert torch.allclose(ro.last_value, torch.tensor(next_x))  # last_value is the new episode's first obs
    assert not np.allclose(terminal_x, next_x)  # the two are really different, so the check above bites


def test_envs_keep_running_across_rollouts():
    torch.manual_seed(0)
    collector = RolloutCollector(AgentXValue(), make_vec_env(1, max_steps=5), n_steps=3, seed=0)
    first = collector.collect()
    obs_after_first = collector.obs["navigator"].copy()
    second = collector.collect()
    # no reset in between: the second rollout starts from the obs that followed the first one
    assert np.allclose(second.obs["navigator"][0].numpy(), obs_after_first)
    assert first.ep_lengths == [] and second.ep_lengths == [5]  # episode of 5 spans both rollouts
