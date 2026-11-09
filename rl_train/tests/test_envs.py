from rl_train.config import A2CConfig
from rl_train.envs import make_env, make_vec_env


def test_make_env_applies_kwargs():
    env = make_env(max_steps=7)
    assert env.max_steps == 7


def test_vec_env_same_step_terminal_info():
    """Pins the gymnasium SAME_STEP behaviour the rollout relies on (spec assumption 7)."""
    vec = make_vec_env(3, max_steps=4)
    obs, _ = vec.reset(seed=0)
    assert set(obs) == {"navigator", "perception", "memory"}
    assert obs["navigator"].shape == (3, 17) and obs["perception"].shape == (3, 1, 3, 3)
    assert obs["memory"].shape == (3, 10, 10)

    for _ in range(4):
        obs, _, terminated, truncated, info = vec.step(vec.action_space.sample())
    done = terminated | truncated
    assert done.all()  # max_steps=4 and the goal is at distance >= 8, so every env times out
    assert truncated.all() and not terminated.any()

    # terminal obs/info live in final_obs/final_info, masked per env
    assert info["_final_obs"].all() and info["_final_info"].all()
    for i in range(3):
        assert set(info["final_obs"][i]) == {"navigator", "perception", "memory"}
    assert info["final_info"]["explored_fraction"].shape == (3,)
    assert info["final_info"]["is_success"].dtype == bool
    # the top-level info is the one of the new episode (only the initial window explored)
    assert (info["explored_fraction"] <= 9 / 225 + 1e-9).all()


def test_config_roundtrip():
    cfg = A2CConfig(run_name="x", env_kwargs={"max_steps": 10})
    d = cfg.to_dict()
    assert d["run_name"] == "x" and d["env_kwargs"] == {"max_steps": 10}
