import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("render_world", ROOT / "scripts" / "render_world.py")
rw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rw)


def test_scripted_explorer_reaches_the_goal_and_never_picks_a_blocked_first_step():
    from rl_env.env import CustomEnv
    env = CustomEnv(N=15, n_obstacles=30, window=5, max_steps=300)
    env.reset(seed=3)
    for _ in range(300):
        before = env._agent
        _, _, terminated, truncated, _ = env.step(rw.next_move(env))
        assert env._agent != before or terminated or truncated  # a path step is never into an obstacle or the border
        if terminated or truncated:
            break
    assert env._agent == env._goal
