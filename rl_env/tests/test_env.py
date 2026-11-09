"""Tests for CustomEnv. One test per rule in ENV_SPEC.md, named test_R<k>_<slug>."""
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless pygame for the render tests

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rl_env.env import ENV_VERSION, CustomEnv


@pytest.fixture
def env():
    return CustomEnv()


def test_check_env():
    check_env(CustomEnv())


def test_reset_matches_spaces(env):
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)


def test_step_matches_spaces(env):
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)


def _rollout(seed, n=20):
    env = CustomEnv()
    env.action_space.seed(seed)
    obs, _ = env.reset(seed=seed)
    traj = [obs]
    for _ in range(n):
        obs, r, term, trunc, _ = env.step(env.action_space.sample())
        traj.append((obs, r, term, trunc))
        if term or trunc:
            break
    return traj


def test_same_seed_same_trajectory():
    a, b = _rollout(123), _rollout(123)
    assert repr(a) == repr(b)


def test_env_version_in_info(env):
    _, info = env.reset(seed=0)
    assert info["env_version"] == ENV_VERSION
    *_, info = env.step(env.action_space.sample())
    assert info["env_version"] == ENV_VERSION


def test_truncation_at_max_steps():
    # Force the step counter (template attribute, domain-independent) so the test does not
    # depend on the dynamics. Retry a few seeds: termination wins over truncation.
    for seed in range(20):
        env = CustomEnv(max_steps=5)
        env.action_space.seed(seed)
        env.reset(seed=seed)
        env._steps = env.max_steps - 1
        *_, terminated, truncated, _ = env.step(env.action_space.sample())
        if not terminated:
            assert truncated
            return
    pytest.fail("every sampled action terminated the episode: truncation is untestable")


def test_state_hook_rejects_invalid_state(env):
    with pytest.raises((ValueError, KeyError, TypeError)):
        env.reset(options={"state": {"__invalid__": None}})


# --- Rule tests (generated from ENV_SPEC.md) ---
N = 15
GRW, CE, CS, MS = 5.0, 0.5, 1.5, 300
BUMP, SHAPE = 0.025, 0.2  # values for the opt-in terms of R11/R12 (both default to 0)
STEP_COST = CS * GRW / MS
CELL_REWARD = CE * GRW / N**2
NAV_N, NAV_E, NAV_S, NAV_W = 5, 8, 11, 14  # start index of each (empty_%, x_free, y_free) triple


def _env(**kwargs):
    kwargs.setdefault("n_obstacles", 0)
    return CustomEnv(**kwargs)


def _reset(env, **state):
    return env.reset(seed=0, options={"state": state})


def _agent_pos(obs):
    x, y = obs["navigator"][1:3] * N
    return int(round(float(x))), int(round(float(y)))


def _all_cells_except(excluded):
    return [(x, y) for x in range(N) for y in range(N) if (x, y) not in set(excluded)]


@pytest.mark.parametrize(
    "action, delta",
    [(0, (0, -1)), (1, (1, -1)), (2, (1, 0)), (3, (1, 1)), (4, (0, 1)), (5, (-1, 1)), (6, (-1, 0)), (7, (-1, -1))],
)
def test_R1_king_move_directions(action, delta):
    env = _env()
    _reset(env, agent=(7, 7), goal=(14, 14))
    obs, *_ = env.step(action)
    assert _agent_pos(obs) == (7 + delta[0], 7 + delta[1])


def test_R2_blocked_move_keeps_position():
    env = _env(n_obstacles=1)
    _reset(env, obstacles=[(8, 7)], agent=(7, 7), goal=(14, 14))
    obs, *_ = env.step(2)  # E into the obstacle
    assert _agent_pos(obs) == (7, 7)
    # diagonal between two diagonally adjacent obstacles is allowed: only the destination matters
    env = _env(n_obstacles=2)
    _reset(env, obstacles=[(8, 7), (7, 8)], agent=(7, 7), goal=(14, 14))
    obs, *_ = env.step(3)  # SE
    assert _agent_pos(obs) == (8, 8)
    # outside the grid
    env = _env()
    for action in (0, 6, 7):  # N, W, NW from the top-left corner
        _reset(env, agent=(0, 0), goal=(14, 14))
        obs, *_ = env.step(action)
        assert _agent_pos(obs) == (0, 0)
    _reset(env, agent=(14, 14), goal=(0, 0))
    obs, *_ = env.step(3)  # SE from the bottom-right corner
    assert _agent_pos(obs) == (14, 14)


def test_R3_layout_distinct_and_constant():
    for seed in range(30):
        env = CustomEnv()
        env.action_space.seed(seed)
        env.reset(seed=seed)
        obstacles, goal = env._obstacles.copy(), env._goal
        assert obstacles.sum() == 60
        assert not obstacles[env._agent[1], env._agent[0]]
        assert not obstacles[goal[1], goal[0]]
        assert env._agent != goal
        assert max(abs(env._agent[0] - goal[0]), abs(env._agent[1] - goal[1])) >= 8
        for _ in range(25):
            *_, terminated, truncated, _ = env.step(env.action_space.sample())
            assert np.array_equal(env._obstacles, obstacles) and env._goal == goal
            if terminated or truncated:
                break
    env = _env(n_obstacles=2)
    bad_states = [
        {"obstacles": [(1, 1)]},  # wrong count
        {"obstacles": [(1, 1), (1, 1)]},  # duplicate
        {"obstacles": [(1, 1), (2, 2)], "agent": (1, 1)},  # agent on obstacle
        {"obstacles": [(1, 1), (2, 2)], "goal": (2, 2)},  # goal on obstacle
        {"agent": (3, 3), "goal": (3, 3)},  # agent == goal
        {"agent": (15, 0)},  # outside the grid
        {"steps": 300},
    ]
    for state in bad_states:
        with pytest.raises(ValueError):
            env.reset(options={"state": state})


def test_R4_same_seed_same_layout():
    def layout(seed):
        env = CustomEnv()
        env.reset(seed=seed)
        return env._obstacles.copy(), env._agent, env._goal

    a, b = layout(7), layout(7)
    assert np.array_equal(a[0], b[0]) and a[1:] == b[1:]
    others = [layout(s) for s in range(8, 18)]
    assert any(not np.array_equal(a[0], o[0]) or a[1:] != o[1:] for o in others)


def test_R5_goal_terminates():
    env = _env()
    _reset(env, agent=(7, 7), goal=(8, 7))
    *_, terminated, truncated, info = env.step(1)  # NE: not the goal
    assert not terminated and not info["is_success"]
    _reset(env, agent=(7, 7), goal=(8, 7))
    *_, terminated, truncated, info = env.step(2)  # E: goal
    assert terminated and not truncated and info["is_success"]


def test_R6_truncation_at_max_steps():
    env = _env(max_steps=10)
    _reset(env, agent=(0, 0), goal=(14, 14), steps=8)
    *_, terminated, truncated, _ = env.step(4)
    assert not terminated and not truncated  # 9 steps
    *_, terminated, truncated, _ = env.step(4)
    assert not terminated and truncated  # 10 steps
    # reaching the goal on the last step is termination, not truncation
    _reset(env, agent=(7, 7), goal=(8, 7), steps=9)
    *_, terminated, truncated, _ = env.step(2)
    assert terminated and not truncated


def test_R7_perception_window():
    env = _env(n_obstacles=2)
    obs, _ = _reset(env, obstacles=[(6, 6), (8, 7)], agent=(7, 7), goal=(14, 14))
    p = obs["perception"]
    assert p.shape == (1, 3, 3) and p.dtype == np.float32
    expected = np.zeros((3, 3), dtype=np.float32)
    expected[0, 0] = 1  # (6, 6): window top-left
    expected[1, 2] = 1  # (8, 7): row 7 -> index 1, column 8 -> index 2
    assert np.array_equal(p[0], expected)
    # window hanging over the top-left corner: outside the grid reads as obstacle
    obs, _ = _reset(env, obstacles=[(14, 14), (14, 13)], agent=(0, 0), goal=(10, 10))
    expected = np.ones((3, 3), dtype=np.float32)
    expected[1:, 1:] = 0
    assert np.array_equal(obs["perception"][0], expected)
    # window=5: same rule on a 5x5 window
    env = _env(n_obstacles=2, window=5)
    obs, _ = _reset(env, obstacles=[(5, 5), (9, 8)], agent=(7, 7), goal=(14, 14))
    p = obs["perception"]
    assert p.shape == (1, 5, 5) and p.dtype == np.float32
    expected = np.zeros((5, 5), dtype=np.float32)
    expected[0, 0] = 1  # (5, 5): window top-left
    expected[3, 4] = 1  # (9, 8): row 8 -> index 3, column 9 -> index 4
    assert np.array_equal(p[0], expected)
    obs, _ = _reset(env, obstacles=[(14, 14), (14, 13)], agent=(0, 0), goal=(10, 10))
    expected = np.ones((5, 5), dtype=np.float32)
    expected[2:, 2:] = 0
    assert np.array_equal(obs["perception"][0], expected)


def test_R8_goal_discovery():
    env = _env()
    obs, _ = _reset(env, agent=(7, 7), goal=(7, 12))
    assert np.allclose(obs["navigator"][3:5], (-1, -1))
    for expected_y in (8, 9, 10):  # window reaches y + 1: still not covering y = 12
        obs, *_ = env.step(4)
        assert np.allclose(obs["navigator"][3:5], (-1, -1))
    obs, *_, info = env.step(4)  # agent at (7, 11): goal at the window edge
    assert info["goal_discovered"]
    assert np.allclose(obs["navigator"][3:5], (0, (11 - 12) / N))
    obs, *_ = env.step(0)  # moving away keeps it discovered and recomputes the offset
    assert np.allclose(obs["navigator"][3:5], (0, (10 - 12) / N))
    # goal already inside the window at reset (forced state)
    obs, _ = _reset(env, agent=(7, 7), goal=(8, 8))
    assert np.allclose(obs["navigator"][3:5], (-1 / N, -1 / N))


def _sector_obs(unexplored, agent=(7, 7)):
    env = _env()
    obs, _ = _reset(env, agent=agent, goal=(0, 0), explored=_all_cells_except(unexplored))
    return obs["navigator"]


def test_R9_navigator_sectors():
    # one unexplored cell per sector, away from the diagonals
    nav = _sector_obs([(7, 2), (12, 7), (7, 12), (2, 7)])
    assert np.allclose(nav[NAV_N : NAV_N + 3], (0.25, 0, 5 / N))
    assert np.allclose(nav[NAV_E : NAV_E + 3], (0.25, -5 / N, 0))
    assert np.allclose(nav[NAV_S : NAV_S + 3], (0.25, 0, -5 / N))
    assert np.allclose(nav[NAV_W : NAV_W + 3], (0.25, 5 / N, 0))
    # cells exactly on the diagonals: NW -> N, NE -> E, SE -> S, SW -> W
    nav = _sector_obs([(3, 3), (11, 3), (11, 11), (3, 11)])
    assert np.allclose(nav[NAV_N : NAV_N + 3], (0.25, 4 / N, 4 / N))
    assert np.allclose(nav[NAV_E : NAV_E + 3], (0.25, -4 / N, 4 / N))
    assert np.allclose(nav[NAV_S : NAV_S + 3], (0.25, -4 / N, -4 / N))
    assert np.allclose(nav[NAV_W : NAV_W + 3], (0.25, 4 / N, -4 / N))
    # tie on Chebyshev distance: smaller row, then smaller column; empty sectors give (0, -1, -1)
    nav = _sector_obs([(8, 3), (6, 3)])
    assert np.allclose(nav[NAV_N : NAV_N + 3], (1.0, 1 / N, 4 / N))
    for start in (NAV_E, NAV_S, NAV_W):
        assert np.allclose(nav[start : start + 3], (0, -1, -1))
    # nearest is by Chebyshev distance, not by Euclidean or Manhattan
    nav = _sector_obs([(10, 5), (7, 3)])  # (10, 5): dx=3, dy=-2 -> E, cheb 3; (7, 3): N, cheb 4
    assert np.allclose(nav[NAV_E : NAV_E + 3], (0.5, -3 / N, 2 / N))
    assert np.allclose(nav[NAV_N : NAV_N + 3], (0.5, 0, 4 / N))
    # fully explored world: all sectors empty
    nav = _sector_obs([])
    for start in (NAV_N, NAV_E, NAV_S, NAV_W):
        assert np.allclose(nav[start : start + 3], (0, -1, -1))


def test_R10_reward_terms():
    env = _env(bump_penalty=0.0, shaping_reward=0.0)  # isolate the base terms; R11/R12 have their own tests
    _reset(env, agent=(7, 7), goal=(14, 14))
    _, reward, *_ = env.step(2)  # E: 3 new cells (column x = 9, rows 6..8)
    assert reward == pytest.approx(3 * CELL_REWARD - STEP_COST)
    _, reward, *_ = env.step(6)  # back W: window already explored
    assert reward == pytest.approx(-STEP_COST)
    _reset(env, agent=(7, 7), goal=(14, 14))
    _, reward, *_ = env.step(1)  # NE: 5 new cells (column x = 9 rows 5..7, row y = 5 cols 7..8)
    assert reward == pytest.approx(5 * CELL_REWARD - STEP_COST)
    _reset(env, agent=(0, 0), goal=(14, 14))
    _, reward, *_ = env.step(0)  # blocked by the border: stays, nothing new, still pays the step (bump disabled here)
    assert reward == pytest.approx(-STEP_COST)
    _reset(env, agent=(7, 7), goal=(8, 7))
    _, reward, terminated, *_ = env.step(2)  # reaches the goal: exploration + step + GRW
    assert terminated and reward == pytest.approx(3 * CELL_REWARD - STEP_COST + GRW)
    # overall bounds with the defaults: timeout without exploring costs coeff_step * GRW
    assert MS * STEP_COST == pytest.approx(CS * GRW)
    # defaults match the QML_Thesis reward: no bump penalty, no goal-distance shaping
    d = CustomEnv()
    assert (d.goal_reward, d.coeff_expl, d.coeff_step, d.bump_penalty, d.shaping_reward) == (GRW, CE, CS, 0.0, 0.0)


def test_R11_bump_penalty():
    env = _env(bump_penalty=BUMP)
    _reset(env, agent=(0, 0), goal=(14, 14))
    _, reward, *_ = env.step(0)  # N: outside the grid
    assert reward == pytest.approx(-STEP_COST - BUMP)
    _reset(env, agent=(7, 7), goal=(14, 14))
    env.step(2)
    _, reward, *_ = env.step(6)  # free move W, nothing new: only the step cost
    assert reward == pytest.approx(-STEP_COST)
    env = _env(n_obstacles=1, bump_penalty=BUMP)
    _reset(env, obstacles=[(8, 7)], agent=(7, 7), goal=(14, 14))
    _, reward, *_ = env.step(2)  # E: obstacle
    assert env._agent == (7, 7) and reward == pytest.approx(-STEP_COST - BUMP)
    _reset(env, obstacles=[(8, 7)], agent=(7, 7), goal=(14, 14))
    _, reward, *_ = env.step(0)  # N: free move, 3 new cells (row y = 5, columns 6..8)
    assert reward == pytest.approx(3 * CELL_REWARD - STEP_COST)


def test_R12_goal_distance_shaping():
    env = _env(shaping_reward=SHAPE)
    # goal not discovered: no shaping, even when moving towards it
    _reset(env, agent=(7, 7), goal=(14, 14))
    _, reward, *_ = env.step(3)  # SE: d 7 -> 6, goal still outside the window, 5 new cells
    assert reward == pytest.approx(5 * CELL_REWARD - STEP_COST)
    # goal discovered at reset (forced explored, outside the window): closer / farther / same distance
    _reset(env, agent=(7, 7), goal=(9, 7), explored=[(9, 7)])
    _, reward, *_ = env.step(2)  # E: d 2 -> 1, 2 new cells (column x = 9 rows 6 and 8; (9, 7) already explored)
    assert reward == pytest.approx(2 * CELL_REWARD - STEP_COST + SHAPE)
    _reset(env, agent=(7, 7), goal=(9, 7), explored=[(9, 7)])
    _, reward, *_ = env.step(6)  # W: d 2 -> 3, 3 new cells
    assert reward == pytest.approx(3 * CELL_REWARD - STEP_COST - SHAPE)
    _reset(env, agent=(7, 7), goal=(9, 7), explored=[(9, 7)])
    _, reward, *_ = env.step(0)  # N: d stays 2 (Chebyshev ring), 3 new cells
    assert reward == pytest.approx(3 * CELL_REWARD - STEP_COST)
    # blocked move with discovered goal: distance unchanged, no shaping
    _reset(env, agent=(0, 0), goal=(2, 0), explored=[(2, 0)])
    _, reward, *_ = env.step(0)
    assert reward == pytest.approx(-STEP_COST)
    # reaching the goal also earns the shaping term
    _reset(env, agent=(7, 7), goal=(8, 7))
    _, reward, terminated, *_ = env.step(2)
    assert terminated and reward == pytest.approx(3 * CELL_REWARD - STEP_COST + SHAPE + GRW)
    # discovery step: shaping applies when the goal enters the window with this move
    _reset(env, agent=(7, 7), goal=(9, 7))
    _, reward, *_ = env.step(2)  # agent at (8, 7): window x 7..9 now contains the goal, d 2 -> 1
    assert reward == pytest.approx(3 * CELL_REWARD - STEP_COST + SHAPE)


def _mem_row(action, dx, dy):
    row = np.zeros(10, dtype=np.float32)
    row[action] = 1.0
    row[8], row[9] = dx, dy
    return row


def test_R13_memory_rows_shift():
    env = _env(memory_k=3, memory_m=3)
    obs, _ = _reset(env, agent=(7, 7), goal=(14, 14))
    assert obs["memory"].shape == (3, 10) and not obs["memory"].any()
    obs, *_ = env.step(2)  # E
    assert not obs["memory"][:2].any()  # rows of moves not made yet stay 0
    assert obs["memory"][2, :8].tolist() == [0, 0, 1, 0, 0, 0, 0, 0]
    for action in (4, 6, 0):  # S, W, N
        obs, *_ = env.step(action)
    # oldest first: the E move has slid out of the 3-row window
    assert [int(np.argmax(r[:8])) for r in obs["memory"]] == [4, 6, 0]
    assert np.all(obs["memory"][:, :8].sum(axis=1) == 1.0)


def test_R14_memory_bump_counts_as_move():
    env = _env()
    obs, _ = _reset(env, agent=(0, 0), goal=(14, 14))
    obs, *_ = env.step(0)  # N: outside the grid
    np.testing.assert_allclose(obs["memory"][-1], _mem_row(0, 0.0, 0.0))
    env = _env(n_obstacles=1)
    _reset(env, obstacles=[(7, 5)], agent=(7, 7), goal=(14, 14))
    env.step(0)  # N, free: (7, 6)
    obs, *_ = env.step(0)  # N into the obstacle: position unchanged at (7, 6)
    assert env._agent == (7, 6)
    np.testing.assert_allclose(obs["memory"][-1], _mem_row(0, 0.0, -0.1), atol=1e-6)
    np.testing.assert_allclose(obs["memory"][-2], _mem_row(0, 0.0, -0.1), atol=1e-6)


def test_R15_memory_displacement_reference():
    env = _env(memory_k=5, memory_m=3)
    _reset(env, agent=(7, 7), goal=(14, 14))
    for _ in range(2):
        obs, *_ = env.step(2)  # E, E: fewer than m moves, reference = initial position (7, 7)
    np.testing.assert_allclose(obs["memory"][-2], _mem_row(2, 1 / 3, 0.0), atol=1e-6)
    np.testing.assert_allclose(obs["memory"][-1], _mem_row(2, 2 / 3, 0.0), atol=1e-6)
    obs, *_ = env.step(2)  # (10, 7), reference (7, 7)
    np.testing.assert_allclose(obs["memory"][-1], _mem_row(2, 1.0, 0.0), atol=1e-6)
    obs, *_ = env.step(2)  # (11, 7), reference (8, 7)
    np.testing.assert_allclose(obs["memory"][-1], _mem_row(2, 1.0, 0.0), atol=1e-6)
    obs, *_ = env.step(4)  # S: (11, 8), reference (9, 7)
    np.testing.assert_allclose(obs["memory"][-1], _mem_row(4, 2 / 3, 1 / 3), atol=1e-6)
    # rows already written never change
    np.testing.assert_allclose(obs["memory"][0], _mem_row(2, 1 / 3, 0.0), atol=1e-6)


def test_R16_memory_reset_and_copy():
    env = _env()
    _reset(env, agent=(7, 7), goal=(14, 14))
    obs, *_ = env.step(2)
    snapshot = obs["memory"].copy()
    env.step(4)
    np.testing.assert_array_equal(obs["memory"], snapshot)  # returned array is a copy
    assert env.step(6)[0]["memory"] is not obs["memory"]
    obs, _ = env.reset(seed=1)
    assert not obs["memory"].any()
    obs, _ = _reset(env, agent=(3, 3), goal=(14, 14), steps=5)  # forced state: memory still zero
    assert not obs["memory"].any()
    obs, *_ = env.step(2)  # history restarts at the forced agent; `steps` does not matter
    np.testing.assert_allclose(obs["memory"][-1], _mem_row(2, 0.1, 0.0), atol=1e-6)


def test_memory_params():
    env = CustomEnv(memory_k=4, memory_m=2)
    assert env.observation_space["memory"].shape == (4, 10)
    assert env.reset(seed=0)[0]["memory"].shape == (4, 10)
    for kwargs in ({"memory_k": 0}, {"memory_m": 0}):
        with pytest.raises(ValueError):
            CustomEnv(**kwargs)


def _layout(env):
    obstacles = tuple(sorted((int(x), int(y)) for y, x in zip(*np.nonzero(env._obstacles))))
    return obstacles, env._agent, env._goal


def _pool_layouts(**kwargs):
    env = CustomEnv(**kwargs)
    return [(tuple(sorted(o)), a, g) for o, a, g in env._layout_pool]


def test_R18_layout_pool():
    pool = _pool_layouts(layout_pool=5, layout_pool_seed=3)
    assert len(pool) == 5 and len(set(pool)) == 5
    # every reset (without the state hook) uses a pool member, and every member is drawn
    env = CustomEnv(layout_pool=5, layout_pool_seed=3)
    seen = set()
    for seed in range(500):
        env.reset(seed=seed)
        layout = _layout(env)
        assert layout in pool
        seen.add(layout)
        if seed < 3:  # the layout stays constant within the episode
            for action in range(8):
                env.step(action)
            assert _layout(env)[0] == layout[0] and env._goal == layout[2]
    assert seen == set(pool)
    # same pool seed -> same pool, whatever the reset seed; another pool seed -> another pool
    assert _pool_layouts(layout_pool=5, layout_pool_seed=3) == pool
    other = CustomEnv(layout_pool=5, layout_pool_seed=3)
    other.reset(seed=12345)
    assert [(tuple(sorted(o)), a, g) for o, a, g in other._layout_pool] == pool
    assert set(_pool_layouts(layout_pool=5, layout_pool_seed=4)).isdisjoint(pool)
    # pool layouts satisfy the layout constraints (R3) and the start-goal distance
    for obstacles, agent, goal in _pool_layouts(layout_pool=50, layout_pool_seed=0, window=5, n_obstacles=30):
        assert len(set(obstacles)) == 30
        assert agent not in obstacles and goal not in obstacles and agent != goal
        assert max(abs(agent[0] - goal[0]), abs(agent[1] - goal[1])) >= 8
    # the state hook ignores the pool
    hooked = CustomEnv(n_obstacles=2, layout_pool=5)
    _reset(hooked, obstacles=[(1, 1), (2, 2)], agent=(7, 7), goal=(14, 14))
    assert _layout(hooked) == (((1, 1), (2, 2)), (7, 7), (14, 14))


def test_layout_pool_params():
    assert CustomEnv()._layout_pool == []
    for kwargs in ({"layout_pool": -1}, {"layout_pool_seed": -1}):
        with pytest.raises(ValueError):
            CustomEnv(**kwargs)
    # layout_pool=0 behaves as without the parameters, whatever layout_pool_seed
    a, b = CustomEnv(), CustomEnv(layout_pool=0, layout_pool_seed=9)
    for seed in range(5):
        a.reset(seed=seed)
        b.reset(seed=seed)
        assert _layout(a) == _layout(b)


def test_window_param():
    env = CustomEnv(window=5)
    assert env.observation_space["perception"].shape == (1, 5, 5)
    obs, _ = env.reset(seed=0)
    assert obs["perception"].shape == (1, 5, 5)
    assert env.observation_space.contains(obs)
    _, info = _reset(_env(window=5), agent=(7, 7), goal=(14, 14))
    assert info["explored_fraction"] == pytest.approx(25 / N**2)  # the initial 5x5 window is explored
    for window in (1, 2, 4, 17):
        with pytest.raises(ValueError):
            CustomEnv(window=window)


def test_R17_navigator_time_fraction():
    env = _env(max_steps=10)
    assert env.observation_space["navigator"].shape == (17,)
    obs, _ = _reset(env, agent=(7, 7), goal=(14, 14))
    assert obs["navigator"][0] == 0.0
    for k in range(1, 4):
        obs, *_ = env.step(2 if k % 2 else 6)
        assert obs["navigator"][0] == pytest.approx(k / 10)
    obs, _ = _reset(env, agent=(7, 7), goal=(14, 14), steps=6)  # forced steps
    assert obs["navigator"][0] == pytest.approx(0.6)
    obs, *_ = env.step(0)
    assert obs["navigator"][0] == pytest.approx(0.7)
    assert env.reset(seed=3)[0]["navigator"][0] == 0.0
    _reset(env, agent=(7, 7), goal=(14, 14), steps=9)
    obs, _, terminated, truncated, _ = env.step(0)
    assert truncated and obs["navigator"][0] == pytest.approx(1.0)


def test_layout_sampling_never_fails():
    # regression: the grid centre has no cell at Chebyshev distance >= 8, a start sampled there crashed reset
    env = _env()
    for seed in range(3000):
        env.reset(seed=seed)
        assert env._agent != env._goal


def test_info_fields():
    env = _env()
    _, info = _reset(env, agent=(7, 7), goal=(14, 14))
    assert info["explored_fraction"] == pytest.approx(9 / N**2)
    assert info["goal_discovered"] is False and info["is_success"] is False



# --- Render tests (non-semantic feature, not tied to a rule id; see ENV_SPEC.md section 9) ---
PX = 40  # CELL_PX


def _render_env(**kwargs):
    pytest.importorskip("pygame")
    env = _env(render_mode="human", **kwargs)
    env.metadata = {**env.metadata, "render_fps": 10_000}  # do not slow the tests down
    return env


def _pixel(cell):
    import pygame

    return tuple(pygame.display.get_surface().get_at((cell[0] * PX + PX // 2, cell[1] * PX + PX // 2)))[:3]


def _brightness(cell):
    return sum(_pixel(cell))


def test_render_rejects_unknown_mode():
    with pytest.raises(AssertionError):
        CustomEnv(render_mode="rgb_array")


def test_render_none_mode_returns_none(env):
    env.reset(seed=0)
    assert env.render() is None


def test_render_sectors_and_static_elements():
    env = _render_env(n_obstacles=1)
    _reset(env, obstacles=[(1, 1)], agent=(7, 7), goal=(14, 14))
    n_unexp, e_unexp, s_unexp, w_unexp = _pixel((7, 2)), _pixel((12, 7)), _pixel((7, 12)), _pixel((2, 7))
    assert len({n_unexp, e_unexp, s_unexp, w_unexp}) == 4  # one colour per sector
    # explored cell is paler than an unexplored one of the same sector
    assert _brightness((7, 6)) > _brightness((7, 2))
    assert _pixel((1, 1)) == (0, 0, 0)  # obstacle
    assert _pixel((14, 14)) == (250, 220, 40)  # goal
    # diagonal cells follow the NW->N / NE->E / SE->S / SW->W convention (same colour as the sector)
    assert _pixel((3, 3)) == n_unexp and _pixel((11, 3)) == e_unexp
    assert _pixel((11, 11)) == s_unexp and _pixel((3, 11)) == w_unexp
    # perception window outline
    import pygame

    surface = pygame.display.get_surface()
    assert tuple(surface.get_at(((7 - 1) * PX + 1, 7 * PX + PX // 2)))[:3] == (255, 255, 255)
    env.close()


def test_render_trail_fades_and_vanishes():
    env = _render_env()
    _reset(env, agent=(2, 2), goal=(14, 14))
    for _ in range(12):
        env.step(2)  # E along row 2: agent ends at (14, 2)
    # the last 10 cells left are x = 4..13, dots grow brighter with recency; x = 2, 3 are out of the window
    brightness = [_brightness((x, 2)) for x in range(4, 14)]
    assert brightness == sorted(brightness) and len(set(brightness)) == 10
    assert _pixel((2, 2)) == _pixel((2, 3)) and _pixel((3, 2)) == _pixel((3, 3))  # no dot: plain explored W colour
    env.close()


def test_render_trail_ignores_blocked_moves_and_resets():
    env = _render_env(n_obstacles=1)
    _reset(env, obstacles=[(6, 7)], agent=(3, 7), goal=(14, 14))
    env.step(2)
    env.step(2)  # agent at (5, 7), trail (3, 7), (4, 7)
    for _ in range(12):
        env.step(2)  # blocked by the obstacle at (6, 7): must not be recorded
    env.step(1)  # NE to (6, 6)
    assert _pixel((3, 7)) != _pixel((3, 8))  # oldest dot still there: bumps did not evict it
    explored = [(x, y) for x in range(15) for y in (7, 8)]
    _reset(env, obstacles=[(6, 7)], agent=(10, 7), goal=(0, 14), explored=explored)
    assert _pixel((3, 7)) == _pixel((3, 8))  # (3, 7) was in the old trail: reset must clear it
    env.close()


def test_render_runs_full_episode_and_close():
    env = _render_env(max_steps=15)
    env.reset(seed=3)
    for _ in range(15):
        *_, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            break
    env.close()
    env.close()  # idempotent


def _bar_bytes():
    import pygame

    surface = pygame.display.get_surface()
    bar = surface.subsurface((0, N * PX, N * PX, surface.get_height() - N * PX))
    return pygame.image.tobytes(bar, "RGB")


def test_render_episode_return_is_sum_of_rewards():
    for seed in range(5):
        env = CustomEnv()
        env.action_space.seed(seed)
        env.reset(seed=seed)
        total = 0.0
        for _ in range(60):
            _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            total += reward
            assert env._episode_return == pytest.approx(total)
            if terminated or truncated:
                break
    # goal reward is included, and reset (also with a forced state) zeroes the counter
    env = _env()
    _reset(env, agent=(7, 7), goal=(8, 7))
    _, reward, terminated, *_ = env.step(2)
    assert terminated and env._episode_return == pytest.approx(reward) and reward > GRW - 1
    _reset(env, agent=(7, 7), goal=(14, 14))
    assert env._episode_return == 0.0
    env.step(2)
    env.reset(seed=1)
    assert env._episode_return == 0.0


def test_render_status_bar_below_grid_shows_return():
    import pygame

    def bar_after(action):
        env = _render_env()
        _reset(env, agent=(0, 0), goal=(14, 14))
        env.step(action)  # same step count in both scenarios, different return
        size, data = pygame.display.get_surface().get_size(), _bar_bytes()
        env.close()
        return size, data

    (size_a, bump), (size_b, explore), (_, bump_again) = bar_after(0), bar_after(3), bar_after(0)
    assert size_a == size_b == (N * PX, N * PX + 32)
    assert bump != explore  # blocked step (return < 0) vs exploring step (return > 0)
    assert bump == bump_again  # deterministic drawing


# --- navigator tables: refactor check, not a spec rule ---

def _reference_navigator(env):
    """Navigator as computed before the precomputed tables (ENV_SPEC v8 changelog), kept as an oracle."""
    from rl_env.env import _sector_of

    n, (x, y), (gx, gy) = env.N, env._agent, env._goal
    nav = np.full(17, -1.0, dtype=np.float32)
    nav[0] = env._steps / env.max_steps
    nav[1], nav[2] = x / n, y / n
    if env._explored[gy, gx]:
        nav[3], nav[4] = (x - gx) / n, (y - gy) / n
    uy, ux = np.nonzero(~env._explored)
    total = uy.size
    for s in range(4):
        nav[5 + 3 * s] = 0.0
    if total:
        dx, dy = ux - x, uy - y
        sector = _sector_of(dx, dy)
        key = (np.maximum(np.abs(dx), np.abs(dy)) * n + uy) * n + ux
        for s in range(4):
            members = np.flatnonzero(sector == s)
            if members.size == 0:
                continue
            best = members[np.argmin(key[members])]
            nav[5 + 3 * s] = members.size / total
            nav[6 + 3 * s] = (x - ux[best]) / n
            nav[7 + 3 * s] = (y - uy[best]) / n
    return nav


@pytest.mark.parametrize("seed", range(20))
def test_navigator_tables_match_reference_on_rollouts(seed):
    env = CustomEnv()
    env.action_space.seed(seed)
    obs, _ = env.reset(seed=seed)
    for _ in range(300):
        assert np.array_equal(obs["navigator"], _reference_navigator(env))
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            obs, _ = env.reset()


def test_navigator_tables_match_reference_on_arbitrary_masks():
    """Random explored masks over every agent position, plus the all-explored and agent-cell-unexplored corners."""
    env = CustomEnv()
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for y in range(N):
        for x in range(N):
            env._agent = (x, y)
            for p in (0.0, 0.3, 0.7, 0.97, 1.0):
                env._explored = rng.random((N, N)) < p
                assert np.array_equal(env._get_obs()["navigator"], _reference_navigator(env))
