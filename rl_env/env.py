"""Partially observable N x N grid: a chess-king agent must reach a goal cell among fixed obstacles.

The layout (obstacles, start, goal) is re-sampled at every reset. The agent sees a local window x window
(default 3x3) perception window plus a compact navigator vector summarising what it has not explored yet.

Source of truth: ENV_SPEC.md. Update the spec first, then this file.
"""
from __future__ import annotations

from collections import deque
from functools import lru_cache
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

ENV_VERSION = 13

WINDOW = 3  # default side of the perception window
MIN_START_GOAL_DIST = 8  # Chebyshev, applied only when start and goal are both sampled
MAX_LAYOUT_TRIES = 100
# (dx, dy) for actions N, NE, E, SE, S, SW, W, NW; x = column, y = row (grows downward)
MOVES = ((0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1))
NAV_DIM = 1 + 4 + 3 * 4
MEMORY_COLS = 8 + 2  # action one-hot + (dx, dy) displacement
STATE_KEYS = {"obstacles", "agent", "goal", "explored", "steps"}

# Rendering (human mode only, see ENV_SPEC.md section 9)
TRAIL_LEN = 10
CELL_PX = 40
SECTOR_COLORS = ((70, 130, 220), (60, 170, 90), (240, 150, 40), (150, 90, 200))  # N, E, S, W
OBSTACLE_COLOR = (0, 0, 0)
GOAL_COLOR = (250, 220, 40)
AGENT_CELL_COLOR = (200, 200, 200)
GRID_COLOR = (40, 40, 40)
BAR_PX = 32  # status bar below the grid
BAR_COLOR = (25, 25, 25)
TEXT_COLOR = (235, 235, 235)


def _sector_of(dx, dy):
    """Sector index (0=N, 1=E, 2=S, 3=W) of cells at offset (dx, dy) from the agent.

    The diagonals split the plane in four wedges; cells exactly on a diagonal go NW->N, NE->E,
    SE->S, SW->W. The offset (0, 0) is meaningless (the agent's own cell).
    """
    adx, ady = np.abs(dx), np.abs(dy)
    diag = np.where(dy < 0, np.where(dx < 0, 0, 1), np.where(dx > 0, 2, 3))
    return np.where(adx > ady, np.where(dx > 0, 1, 3), np.where(ady > adx, np.where(dy < 0, 0, 2), diag))


@lru_cache
def _navigator_tables(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Per agent position `pos = y * n + x`, computed once per grid size:

    order[pos]   (n*n,) every cell index `y * n + x`, grouped by sector (N, E, S, W) and, inside a
                 sector, sorted by the navigator key (Chebyshev distance, row, column)
    starts[pos]  (5,)   start of each sector in order[pos], plus the end

    The agent's own cell is included (sector W, distance 0, as `_sector_of` gives it), so the
    result matches a direct computation even for a forced state where it is unexplored.
    """
    cy, cx = np.divmod(np.arange(n * n), n)
    order = np.empty((n * n, n * n), dtype=np.int64)
    starts = np.empty((n * n, 5), dtype=np.int64)
    for pos in range(n * n):
        y, x = divmod(pos, n)
        dx, dy = cx - x, cy - y
        sector = _sector_of(dx, dy)
        key = (np.maximum(np.abs(dx), np.abs(dy)) * n + cy) * n + cx
        order[pos] = np.lexsort((key, sector))
        starts[pos] = np.concatenate([[0], np.cumsum(np.bincount(sector, minlength=4))])
    return order, starts


def _tint(color, explored):
    """Pale tint for explored cells, darker shade for unexplored ones."""
    if explored:
        return tuple(int(0.45 * c + 0.55 * 255) for c in color)
    return tuple(int(0.55 * c) for c in color)


class CustomEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        render_mode: str | None = None,
        max_steps: int = 300,
        N: int = 15,
        n_obstacles: int = 60,
        goal_reward: float = 5.0,
        coeff_expl: float = 0.5,
        coeff_step: float = 1.5,
        bump_penalty: float = 0.0,
        shaping_reward: float = 0.0,
        memory_k: int = 10,
        memory_m: int = 10,
        window: int = WINDOW,
        layout_pool: int = 0,
        layout_pool_seed: int = 0,
    ):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        if N < MIN_START_GOAL_DIST + 1:
            raise ValueError(f"N must be >= {MIN_START_GOAL_DIST + 1} to sample start and goal")
        if not 0 <= n_obstacles <= N * N - 2:
            raise ValueError("n_obstacles must be in [0, N*N - 2]")
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if memory_k < 1 or memory_m < 1:
            raise ValueError("memory_k and memory_m must be >= 1")
        if window % 2 == 0 or not 3 <= window <= 2 * MIN_START_GOAL_DIST - 1:
            # half the window must stay below the start-goal distance, so the goal is never visible at reset
            raise ValueError(f"window must be odd and in [3, {2 * MIN_START_GOAL_DIST - 1}]")
        if layout_pool < 0 or layout_pool_seed < 0:
            raise ValueError("layout_pool and layout_pool_seed must be >= 0")
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.N = N
        self.n_obstacles = n_obstacles
        self.goal_reward = goal_reward
        self.coeff_expl = coeff_expl
        self.coeff_step = coeff_step
        self.bump_penalty = bump_penalty
        self.shaping_reward = shaping_reward
        self.memory_k = memory_k
        self.memory_m = memory_m
        self.window = window
        self._half = window // 2
        if render_mode == "human":
            import pygame  # noqa: F401  (fail early if the optional dependency is missing)

        self.observation_space = spaces.Dict(
            {
                "navigator": spaces.Box(-1.0, 1.0, shape=(NAV_DIM,), dtype=np.float32),
                "perception": spaces.Box(0.0, 1.0, shape=(1, window, window), dtype=np.float32),
                "memory": spaces.Box(-1.0, 1.0, shape=(memory_k, MEMORY_COLS), dtype=np.float32),
            }
        )
        self.action_space = spaces.Discrete(8)

        self._steps = 0
        self._episode_return = 0.0  # cumulative reward since reset, display only
        self._obstacles = np.zeros((N, N), dtype=bool)  # indexed [y, x]
        self._padded = np.ones((N + 2 * self._half, N + 2 * self._half), dtype=np.float32)
        self._explored = np.zeros((N, N), dtype=bool)
        self._nav_order, self._nav_starts = _navigator_tables(N)
        self._agent = (0, 0)
        self._goal = (N - 1, N - 1)
        self._trail: deque[tuple[int, int]] = deque(maxlen=TRAIL_LEN)  # cells left, oldest first
        self._memory = np.zeros((memory_k, MEMORY_COLS), dtype=np.float32)  # oldest move first
        self._pos_history: deque[tuple[int, int]] = deque(maxlen=memory_m + 1)  # positions, oldest first
        self._window = None  # pygame display surface, created lazily by render()
        self._clock = None
        self._font = None
        self._layout_pool = self._sample_layout_pool(layout_pool, layout_pool_seed)

    # ------------------------------------------------------------------ observation

    def _get_obs(self):
        n = self.N
        x, y = self._agent
        gx, gy = self._goal

        perception = self._padded[y : y + self.window, x : x + self.window][None].copy()

        nav = np.full(NAV_DIM, -1.0, dtype=np.float32)
        nav[0] = self._steps / self.max_steps  # R17
        nav[1], nav[2] = x / n, y / n
        if self._explored[gy, gx]:  # goal discovered
            nav[3], nav[4] = (x - gx) / n, (y - gy) / n

        # unexplored cells read in (sector, Chebyshev, row, column) order: the first of each sector is the nearest
        pos = y * n + x
        cells = self._nav_order[pos]
        unseen = np.flatnonzero(~self._explored.ravel()[cells])
        total = unseen.size
        for s in range(4):
            nav[5 + 3 * s] = 0.0  # empty_%; (x_free, y_free) stay at the -1 sentinel
        if total:
            bounds = unseen.searchsorted(self._nav_starts[pos]).tolist()  # sector boundaries inside `unseen`
            for s in range(4):
                count = bounds[s + 1] - bounds[s]
                if count == 0:
                    continue
                best = int(cells[unseen[bounds[s]]])
                nav[5 + 3 * s] = count / total
                nav[6 + 3 * s] = (x - best % n) / n
                nav[7 + 3 * s] = (y - best // n) / n
        return {"navigator": nav, "perception": perception, "memory": self._memory.copy()}

    def _get_info(self) -> dict[str, Any]:
        info = {"env_version": ENV_VERSION}
        gx, gy = self._goal
        info["explored_fraction"] = float(self._explored.mean())
        info["goal_discovered"] = bool(self._explored[gy, gx])
        info["is_success"] = self._agent == self._goal
        return info

    # ------------------------------------------------------------------ reset / state

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)  # seeds self.np_random; use it for ALL randomness
        self._steps = 0
        self._episode_return = 0.0
        if options is not None and "state" in options:
            # Test-only hook (see ENV_SPEC.md section 8): never use it in training code.
            self._set_state(options["state"])
        elif self._layout_pool:
            obstacles, agent, goal = self._layout_pool[int(self.np_random.integers(len(self._layout_pool)))]
            self._set_state({"obstacles": obstacles, "agent": agent, "goal": goal})
        else:
            self._set_state({})
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), self._get_info()

    def _sample_layout_pool(self, size: int, seed: int) -> list[tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int]]]:
        """Sample `size` layouts (obstacles, agent, goal) with the reset sampler, driven by a dedicated generator.

        `self.np_random` is swapped for `default_rng(seed)` so the pool uses exactly the same procedure as a fresh
        reset, then restored (it is re-seeded by `reset` anyway). Must run after all the state buffers exist.
        """
        if size == 0:
            return []
        saved = self._np_random
        self._np_random = np.random.default_rng(seed)
        try:
            pool = []
            for _ in range(size):
                self._set_state({})
                ys, xs = np.nonzero(self._obstacles)
                pool.append(([(int(x), int(y)) for x, y in zip(xs, ys)], self._agent, self._goal))
        finally:
            self._np_random = saved
        return pool

    def _cell(self, value: Any, name: str) -> tuple[int, int]:
        try:
            x, y = value
        except (TypeError, ValueError):
            raise ValueError(f"{name}: expected an (x, y) pair, got {value!r}") from None
        for v in (x, y):
            if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)):
                raise ValueError(f"{name}: coordinates must be ints, got {value!r}")
        if not (0 <= x < self.N and 0 <= y < self.N):
            raise ValueError(f"{name}: cell {value!r} is outside the grid")
        return int(x), int(y)

    def _cells(self, values: Any, name: str) -> list[tuple[int, int]]:
        try:
            items = list(values)
        except TypeError:
            raise ValueError(f"{name}: expected an iterable of cells") from None
        return [self._cell(v, name) for v in items]

    def _sample_cell(self, blocked: set[tuple[int, int]], anchor: tuple[int, int] | None = None):
        n = self.N
        ids = np.arange(n * n)
        mask = np.ones(n * n, dtype=bool)
        for bx, by in blocked:
            mask[by * n + bx] = False
        if anchor is not None:
            ax, ay = anchor
            mask &= np.maximum(np.abs(ids % n - ax), np.abs(ids // n - ay)) >= MIN_START_GOAL_DIST
        candidates = np.flatnonzero(mask)
        if candidates.size == 0:
            raise ValueError("no free cell available to sample")
        c = int(self.np_random.choice(candidates))
        return c % n, c // n

    def _set_state(self, state: dict[str, Any]) -> None:
        """Force the internal state to a valid configuration. Keys: see ENV_SPEC.md section 8.

        An empty dict samples a fresh layout. Unforced parts are sampled with self.np_random.
        """
        if not isinstance(state, dict):
            raise ValueError("state must be a dict")
        unknown = set(state) - STATE_KEYS
        if unknown:
            raise ValueError(f"unknown state keys: {sorted(unknown)}")
        n = self.N

        obstacles = None
        if "obstacles" in state:
            cells = self._cells(state["obstacles"], "obstacles")
            if len(set(cells)) != len(cells):
                raise ValueError("obstacles: duplicate cells")
            if len(cells) != self.n_obstacles:
                raise ValueError(f"obstacles: expected {self.n_obstacles} cells, got {len(cells)}")
            obstacles = set(cells)
        agent = self._cell(state["agent"], "agent") if "agent" in state else None
        goal = self._cell(state["goal"], "goal") if "goal" in state else None
        explored_cells = self._cells(state["explored"], "explored") if "explored" in state else []
        steps = 0
        if "steps" in state:
            steps = state["steps"]
            if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, (int, np.integer)):
                raise ValueError("steps: expected an int")
            if not 0 <= steps < self.max_steps:
                raise ValueError(f"steps: must be in [0, {self.max_steps})")
            steps = int(steps)

        blocked = set(obstacles) if obstacles is not None else set()
        for name, cell in (("agent", agent), ("goal", goal)):
            if cell is not None and cell in blocked:
                raise ValueError(f"{name} lies on an obstacle")
        if agent is not None and agent == goal:
            raise ValueError("agent and goal must be different cells")

        if agent is None and goal is None:
            # The grid centre has no cell at distance >= MIN_START_GOAL_DIST, so a start sampled
            # there is rejected; the retry makes the start uniform over cells admitting a goal.
            for _ in range(MAX_LAYOUT_TRIES):
                agent = self._sample_cell(blocked)
                try:
                    goal = self._sample_cell(blocked | {agent}, anchor=agent)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError("could not sample a start-goal pair")
        elif agent is None:
            agent = self._sample_cell(blocked | {goal})
        elif goal is None:
            goal = self._sample_cell(blocked | {agent})

        if obstacles is None:
            mask = np.ones(n * n, dtype=bool)
            for cx, cy in (agent, goal):
                mask[cy * n + cx] = False
            chosen = self.np_random.choice(np.flatnonzero(mask), size=self.n_obstacles, replace=False)
            obstacles = {(int(c) % n, int(c) // n) for c in chosen}

        grid = np.zeros((n, n), dtype=bool)
        for ox, oy in obstacles:
            grid[oy, ox] = True
        explored = np.zeros((n, n), dtype=bool)
        for ex, ey in explored_cells:
            explored[ey, ex] = True

        self._obstacles = grid
        h = self._half
        self._padded = np.ones((n + 2 * h, n + 2 * h), dtype=np.float32)
        self._padded[h:-h, h:-h] = grid
        self._agent, self._goal = agent, goal
        self._explored = explored
        self._steps = steps
        self._trail.clear()
        self._memory[:] = 0.0
        self._pos_history.clear()
        self._pos_history.append(agent)
        self._mark_window()

    def _push_memory(self, action: int) -> None:
        """Shift the memory rows and write the last one for the move just made (blocked moves too)."""
        self._pos_history.append(self._agent)
        ref = self._pos_history[0]  # position memory_m moves earlier, or the initial one
        self._memory[:-1] = self._memory[1:]
        row = self._memory[-1]
        row[:] = 0.0
        row[action] = 1.0
        row[8] = (self._agent[0] - ref[0]) / self.memory_m
        row[9] = (self._agent[1] - ref[1]) / self.memory_m

    def _mark_window(self) -> int:
        """Mark the perception window around the agent as explored; return the number of new cells."""
        x, y = self._agent
        h = self._half
        region = self._explored[max(y - h, 0) : y + h + 1, max(x - h, 0) : x + h + 1]
        new_cells = int((~region).sum())
        region[:] = True  # view into self._explored
        return new_cells

    # ------------------------------------------------------------------ dynamics

    def step(self, action):
        assert self.action_space.contains(action), f"invalid action {action!r}"
        self._steps += 1
        # R1, R2: king move; stay if the destination is outside the grid or an obstacle
        dx, dy = MOVES[int(action)]
        nx, ny = self._agent[0] + dx, self._agent[1] + dy
        gx, gy = self._goal
        dist_before = max(abs(self._agent[0] - gx), abs(self._agent[1] - gy))  # Chebyshev
        blocked = not (0 <= nx < self.N and 0 <= ny < self.N and not self._obstacles[ny, nx])
        if not blocked:
            self._trail.append(self._agent)  # only real moves are recorded (render trail)
            self._agent = (nx, ny)
        self._push_memory(int(action))  # R13-R15
        new_cells = self._mark_window()

        terminated = self._agent == self._goal  # R5
        reward = (
            self.coeff_expl * new_cells * self.goal_reward / self.N**2
            - self.coeff_step * self.goal_reward / self.max_steps
        )
        if blocked:  # R11
            reward -= self.bump_penalty
        if self._explored[gy, gx]:  # R12: shaping only once the goal is discovered
            dist_after = max(abs(self._agent[0] - gx), abs(self._agent[1] - gy))
            reward += self.shaping_reward * float(np.sign(dist_before - dist_after))
        if terminated:
            reward += self.goal_reward
        self._episode_return += reward
        truncated = self._steps >= self.max_steps and not terminated  # R6
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), float(reward), bool(terminated), bool(truncated), self._get_info()

    def render(self):
        if self.render_mode != "human":
            return None
        import pygame

        n, px = self.N, CELL_PX
        if self._window is None:
            pygame.init()
            self._window = pygame.display.set_mode((n * px, n * px + BAR_PX))
            self._clock = pygame.time.Clock()
            self._font = pygame.font.Font(None, 24)
        win = self._window
        pygame.display.set_caption(f"step {self._steps}/{self.max_steps}")

        ax, ay = self._agent
        ys, xs = np.mgrid[0:n, 0:n]
        sector = _sector_of(xs - ax, ys - ay)
        win.fill(GRID_COLOR)
        for y in range(n):
            for x in range(n):
                if self._obstacles[y, x]:
                    color = OBSTACLE_COLOR
                else:
                    color = _tint(SECTOR_COLORS[sector[y, x]], bool(self._explored[y, x]))
                if (x, y) == self._goal:
                    color = GOAL_COLOR
                elif (x, y) == self._agent:
                    color = AGENT_CELL_COLOR
                win.fill(color, (x * px + 1, y * px + 1, px - 1, px - 1))

        # perception window outline, clipped to the grid
        h, w = self._half, self.window
        outline = pygame.Rect((ax - h) * px, (ay - h) * px, w * px, w * px)
        pygame.draw.rect(win, (255, 255, 255), outline.clip(pygame.Rect(0, 0, n * px, n * px)), width=3)

        # trail: oldest first so the most recent dot ends up on top; alpha grows with recency
        dot = pygame.Surface((px, px), pygame.SRCALPHA)
        for k, (tx, ty) in enumerate(self._trail):
            alpha = int(40 + (255 - 40) * (k + 1) / TRAIL_LEN)
            dot.fill((0, 0, 0, 0))
            pygame.draw.circle(dot, (30, 30, 30, alpha), (px // 2, px // 2), int(px * 0.24))
            pygame.draw.circle(dot, (255, 255, 255, alpha), (px // 2, px // 2), int(px * 0.24) - 2)
            win.blit(dot, (tx * px, ty * px))

        pygame.draw.circle(win, (255, 255, 255), (ax * px + px // 2, ay * px + px // 2), int(px * 0.38))
        pygame.draw.circle(win, (20, 20, 20), (ax * px + px // 2, ay * px + px // 2), int(px * 0.38), width=3)

        # status bar: step counter and cumulative reward of the episode
        win.fill(BAR_COLOR, (0, n * px, n * px, BAR_PX))
        text = f"step {self._steps}/{self.max_steps}   return {self._episode_return:+.3f}"
        label = self._font.render(text, True, TEXT_COLOR)
        win.blit(label, (8, n * px + (BAR_PX - label.get_height()) // 2))

        pygame.event.pump()
        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])
        return None

    def close(self):
        if self._window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self._window = None
            self._clock = None
            self._font = None
