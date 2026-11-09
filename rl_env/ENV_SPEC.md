# Environment Spec: Partially Observable King Grid

Source of truth for `env.py`. Update this file first, then the code and tests.

- **Current version:** 13 (`ENV_VERSION` in `env.py`, tag `env-v13`)
- **Gymnasium API:** single-agent, `reset` -> `(obs, info)`, `step` -> `(obs, reward, terminated, truncated, info)`

## 1. Goal
Reach the goal cell of an `N x N` grid (`N = 15`) containing `n_obstacles` (`60`) obstacle cells. The agent moves like a chess king and only sees a local `window x window` window (constructor parameter `window`, default 3) plus a compact "navigator" summary, so the world is partially observable. One episode is one layout: it ends when the agent enters the goal cell or after `max_steps` steps.

## 2. State and dynamics
Coordinates: `x` = column, `y` = row, `0 <= x, y < N`, `y` grows downward, "north" means `y - 1`.

State:
- `obstacles`: set of `n_obstacles` distinct cells.
- `agent`: current cell. `goal`: goal cell.
- `explored`: boolean `N x N` mask of cells that have ever been inside the agent's `window x window` perception window (only in-grid cells). The goal is *discovered* iff `explored[goal]`.
- `steps`: step counter.

Layout (obstacles, start, goal) is re-sampled at every `reset()` and is constant within the episode. Sampling, all via `self.np_random`: start uniform over the cells that admit a goal at Chebyshev distance >= 8 (every cell but the grid centre for `N = 15`); goal uniform over cells with Chebyshev distance >= 8 from start; obstacles uniform without replacement over the remaining cells. No reachability check is performed (BFS was deliberately dropped to keep `reset` cheap): measured on 200k layouts from this sampler with `n_obstacles = 60`, about 0.449% (897) had an unreachable goal (0.038% with the previous default of 35); those episodes end by truncation. Known limitation.

Layout pool (constructor parameters `layout_pool` = K >= 0, default 0, and `layout_pool_seed` >= 0, default 0; negative values raise `ValueError`): with K > 0 the constructor samples K layouts with exactly the sampler above, driven by a dedicated `np.random.default_rng(layout_pool_seed)` instead of `self.np_random`; every `reset` without the state hook then uses one of these K layouts, chosen uniformly with `self.np_random`. The pool is fixed for the life of the env and does not depend on `reset(seed=...)`. With K = 0 every reset samples a fresh layout as above. A forced state (section 8) ignores the pool.

Dynamics: the action selects one of 8 king directions. If the destination is inside the grid and not an obstacle, the agent moves there; otherwise it stays (the step still counts). Only the destination cell matters, so diagonal moves between two diagonally adjacent obstacles are allowed. After moving, the `window x window` window around the agent is added to `explored`. At `reset` the initial window is already explored.

## 3. Rules
Each rule has an id and one test named `test_R<k>_<slug>` in `tests/test_env.py`.

| Id | Rule | Test |
|----|------|------|
| R1 | Each action 0..7 moves the agent one cell in direction N, NE, E, SE, S, SW, W, NW if the destination is free | `test_R1_king_move_directions` |
| R2 | A move whose destination is outside the grid or an obstacle leaves the position unchanged | `test_R2_blocked_move_keeps_position` |
| R3 | A layout has exactly `n_obstacles` obstacles; start, goal and obstacles are pairwise distinct and constant within an episode | `test_R3_layout_distinct_and_constant` |
| R4 | Same seed gives the same layout | `test_R4_same_seed_same_layout` |
| R5 | Entering the goal cell sets `terminated=True` | `test_R5_goal_terminates` |
| R6 | After `max_steps` steps without reaching the goal `truncated=True`; reaching the goal on the last step is `terminated`, not `truncated` | `test_R6_truncation_at_max_steps` |
| R7 | `perception` is the `window x window` window centred on the agent: free = 0, obstacle = 1, outside the grid = 1 | `test_R7_perception_window` |
| R8 | The goal entry of `navigator` is (-1, -1) until a goal cell enters the window; afterwards it is the relative normalised position, recomputed every step | `test_R8_goal_discovery` |
| R9 | `navigator` sector triples (`empty_%`, nearest unexplored cell) follow the sector, diagonal, tie-break and sentinel conventions of section 5 | `test_R9_navigator_sectors` |
| R10 | Reward = exploration term - step penalty (+ `goal_reward` on entering the goal) + the terms of R11 and R12; `reset` gives no reward | `test_R10_reward_terms` |
| R11 | A blocked move (destination outside the grid or an obstacle) adds `-bump_penalty` to the reward (default 0: off) | `test_R11_bump_penalty` |
| R12 | With `shaping_reward` > 0 (default 0: off), once the goal is discovered, a move that reduces the Chebyshev distance to the goal adds `+shaping_reward`, one that increases it adds `-shaping_reward`, an unchanged distance (including blocked moves) adds 0; no shaping before discovery | `test_R12_goal_distance_shaping` |
| R13 | `memory` has one row per past move, oldest first: at every step the rows shift up by one and the new last row holds the one-hot action; rows of moves not yet made are 0 | `test_R13_memory_rows_shift` |
| R14 | A blocked move (outside the grid or an obstacle) still writes a memory row: one-hot action, displacement computed with the unchanged position | `test_R14_memory_bump_counts_as_move` |
| R15 | The displacement part of a row is `(pos_after - pos_(t-m)) / memory_m` in `(x, y)`; if fewer than `memory_m` moves have been made the reference is the initial position | `test_R15_memory_displacement_reference` |
| R16 | `reset` zeroes `memory` (also with a forced state); the returned observation is a copy that later steps do not modify | `test_R16_memory_reset_and_copy` |
| R17 | `navigator[0]` is `steps / max_steps`, with `steps` the number of steps already made: 0 after `reset` (`steps / max_steps` with a forced `steps`), +`1/max_steps` per step | `test_R17_navigator_time_fraction` |
| R18 | With `layout_pool` = K > 0, every `reset` without the state hook uses one of K layouts fixed at construction by `layout_pool_seed` (same pool seed gives the same pool, independent of the reset seed; every member can be drawn); each pool layout satisfies the constraints of R3 and the start-goal distance >= 8 | `test_R18_layout_pool` |

## 4. Actions
- **Space:** `Discrete(8)`. An action outside the space fails an assertion.

| Action | Direction | (dx, dy) |
|--------|-----------|----------|
| 0 | N  | (0, -1) |
| 1 | NE | (1, -1) |
| 2 | E  | (1, 0) |
| 3 | SE | (1, 1) |
| 4 | S  | (0, 1) |
| 5 | SW | (-1, 1) |
| 6 | W  | (-1, 0) |
| 7 | NW | (-1, -1) |

## 5. Observation
- **Space:** `Dict({"navigator": Box(-1, 1, (17,), float32), "perception": Box(0, 1, (1, window, window), float32), "memory": Box(-1, 1, (memory_k, 10), float32)})`

`perception`: shape `(1, window, window)`, window centred on the agent, indexed `[0, row, col]`. Free cell = 0, obstacle = 1, outside the grid = 1. Constructor parameter `window` (default 3): odd and `3 <= window <= 2 * 8 - 1 = 15` (so that half the window stays below the minimum start-goal distance of 8), otherwise `ValueError`.

`navigator`: 17 floats, `n = N = 15`:
- `[0]` elapsed fraction `steps / max_steps` in [0, 1], `steps` being the number of steps already made (R17). 0 after `reset` (unless `steps` is forced by the state hook); it reaches 1 only on the last step, when the episode has already ended.
- `[1:3]` agent `(x / n, y / n)`, in [0, 1).
- `[3:5]` goal `((x_a - x_g) / n, (y_a - y_g) / n)` once discovered (recomputed every step); `(-1, -1)` before discovery. Real relative values satisfy |v| <= 14/15, so the sentinel is unambiguous. Discovery happens when a goal cell enters the perception window (not when the agent steps on it); at reset the goal is never in the window with the default layout sampler (distance >= 8 > window // 2).
- `[5:17]` four triples `(empty_%, x_free, y_free)` for sectors N, E, S, W, in this order.

Sectors: the two diagonals through the agent split the plane into four wedges. With `dx = x - x_a`, `dy = y - y_a`, a cell with `|dx| > |dy|` is E (`dx > 0`) or W (`dx < 0`); `|dy| > |dx|` is N (`dy < 0`) or S (`dy > 0`). Cells exactly on a diagonal (`|dx| == |dy| != 0`) are assigned as: NW -> N, NE -> E, SE -> S, SW -> W.

An *unexplored* cell is an in-grid cell with `explored == False` (it may or may not be an obstacle; this is not known yet).
- `empty_%` = unexplored cells in the sector / unexplored cells in the world (0 if the world is fully explored).
- `(x_free, y_free)` = `((x_a - x) / n, (y_a - y) / n)` of the unexplored cell of the sector nearest to the agent in Chebyshev distance; ties broken by smaller row (`y`), then smaller column (`x`).
- A sector with no unexplored cell gives `(0, -1, -1)`.

`memory`: shape `(memory_k, 10)` (constructor parameters `memory_k` = 10 and `memory_m` = 10, both >= 1, otherwise `ValueError`). Same observable as the one fed to the GRU in QML_Thesis. One row per past move, oldest in row 0 and most recent in the last row; at every step the rows shift up by one and only the last row is written. Row layout:
- `[0:8]` one-hot of the action, ordered as the table of section 4.
- `[8:10]` displacement `(dx, dy) = (pos_after - pos_ref) / memory_m` in `(x, y)` grid units, where `pos_after` is the agent position after the move and `pos_ref` the position `memory_m` moves earlier (or the initial position if fewer moves have been made). Values lie in [-1, 1]. Note the sign: it is the agent's own motion, opposite to the `navigator` goal entry, which is `agent - goal`.
- A blocked move counts as a move: the row is written with the unchanged position (R14).
- Rows of moves not yet made are all 0. `reset` zeroes the whole array; the returned array is a copy.
- The position history used for `pos_ref` starts at the agent cell at `reset`, so the state hook key `steps` does not affect it.

Hidden from the agent: the full layout and the exploration mask (only summarised via the navigator).

## 6. Reward
Constructor hyperparameters: `goal_reward` (GRW) = 5.0, `coeff_expl` = 0.5, `coeff_step` = 1.5, `bump_penalty` = 0.0, `shaping_reward` = 0.0, `N` = 15, `max_steps` = 300.

Per step, all terms add up:
`reward = coeff_expl * new_cells * GRW / N^2 - coeff_step * GRW / max_steps - bump_penalty * blocked + shaping_reward * sign(d_before - d_after) * discovered`, and `+ GRW` on the step in which the agent enters the goal (all terms apply on that step).

- `new_cells` = in-grid cells that enter `explored` during this step. `reset` returns no reward, so the initial window (up to `window^2` cells) is never paid: the exploration total is slightly below `coeff_expl * GRW`.
- `blocked` = 1 if the move destination is outside the grid or an obstacle (R2), else 0.
- `d` = Chebyshev distance between agent and goal; `d_before`/`d_after` are measured before and after the move. `discovered` = `explored[goal]` after the move (so shaping already applies on the step in which the goal enters the window). A blocked move leaves `d` unchanged, hence no shaping. On the step that reaches the goal `d_after = 0`, so the shaping term applies as well.

With defaults the reward is the one of QML_Thesis: +0.0111 per new cell, -0.025 per step, timeout without exploring -7.5, goal +5; the `bump_penalty` and `shaping_reward` terms are off (0). They stay available as opt-in: both are absolute and do not scale with GRW (v5 used 0.025 and 0.2). Exploration is a finite bounded term (no farmable loop). Each king move changes `d` by -1, 0 or +1, so the shaping, when enabled, telescopes: its total after discovery is `shaping_reward * (d_at_discovery_step_start - d_end)`, and going back and forth earns nothing. The bump penalty can only lower the return.

## 7. Termination and truncation
- **Terminated when:** the agent is on the goal cell (also at step `max_steps`).
- **Truncated when:** `max_steps` = 300 reached and not terminated

## 8. Seeding and reproducibility
- All randomness through `self.np_random` (seeded by `reset(seed=...)`), with one exception: the layout pool (section 2) is sampled once at construction from `np.random.default_rng(layout_pool_seed)`. Which pool member a reset uses goes through `self.np_random`.
- The only randomness is the layout sampled (or picked from the pool) at `reset`; dynamics are deterministic. Same seed gives the same layout.
- **Test-only state hook:** `reset(options={"state": {...}})` forces the internal state. It exists only so rule tests can reach a precise situation; training code must not use it. Accepted keys (validated, invalid values raise `ValueError`; unknown keys raise `ValueError`; cells are `(x, y)` pairs of ints inside the grid):
  - `obstacles`: exactly `n_obstacles` distinct cells. For small maps in tests build the env with a smaller `n_obstacles`.
  - `agent`: not on an obstacle.
  - `goal`: not on an obstacle and different from `agent`.
  - `explored`: iterable of cells; the window around `agent` is always added.
  - `steps`: int in `[0, max_steps)`.
  - `memory` is always zero after a forced reset (it is not a hook key); the position history restarts at the forced `agent`.
  - Unforced parts are sampled as usual (agent/goal/obstacles avoid the forced cells). The minimum start-goal distance of 8 is applied only when neither `agent` nor `goal` is forced; forced obstacles are never moved.

## 9. Rendering
- **Modes:** `human` only (`render_modes = ["human"]`, `render_fps = 4`). Requires `pygame`, imported lazily: the env works without it unless `render_mode="human"`. In `human` mode `reset` and `step` call `render()` themselves. Rendering never changes state, observations or rewards.
- **Debug view** (shows ground truth the agent does not see), `N x N` cells of 40 px:
  - Sectors: the four wedges of section 5, recomputed around the agent at every frame with the same diagonal convention. Colours N blue, E green, S orange, W purple; *explored* cells in a pale tint of the sector colour, *unexplored* cells in a darker shade of it.
  - Obstacles black (always visible), goal yellow square (always visible), agent a white disc with a dark outline on a neutral cell.
  - White outline around the `window x window` perception window (clipped to the grid).
  - Trail: the last 10 cells the agent left, as dots inside the cell with opacity growing from the oldest (faint) to the most recent (opaque); a dot disappears once it falls out of the 10-move window. A position is recorded only when the agent actually changes cell (blocked moves are not recorded); the start cell is the first entry. The trail is cleared at every `reset`. Revisited cells show several overlapping dots.
  - Window caption: `step / max_steps`.
  - Status bar (32 px) below the grid, window `N*40 x (N*40 + 32)`: `step <t>/<max_steps>   return <R>`, where `R` is the cumulative reward returned by `step` since the last `reset` (exploration, step penalty and goal reward; 3 decimals, signed). The return is an internal counter used only for display: it is not part of the observation or `info`, and is reset to 0 at every `reset` (also with a forced state).

## 10. Info dict
- `env_version`: int, always present
- `explored_fraction`: float, explored cells / `N^2`
- `goal_discovered`: bool
- `is_success`: bool, True only if the agent is on the goal cell

## Changelog
Newest first. Semantic changes (reward, spaces, rules, termination, seeding) bump the version and get a tag.

| Version | Type | Change | Why |
|---------|------|--------|-----|
| 13 | semantic | Add constructor parameters `layout_pool` (K, default 0 = off) and `layout_pool_seed`: K layouts sampled once at construction with a dedicated generator, each reset picks one uniformly (R18). Default behaviour identical to v12 | Generalization pilot: train on K fixed layouts, test on unseen ones |
| 12 | semantic | Add constructor parameter `window` (default 3, odd, 3..15) for the side of the perception window; `perception` shape, `explored`, goal discovery, exploration reward and render outline follow it. Default behaviour identical to v11 | Generalization pilot: `window=5, n_obstacles=30` reproduces v8 (5x5 window, 30 obstacles) |
| 11 | semantic | Raise `n_obstacles` default from 35 to 60 (unreachable-goal rate measured on 200k layouts: 0.038% -> 0.449%) | User request: make navigation harder with the 3x3 window |
| 10 | semantic | Shrink the perception window from 5x5 to 3x3 (`perception` `(1, 5, 5)` -> `(1, 3, 3)`); `explored`, goal discovery and the exploration reward follow the smaller window (at most 3 new cells per straight move, 5 per diagonal). Navigator layout, memory and reward formula unchanged | User request (motivation not stated): stronger partial observability |
| 9 | semantic | Raise `n_obstacles` default from 30 to 35 (unreachable-goal rate measured on 200k layouts: 0.023% -> 0.038%) | User request (motivation not stated): denser obstacle layout |
| 8 | non-semantic | Precompute per-position navigator tables (cells grouped by sector, ordered by Chebyshev distance, row, column) so `_get_obs` no longer recomputes sectors and sort keys every step; observations identical | Speed: `_get_obs` was ~73% of a ~37 us step, the main cost of rollout collection |
| 8 | semantic | Add the elapsed fraction `steps / max_steps` as `navigator[0]` (`navigator` 16 -> 17 floats; agent, goal and sector entries shift by 1); new rule R17. Reward, dynamics, `perception`, `memory` unchanged | Give the policy a time-left signal, since the step penalty and the truncation depend on the remaining time |
| 7 | semantic | Align the reward to QML_Thesis: raise `coeff_step` default from 0.5 to 1.5 and turn off `bump_penalty` (0.025 -> 0.0) and `shaping_reward` (0.2 -> 0.0); the two terms and rules R11/R12 stay available as opt-in | Train with the same reward as the thesis experiments so results are comparable |
| 6 | semantic | Add the `memory` observation `(memory_k=10, 10)`: last `memory_k` moves as action one-hot (8) + displacement `(pos - pos_(t-m)) / memory_m` (2), with `memory_m=10`; new constructor parameters `memory_k`, `memory_m`. Reward, dynamics, `navigator`, `perception` unchanged | Give the agent the same move-history observable used with the GRU in QML_Thesis |
| 5 | semantic | Lower `goal_reward` (GRW) default from 10.0 to 5.0, lower `bump_penalty` from 0.25 to 0.025 and raise `shaping_reward` from 0.1 to 0.2 | Make goal-distance shaping dominant over the other terms so the perception branch gets a stronger learning signal |
| 4 | semantic | Raise `goal_reward` (GRW) default from 5.0 to 10.0 (exploration and step terms scale with it; `bump_penalty` and `shaping_reward` stay absolute) | Overall episode returns in v3 are too low |
| 3 | semantic | Lower `coeff_step` default from 1.0 to 0.5; add `bump_penalty` (-0.25 on blocked moves, border included) and `shaping_reward` (+/-0.1 per step that reduces/increases the Chebyshev distance to the goal, once discovered) | The perception branch of the encoder is nearly ignored in training; bump penalty and goal-distance shaping give it a learning signal |
| 2 | semantic | Raise `coeff_step` default from 0.5 to 1.0 | With 0.5 the agent explores a lot but does not head to the goal; a stronger step penalty favours reaching it |
| 1 | non-semantic | Show cumulative episode reward in a status bar of the `human` render | Follow the return while watching an episode; display only, no change to observations, rewards or `info` |
| 1 | non-semantic | Add `human` render (sector colours, perception window, 10-move trail) | Visual debugging of exploration and loops; no change to observations, actions, rewards or termination |
| 1 | initial | Initial version | Partially observable 15x15 king-move navigation task with exploration shaping |
