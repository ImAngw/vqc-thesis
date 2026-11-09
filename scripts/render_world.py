"""Snapshot of the environment for the thesis: a layout, an agent part-way through an episode and what it sees.

    python scripts/render_world.py --out thesis/figures/mondo.pdf

The agent is a scripted explorer (it walks towards the nearest unexplored cell, then towards the goal once it is
discovered), not a trained policy: the figure illustrates the world and the observation, not a result. The drawing
follows the pygame renderer of rl_env/env.py: cells coloured by the sector (N, E, S, W) they fall in relative to the
agent, dark if unexplored and pale if explored, obstacles in black, the goal in yellow, the last 10 cells of the trail
and the outline of the perception window. Needs matplotlib (not in requirements.txt for the library, only for figures).
"""

import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl_env.env import MOVES, SECTOR_COLORS, TRAIL_LEN, CustomEnv, _sector_of, _tint  # noqa: E402

SECTOR_NAMES = ("N", "E", "S", "W")


def next_move(env: CustomEnv) -> int:
    """First step of a shortest path (8-connected) to the goal if discovered, else to the nearest unexplored cell."""
    n, goal, start = env.N, env._goal, env._agent
    target = (lambda c: c == goal) if env._explored[goal[1], goal[0]] else (lambda c: not env._explored[c[1], c[0]])
    seen = {start: None}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        if cell != start and target(cell):
            while seen[cell] != start:
                cell = seen[cell]
            return MOVES.index((cell[0] - start[0], cell[1] - start[1]))
        for dx, dy in MOVES:
            nxt = (cell[0] + dx, cell[1] + dy)
            if 0 <= nxt[0] < n and 0 <= nxt[1] < n and not env._obstacles[nxt[1], nxt[0]] and nxt not in seen:
                seen[nxt] = cell
                queue.append(nxt)
    return 0


def draw(env: CustomEnv, out: Path, size: float) -> None:
    import matplotlib
    matplotlib.use("pdf")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    matplotlib.rcParams["pdf.fonttype"] = 42

    n, (ax_, ay_) = env.N, env._agent
    ys, xs = np.mgrid[0:n, 0:n]
    sector = _sector_of(xs - ax_, ys - ay_)
    fig, ax = plt.subplots(figsize=(size, size))
    for y in range(n):
        for x in range(n):
            if env._obstacles[y, x]:
                color = (0, 0, 0)
            else:
                color = _tint(SECTOR_COLORS[sector[y, x]], bool(env._explored[y, x]))
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor=tuple(c / 255 for c in color), edgecolor="#282828", linewidth=0.4))
    gx, gy = env._goal
    ax.add_patch(Rectangle((gx, gy), 1, 1, facecolor="#fadc28", edgecolor="#282828", linewidth=0.4))
    ax.plot(gx + 0.5, gy + 0.5, marker="*", color="black", markersize=size * 3.2, zorder=4)
    h, w = env._half, env.window
    ax.add_patch(Rectangle((ax_ - h, ay_ - h), w, w, fill=False, edgecolor="white", linewidth=2.2, zorder=3))
    trail = list(env._trail)
    for k, (tx, ty) in enumerate(trail):
        alpha = 0.15 + 0.85 * (k + 1) / TRAIL_LEN
        ax.add_patch(Circle((tx + 0.5, ty + 0.5), 0.22, facecolor="white", edgecolor="#1e1e1e", linewidth=0.6, alpha=alpha, zorder=5))
    ax.add_patch(Circle((ax_ + 0.5, ay_ + 0.5), 0.38, facecolor="white", edgecolor="#141414", linewidth=1.4, zorder=6))
    # sector names on the border, at the middle of each side
    for name, (px, py) in zip(SECTOR_NAMES, ((n / 2, -0.55), (n + 0.55, n / 2), (n / 2, n + 0.55), (-0.55, n / 2))):
        ax.text(px, py, name, ha="center", va="center", fontsize=size * 1.8)
    ax.set_xlim(-0.1, n + 0.1)
    ax.set_ylim(n + 0.1, -0.1)
    ax.set_aspect("equal")
    ax.axis("off")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.12)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="thesis/figures/mondo.pdf")
    ap.add_argument("--seed", type=int, default=3, help="layout seed")
    ap.add_argument("--steps", type=int, default=12, help="steps of the scripted explorer before the snapshot")
    ap.add_argument("--size", type=float, default=3.4, help="figure size in inches")
    args = ap.parse_args()
    env = CustomEnv(N=15, n_obstacles=30, window=5, max_steps=300)
    env.reset(seed=args.seed)
    for _ in range(args.steps):
        if env._agent == env._goal:
            break
        env.step(next_move(env))
    print(f"step {env._steps}, explored {env._explored.mean():.0%}, goal discovered {bool(env._explored[env._goal[1], env._goal[0]])}", file=sys.stderr)
    draw(env, Path(args.out), args.size)


if __name__ == "__main__":
    main()
