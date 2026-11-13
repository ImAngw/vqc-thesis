# Spec: Generalization pilot (fixed pool of K training layouts)

## Objective
Decide within a few days whether the thesis direction of plan section 12 is viable: train on a fixed pool of K
layouts, evaluate on unseen layouts, and look for a K where the VQC head beats the MLP on **absolute test return**.

The pilot answers two questions:
1. **Overfitting:** with a small K, does the MLP do clearly better on its training layouts than on unseen ones?
   If no head overfits at any K, the regime cannot favour the VQC for the reason X of section 12.
2. **First signal:** at some K, is J's test return at or above the MLP's?

A layout is the triple (obstacles, start, goal). Today `CustomEnv` re-samples it at every reset (K = inf).

Environment: the v8 configuration (5x5 window, 30 obstacles, CNN perception), where section 11 measured J 4.02 +-
0.28 and h128 5.11 +- 0.15 (10 seeds, `detach_critic=1`, 7.5k updates). Chosen by the user on 2026-10-14 over v11,
where every head is still low at 7.5k (3 seeds: h128 2.44, h5 1.23, J 0.72) and a 3x3 window says little about which
layout the agent is in, which makes overfitting less likely.

## Tech Stack
Python 3.13, torch, numpy, gymnasium, pytest. No new dependencies.

## Design
- **Env (`rl_env/env.py`, via `/rl-env:design-env`, ENV_SPEC first; `window` = v12, layout pool = v13):** three new constructor kwargs.
  - `window: int = 3` replaces the module constant `WINDOW` (odd, >= 3). It sets the observation box, the padding,
    the explored-marking region (so it also affects the exploration reward) and the render rectangle.
    `window=5, n_obstacles=30` must reproduce v8 bit-for-bit: obs, reward, terminated and truncated identical to the
    env at commit `498b6f2` for 20 seeds x 300 random-action steps. This is checked once with a scratch script
    against `git show 498b6f2:rl_env/env.py`, and the outcome goes into the plan.
  - `layout_pool: int = 0` and `layout_pool_seed: int = 0`. With `layout_pool = K > 0`, `__init__` samples K layouts
    with the same procedure as today, but from a dedicated `np.random.default_rng(layout_pool_seed)` instead of
    `self.np_random`. Each `reset` without `options` then picks one of them uniformly with `self.np_random`.
  - With all defaults, behaviour is bit-identical to v11.
- **Config (`rl_train/config.py`):**
  - `layout_pool: int = 0` and `layout_pool_seed: int = -1` (-1 = the run's env seed). The trainer and the adapter
    merge them into the kwargs of the **training** env only.
  - `perception: str = "linear"` (`"cnn"` = the XS encoder of section 11). The encoder window is read from
    `env_kwargs.get("window", 3)`.
  - `env_kwargs` becomes settable per run in the adapter as JSON: `--set env_kwargs='{"window": 5, "n_obstacles": 30}'`.
    Today `env_kwargs` is excluded from `--set`.
- **Eval (`rl_train/a2c.py`, `parallel_adapter.py`):**
  - The existing `greedy/*` and `sample/*` keys keep their meaning: unseen layouts (seeds `eval_seed..`, env without
    pool). Tooling, `best_metric` and the K = inf runs therefore stay comparable.
  - With K > 0, the same eval also runs on the training pool and is logged as `pool/greedy/*` and `pool/sample/*`
    (100 episodes, same seeds, so the pool member of each episode is drawn reproducibly).
- **Summary script (`scripts/summarize_gen_pilot.py`):** reads `logs/<run>/node_0.log` with C-locale parsing and
  prints, per run and per (head, K), the mean of eval rounds 41-50 of `sample/return_mean` (test) and
  `pool/sample/return_mean` (train), their gap, and paired differences between heads per seed.

## Pilot grid
- **Heads:** h128 (`mlp128`), h5 (`mlp5`, parameter-matched MLP), J (`vqc-nav-id`, `head_lr=2e-3`).
  All with `detach_critic=1` and `split_init=1` (the default).
- **K:** 5, 25, 200 and inf (no pool). For K = inf:
  - J reuses the section 11.14 runs `nav-id-hlr2e-3-detc-s{0,1,2}-7.5k-cpu-outer` (v8, split_init=1).
  - h128 and h5 are rerun with split_init=1. The section 11 runs used split_init=0 and are not comparable.
  - One control run, `gen-nav-id-kinf-s0`, uses the new code (`window=5`, `perception=cnn`) and must match 3.88.
    The old runs used uncommitted code, and the encoder default has since changed. If it does not match, stop and
    report before reading any K comparison.
- **Seeds:** 0-2, with `env_seed = seed` and `layout_pool_seed = env_seed`. For a given seed every head trains on the
  same pool; different seeds use different pools.
- **Total:** 27 runs with finite K (3 heads x 3 K x 3 seeds) + 6 with K = inf (h128, h5) + 1 J control = 34 runs.
  At ~27 updates/s, one 7.5k-update CPU run takes ~5-10 min (measured on h128 v11). The pilot takes about one
  afternoon on the mac.
- **Names:** `gen-<head>-k<K>-s<seed>-7.5k-cpu-outer`, with `<head>` in {h128, h5, nav-id} and `kinf` for no pool.

## Commands
```
Env + all tests:  .venv/bin/python -m pytest -q
Env tests only:   .venv/bin/python -m pytest rl_env/tests -q
One pilot run:    python train.py with the options of the cell (generated by scripts/cell_command.py); J adds head=vqc-nav-id, head_lr=2e-3
Summary:          .venv/bin/python scripts/summarize_gen_pilot.py 'gen-*'
```

## Project Structure
- `rl_env/env.py`, `rl_env/ENV_SPEC.md`, `rl_env/tests/test_env.py`: `window` kwarg and layout pool (v12, v13).
- `rl_train/config.py`: `layout_pool`, `layout_pool_seed`, `perception`.
- `rl_train/policy.py`: encoder built with `perception` and the env window.
- `rl_train/a2c.py`, `parallel_adapter.py`: training env with pool, eval on unseen layouts plus pool eval.
- `rl_train/tests/test_pipeline.py`: config plumbing and eval keys.
- `scripts/summarize_gen_pilot.py`: pilot table.
- `docs/design/PLAN-vqc-head.md`: results as section 12.1.

## Code Style
Match the surrounding code: English docstrings, `(x, y)` = (column, row), numpy only in the env, int instead of bool
for config flags (argparse, see `norm_advantage`). Example of the intended env shape:

```python
if layout_pool > 0:
    rng = np.random.default_rng(layout_pool_seed)
    self._layout_pool = [self._sample_layout(rng) for _ in range(layout_pool)]  # (obstacles, agent, goal)
```

## Testing Strategy
pytest, existing locations.
- **Env, regression:**
  - with default kwargs, obs/reward/terminated/truncated are identical to v11 for 20 seeds x 300 random-action steps,
    resets included;
  - `window=5, n_obstacles=30` reproduces v8 (scratch check above);
  - all existing R-tests pass unchanged;
  - an even or too-small `window` raises `ValueError`.
- **Env, pool:**
  - every reset layout belongs to the pool;
  - the pool is deterministic given `layout_pool_seed` and differs across pool seeds;
  - all K members are drawn over enough resets;
  - pool layouts respect the existing constraints (obstacle count, start-goal distance);
  - the pool does not depend on `reset(seed=...)`.
- **Unseen:** for K up to 200 and pool seeds 0-2, none of the 100 eval layouts (seeds 10000-10099) is in the pool.
- **Pipeline:**
  - `layout_pool` reaches the training env and not the test env;
  - `pool/*` keys appear only when K > 0;
  - `layout_pool_seed=-1` resolves to the env seed;
  - `perception="cnn"` with `window=5` builds the encoder of section 11;
  - `--set env_kwargs=<json>` reaches the config.
- Report the commands run and their outcome; no claim without running them.

## Boundaries
- **Always:**
  - route the env change through `/rl-env:design-env`;
  - use the same budget, flags, seeds and pools for every head;
  - report the mean of eval rounds 41-50 per seed, never the best round;
  - pass the `--show` list above.
- **Ask first:**
  - any env configuration other than `window=5, n_obstacles=30`;
  - K values or seeds beyond the grid;
  - any J hyperparameter other than `head_lr=2e-3`.
- **Never:**
  - pick checkpoints or hyperparameters on the test layouts;
  - non-Born readouts;
  - changes to the J circuit;
  - `--no-window` on the queue;
  - commits without being asked.

## Success Criteria (pilot decision rule, on means over seeds 0-2)
- [ ] Tests above pass, with commands and outcome reported.
- [ ] 34 runs finished; the summary table is written to plan section 12.1.
- [ ] **Overfitting** = h128 or h5 at K=5 has `pool` - `test` >= 1.0 return, on at least 2 of 3 seeds.
- [ ] Decision recorded in the plan:
  - **GO:** overfitting present and, at some K, J test >= h5 test and J test >= h128 test - 0.5. Next: the full study
    (more K values, 10 seeds).
  - **CHANGE REGIME:** no head overfits at any K. Back to the user (other regimes or plan B).
  - **NO-GO:** overfitting present but J is below both MLPs on test at every K. Plan B (diagnostic thesis).

## Out of Scope
- Closing the K = inf gap.
- New ansatz.
- Head-lr or entropy sweeps.
- Noise robustness.
- The full study (more K values, 10 seeds) before the GO decision.

## Open Questions
1. Layout = obstacles + start + goal, all fixed (assumed: it maximizes what can be memorized). The alternative is
   fixed obstacles with start and goal re-sampled.
2. Pool paired across heads and different across seeds (`layout_pool_seed = env_seed`): assumed.
