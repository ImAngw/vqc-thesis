# Spec: Rollout speedup

## Objective
Cut the wall-clock time of A2C training by about 2.5x without changing what the agent sees or how the env behaves, so new runs stay comparable with the existing ones.

Measured baseline (2026-10-12, 64 envs x 16 steps, CPU, 4 torch threads, eval amortized over 100 updates):

| head | ms/update | collect | update (fwd+bwd) | eval share |
|---|---|---|---|---|
| mlp16 | 116 | 52 | 38 | 22% |
| vqc-optim | 138 | 62 | 48 | 20% |

- Inside `collect()`, `env.step` takes ~78% of the time. `CustomEnv.step` costs ~37 us, and ~73% of that goes to `_get_obs` and `_sector_of`, which recompute sectors and sort keys over all unexplored cells on every step.
- QML_Thesis `GridWorld` costs ~14 us/step. It precomputes per-position cell order and sector bounds once (`NavigatorTables`).
- With `torch.set_num_threads(1)` alone, mlp16 drops to 77 ms/update and vqc-optim to 98 ms/update. The update step goes from 38 to 8.4 ms.

Two changes:
1. **Threads:** call `torch.set_num_threads(1)` in `train.py`.
2. **Navigator tables:** in `rl_env/env.py`, precompute for every agent position the other cells grouped by sector and ordered by the current key `(Chebyshev, row, column)`. `_get_obs` then only filters the unexplored cells and reads the first one of each sector. This change goes through `/rl-env`.

## Tech Stack
Python 3.13, numpy, torch, gymnasium (`SyncVectorEnv`), pytest. No new dependencies.

## Commands
```
Env tests:      .venv/bin/python -m pytest rl_env/tests -q
All tests:      .venv/bin/python -m pytest -q
Env benchmark:  .venv/bin/python <scratchpad>/bench_new.py           # us/step, CustomEnv and SyncVectorEnv
Train profile:  .venv/bin/python <scratchpad>/prof_update.py <logdir> 100 mlp16 vqc-optim
```

## Project Structure
- `train.py`: thread setting, in `main()` before the model is built.
- `rl_env/env.py`: navigator tables (module-level, cached per `N`) and the new `_get_obs`.
- `rl_env/tests/test_env.py`: equivalence test against a reference copy of the current `_get_obs`.
- `rl_env/ENV_SPEC.md`: changelog entry for a non-semantic change (`/rl-env` decides whether `ENV_VERSION` changes).

## Code Style
Match `rl_env/env.py`: English docstrings, `(x, y)` = (column, row), `_private` helpers, numpy only. Table layout follows `NavigatorTables` in QML_Thesis (`order[pos]`, `starts[pos]`), but with this env's sector and diagonal conventions (NW->N, NE->E, SE->S, SW->W) and its tie-break, not QML_Thesis's.

## Testing Strategy
- **Equivalence (new):** run the old `_get_obs` (kept in the test as a reference function) and the new one on the same states. Use at least 20 seeds x 300 random-action steps, including episode resets and the case with every cell explored. Require `np.array_equal` on `navigator`, `perception` and `memory`. Also compare full trajectories (obs, reward, terminated, truncated) between the two implementations under the same seed and actions.
- **Existing:** all R1–R17 tests in `rl_env/tests/test_env.py` must still pass unchanged.
- **Performance:** measured with the benchmark scripts. These are not assertions in the test suite.

## Boundaries
- **Always:** route every env change through `/rl-env`, run the env tests before declaring done, and report the measured numbers.
- **Ask first:** any change to obs, reward or dynamics semantics, any change to `SyncVectorEnv` usage, any new dependency.
- **Never:** change sector, diagonal or tie-break conventions; touch the VQC heads or the A2C update; commit without being asked.

## Success Criteria
- [ ] Equivalence test passes: obs bit-identical across the tested seeds and steps.
- [ ] Existing env tests pass.
- [ ] `CustomEnv.step` at or below ~15 us/step on this machine (baseline 37 us).
- [ ] mlp16 at or below ~50 ms/update with the profile script (baseline 116 ms).

## Out of Scope
Batched or vectorized env, `AsyncVectorEnv` or multi-process collection, multi-node training, optimizing the A2C update or the VQC heads, the `vqc` / `vqc-reup3` shape crash at `quantum_heads/vqc_head_h.py:79`, and removing `SyncVectorEnv`.

## Open Questions
1. Should eval keep the same thread setting? (Assumed yes: it runs in the same process.)
2. If the new thread setting costs more than it saves on the GPU path (`device != cpu`), should it apply only to CPU? (Assumed: set it unconditionally, as QML_Thesis does. Training here runs on CPU.)
