# Spec: Entropy-matched MLP control

## Objective
Test whether J's generalization advantage in the pilot (plan section 12.1) survives when the MLP policies are as
stochastic as J's.

In the pilot, at K=25, J scores 4.71 on fresh layouts against 2.09 (h5) and 1.62 (h128). However, the MLP policies
collapse (h128 entropy ~0.03, h5 0.83-1.27) while J stays at ~1.42, and the `sample/*` eval draws actions from the
policy. A stochastic policy can do better on a layout it has not memorized simply by not repeating a memorized route.
If that explains the gap, reason X is "J stays stochastic", which an MLP can get too, not "J generalizes better".

The control: retrain h128 and h5 with an **adaptive entropy coefficient** that holds their training entropy at J's
measured entropy for the same K. Then compare the test return against J's pilot runs (same seeds, same pools).

Targets (J's pilot entropy, mean of rounds 41-50 over seeds 0-2):
- **K=25:** 1.425 (seeds: 1.423, 1.389, 1.463).
- **K=inf:** 1.521 (seeds: 1.542, 1.556, 1.464).

K=inf checks that a forced-stochastic MLP does not lose much return when there is nothing to memorize.

## Tech Stack
Python 3.13, torch, pytest, parallel-train. No new dependencies.

## Design
- **Config (`rl_train/config.py`):**
  - `target_entropy: float = 0.0`: 0 = off, the current linear schedule `entropy_start -> entropy_end`. If > 0, the
    entropy coefficient is adaptive.
  - `entropy_lr: float = 0.01`: step size of the coefficient's log.
  - Validation: `target_entropy` in `[0, ln 8)`; `entropy_lr > 0`.
- **Trainer (`rl_train/a2c.py`), in the update shared by `train.py` and the adapter:**
  - State `log_ent_coef`, initialised to `log(entropy_start)`.
  - With a target, `entropy_coef()` returns `exp(log_ent_coef)`. After each update:
    `log_ent_coef += entropy_lr * (target_entropy - entropy)`, where `entropy` is the batch mean already computed
    (detached). The coefficient therefore grows while the entropy is below target and shrinks while it is above.
  - The log is clamped to `[log 1e-4, log 1.0]`, so the coefficient cannot run away.
  - This is the SAC-style temperature update written as plain SGD on log alpha: no extra optimizer state.
  - The coefficient is a training-loss term only. The policy, the sampling and the eval are unchanged (Born readout
    untouched; it does not apply to the MLPs anyway).
- **Adapter (`parallel_adapter.py`):**
  - `_build_trainer` initialises the same state, because it bypasses `A2CTrainer.__init__`.
  - With a target, `train_step` does not overwrite the coefficient through `n_updates`. The update lives in
    `A2CTrainer.update`, so both paths share it.
  - `entropy_coef` becomes a logged train metric (added to `LOGGED`).
  - Limitation: the coefficient is not checkpointed; a resume restarts it from `entropy_start`.
- **Eval:** unchanged. The runs also show `greedy/return_mean`, which is already computed, so we also get the argmax
  policy's return on fresh layouts as a second, entropy-free view.

## Run grid
- **Heads:** h128 (`mlp128`) and h5 (`mlp5`), with `--set target_entropy=<target of K>`. Everything else is identical
  to the pilot: `detach_critic=1`, `perception=cnn`, `env_kwargs={"window": 5, "n_obstacles": 30}`, split_init=1,
  7.5k updates, outer step 0.7/0.9, `anneal_by=steps`.
- **K:** 25 and inf. **Seeds:** 0-2, with `env_seed = layout_pool_seed = seed`: the same pools as the pilot.
- **Total:** 2 heads x 2 K x 3 seeds = 12 runs, about 30 min on mac + nodo2 (4+4 slots).
- **Calibration first:** one run, h128 K=25 seed 0. Its entropy in rounds 41-50 must sit within +-0.1 of 1.425 before
  the other 11 are queued. If it does not, adjust `entropy_lr` (ask first).
- **Names:** `ent-<h128|h5>-k<25|inf>-s<seed>-7.5k-cpu-outer`, queued with `--prefix ent` (one sweep on the watch page).
- **Columns:** `--show sample/return_mean,greedy/return_mean,pool/sample/return_mean,sample/success_rate,entropy,entropy_coef,value_loss`.

## Commands
```
Tests:      .venv/bin/python -m pytest -q
Calibrate:  parallel-train queue add --prefix ent --name ent-h128-k25-s0-7.5k-cpu-outer --seed 0 \
              --set head=mlp128 --set target_entropy=1.425 --set layout_pool=25 --set env_seed=0 \
              --set anneal_by=steps --set detach_critic=1 --set perception=cnn \
              --set 'env_kwargs={"window": 5, "n_obstacles": 30}' -- <pilot options, --show above>
Summary:    .venv/bin/python scripts/summarize_gen_pilot.py 'ent-*' 'gen-nav-id-k25-*' \
              --ref 'gen-nav-id-kinf-s0-7.5k-cpu-outer=nav-id:inf' \
              --ref 'nav-id-hlr2e-3-detc-s[12]-7.5k-cpu-outer=nav-id:inf' --pairs nav-id:h5,nav-id:h128
```
`scripts/summarize_gen_pilot.py` learns the `ent-` name pattern (head = `h128-ent`, `h5-ent`), so the pairs read
`nav-id:h5-ent,nav-id:h128-ent`. It also gets an optional per-cell column for any logged metric (`--extra
entropy,greedy/return_mean`).

## Project Structure
- `rl_train/config.py`: `target_entropy`, `entropy_lr`.
- `rl_train/a2c.py`: adaptive coefficient in `entropy_coef` / `update`.
- `parallel_adapter.py`: entropy state in `_build_trainer`, no schedule override with a target, `entropy_coef` logged.
- `rl_train/tests/test_a2c.py` or `test_pipeline.py`: tests below.
- `scripts/summarize_gen_pilot.py`: `ent-` names and `--extra` columns.
- `docs/design/PLAN-vqc-head.md`: results as section 12.2.

## Code Style
Match `rl_train/a2c.py`: English docstrings, config flags as int/float (argparse), no new modules. Intended shape:

```python
def entropy_coef(self) -> float:
    if self.cfg.target_entropy > 0:  # adaptive: held at the target entropy (SAC-style, SGD on log alpha)
        return math.exp(self.log_ent_coef)
    ...  # current linear schedule
```

## Testing Strategy
- **Off = unchanged:** with `target_entropy=0`, the coefficient sequence over 20 updates equals the current schedule,
  and a short run gives identical metrics to the code before the change, for the same seed.
- **Direction:**
  - an update with batch entropy below target raises the coefficient, one above lowers it, by
    `exp(entropy_lr * (target - H))`;
  - the clamp bounds hold.
- **Integration:** a short `A2CTrainer` run with `target_entropy=1.5` logs `entropy_coef` and stays finite.
- **Config:** invalid `target_entropy` (< 0 or >= ln 8) and `entropy_lr <= 0` raise `ValueError`.
- **Script:** parses `ent-` names; `--extra` reads `entropy` and `greedy/return_mean`.
- Report the commands run and the outcome.

## Boundaries
- **Always:**
  - same seeds, pools, budget and flags as the pilot;
  - J is not rerun: its pilot runs are the comparison;
  - report the mean of rounds 41-50, never the best round;
  - `--prefix ent` with `--name`.
- **Ask first:**
  - a different `entropy_lr` after the calibration;
  - adding K=5 (J's entropy there is 0.54-1.21, not a clean target);
  - more seeds.
- **Never:**
  - change J or the VQC heads;
  - change the eval;
  - pick checkpoints on test layouts;
  - `--no-window`;
  - commit without being asked.

## Success Criteria
- [ ] Tests pass, commands and outcome reported.
- [ ] Calibration: h128 K=25 s0 entropy (rounds 41-50) within +-0.1 of 1.425.
- [ ] All 12 runs finished. Every run's entropy is within +-0.15 of its target, otherwise it is flagged as not matched.
- [ ] Table and reading in plan section 12.2: test, greedy, pool and entropy per cell, and paired J - MLP per seed.
  Reading rule at K=25, on means over seeds 0-2:
  - **Entropy does not explain it:** J - matched MLP >= 1.0 for both MLPs, and J ahead on >= 2/3 seeds.
  - **Entropy explains most of it:** the matched MLP comes within 0.5 of J. Reason X is reframed as stochasticity.
  - **In between:** partial. Reported as such, with the share of the pilot gap closed.

## Out of Scope
- The Fourier-feature control (its own spec, next).
- Entropy-matching J to the MLPs.
- K=5 and K=200.
- The full 10-seed study.
- Checkpointing the coefficient.

## Open Questions
1. Re-evaluating J's pilot runs greedily needs reading `checkpoints/`, which was denied earlier in the session. The
   control runs above log `greedy/return_mean`, but J's pilot logs do not.
   - **Assumed:** compare sample vs sample for J; greedy only within the new runs.
   - **Alternative:** you allow reading `checkpoints/gen-nav-id-*` for an offline greedy eval of J (no training).
2. `entropy_lr = 0.01` is a guess: with an entropy error of ~1, the coefficient changes by a factor of e every
   ~100 updates. The calibration run checks it.
