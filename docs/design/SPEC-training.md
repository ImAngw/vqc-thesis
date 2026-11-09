# Spec: A2C training pipeline

Source of truth for the training code. Derived from `docs/intent/a2c-training.md`. Update this file first, then code and tests.

- **Status:** draft, open questions resolved, waiting for final approval (no code written yet).
- **Env:** `rl_env.env.CustomEnv` (see `rl_env/ENV_SPEC.md`), observation is a `Dict`, action space `Discrete(8)`.

## 1. Objective
A generic A2C `Trainer`: given an `nn.Module`, run training with K parallel envs, evaluate every N updates, test the best model at the end, and write plot-ready logs to `logs/`.

Success means one command runs a full training and leaves in `logs/<run_name>/`:
- `metrics.json`: a single dict `{metric: [values]}`;
- `best.pt` and `last.pt`: model weights;
- `config.json`: the full config used (reproducibility).

## 2. Assumptions (correct me before approval)
1. The observation is a `Dict`, so the network receives a dict of batched tensors, not a flat array: `obs["navigator"]` `(B, 17)` and `obs["perception"]` `(B, 1, 5, 5)`. The intent said `forward(obs)`; this is its concrete form.
2. Plain A2C: n-step returns bootstrapped with `V(s_{t+n})`, no GAE, one gradient step per rollout, no minibatches.
3. Default hyperparameters (all in `A2CConfig`, overridable by CLI): `n_envs=16`, `n_steps=16`, `gamma=0.99`, `lr=7e-4` with RMSprop (classic A2C), `value_coef=0.5`, `max_grad_norm=1.0`, advantage standardised per rollout (`norm_advantage=1`), entropy coefficient annealed linearly from `entropy_start=0.02` to `entropy_end=0.001` over `entropy_anneal_updates=2500` updates, then constant (same schedule as QML_Thesis). These are standard guesses, not tuned for this env.
4. Training length is set in updates (`total_updates`), consistent with eval every N updates. One update consumes `n_envs * n_steps = 256` env steps by default.
5. Device: CPU by default (small net, MPS/CUDA overhead not worth it); `--device` overrides.
6. Truncation vs termination: on `terminated` the bootstrap value is 0; on `truncated` (timeout at `max_steps`) it is `V(final_obs)`. This matters because timeouts are common early in training.
7. Vector env uses `AutoresetMode.SAME_STEP` (gymnasium 1.4.0). Verified: on the step an episode ends, the returned obs and top-level `info` already belong to the next episode; the terminal obs is `info["final_obs"][i]` (object array of per-env dicts) and the terminal info is `info["final_info"]` (dict of batched arrays, e.g. `final_info["explored_fraction"][i]`, `final_info["is_success"][i]`), both masked by `info["_final_obs"]` / `info["_final_info"]`. Pinned by `test_vec_env_same_step_terminal_info`.

## 3. Network contract
```python
class Policy(nn.Module):
    def forward(self, obs: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """obs: {"navigator": (B,17), "perception": (B,1,5,5)} -> (logits (B,8), value (B,))"""
```
- Feedforward only, no hidden state.
- The trainer treats the network as a black box: it only calls `forward`, `parameters()`, `state_dict()`.
- The implementation is `rl_train/policy.py`: `encoder/` (perception CNN -> 8 features, navigator MLP -> 8 features), a head from `classic_heads/` (default `MLPHead(16)`) and a critic `Linear(16,64)-ReLU-Linear(64,1)`. It is not part of the trainer.

## 4. Training loop
Per update:
1. Collect `n_steps` steps from `n_envs` envs: store obs, actions, rewards, terminated, truncated, value estimates. Actions are sampled from `Categorical(logits)`.
2. Compute bootstrapped n-step returns (assumption 6), with the recursion cut at episode boundaries.
3. Loss = `policy_loss + value_coef * value_loss - entropy_coef(t) * entropy`, with `policy_loss = -(log_prob * advantage.detach()).mean()` (advantage standardised when `norm_advantage=1`), `value_loss = mse(value, return)`.
4. One optimiser step with gradient clipping.
5. Record the per-update training metrics (section 6).
6. If `update % eval_every == 0`: run evaluation (section 5), update `best.pt` if the `best_metric` improved, flush `metrics.json`.

At the end: if the last update was not an eval update, run one more eval on the final weights (so `best.pt` always exists, even when `total_updates < eval_every`); save `last.pt`; run the test (section 5) on `best.pt`; flush `metrics.json`. `config.json` is written when the trainer is created.

## 5. Evaluation and test
- Separate single env instance, episodes run one at a time. `torch.no_grad()`, network in `eval()` mode.
- **Eval:** every `eval_every` updates, `eval_episodes` (M = 100) episodes with fixed seeds `eval_seed + i`, so successive evals use the same layouts and are comparable. Run twice on the same seeds: `greedy` (argmax) and `sample` (stochastic).
- **Best model:** selected on `eval/<best_metric>`, default `sample/return_mean` (the stochastic policy is the one A2C optimises; with a memoryless net on a partially observable env the greedy policy gets stuck in loops, see Decisions 6). Only strictly better values replace `best.pt`; ties keep the earlier model. `best_metric` is `<greedy|sample>/<return_mean|success_rate|explored_fraction_mean>` (all higher-is-better), validated in `A2CConfig`. Greedy and sample are both always logged.
- **Test:** once, on the weights of `best.pt`, `test_episodes` (500) episodes with seeds `test_seed + i`, disjoint from eval seeds (default `eval_seed = 10_000`, `test_seed = 20_000`), both modes.
- Per mode, per run: `return_mean`, `return_std`, `length_mean`, `success_rate` (`info["is_success"]` at the last step), `explored_fraction_mean` (`info["explored_fraction"]` at the last step).

## 6. Metrics file
`logs/<run_name>/metrics.json` is one flat dict `{name: [values]}`. Lists of different groups have different lengths, so each group carries its own index metric.

| Group | Index metrics (one value per row) | Metrics |
|-------|-----------------------------------|---------|
| `train/` (one row per update) | `train/update`, `train/step`, `train/episode` (cumulative) | `train/policy_loss`, `train/value_loss`, `train/entropy`, `train/return_mean`, `train/length_mean`, `train/explored_fraction_mean` |
| `eval/` (one row per eval) | `eval/update`, `eval/step`, `eval/episode` | `eval/{greedy,sample}/{return_mean, return_std, length_mean, success_rate, explored_fraction_mean}` |
| `test/` (one row) | same index metrics | `test/{greedy,sample}/{same five metrics}` |

- `train/return_mean`, `train/length_mean`, `train/explored_fraction_mean` are means over the episodes that finished in that rollout, evaluated at episode end; `NaN` if none finished (no false zeros in plots).
- Written with an atomic replace (write to temp file, then `os.replace`) so an interrupted run never leaves a corrupt file. Rewritten at every eval and at the end, so an interrupt loses at most the training rows since the last eval.
- Format is JSON with `NaN` literals (readable with Python's `json`, not strict JSON). See Open Questions.

## 7. Project structure
```
rl_train/
  __init__.py
  config.py        A2CConfig dataclass (hyperparameters, eval/test settings, seeds, paths)
  a2c.py           Trainer: rollout, returns, loss, update, train() / test()
  evaluate.py      evaluate(model, env_fn, episodes, seed0, greedy) -> dict of metrics
  logger.py        MetricsLogger: append rows, atomic flush to metrics.json
  returns.py, rollout.py, envs.py   n-step returns, rollout collector, env factories
  policy.py        Policy: encoder + head + critic
  ../encoder/      Encoder (perception 8 + navigator 8 features)
  ../classic_heads/ MLPHead, SharedHead
  tests/           pytest tests
train.py           CLI entry point
scripts/           summarize_metrics.py
docs/              intent and this spec
logs/              run outputs (gitignored)
```

## 8. Commands
```
Install deps:  .venv/bin/pip install -r requirements.txt  (torch is now included)
Train:         .venv/bin/python train.py --run-name a2c_v1 --total-updates 2000 --eval-every 100
Summary:       .venv/bin/python scripts/summarize_metrics.py logs/<run_name>
Tests:         .venv/bin/python -m pytest rl_train/tests -q
Env tests:     .venv/bin/python -m pytest rl_env/tests -q
```

## 9. Code style
Match `rl_env/env.py`: type hints, short docstrings in English, `from __future__`-free modern syntax (`dict[str, Any]`, `X | None`), constants in upper case at module top, no comment noise. Comments, docstrings and commit messages in English.

```python
def n_step_returns(rewards, terminated, truncated, bootstrap_values, last_value, gamma):
    """Bootstrapped n-step returns; cut at episode ends, bootstrap on truncation only."""
```

## 10. Testing strategy
pytest, tests in `rl_train/tests/`. Unit tests are deterministic and CPU only; one smoke test runs a real training.

| Id | Test |
|----|------|
| T1 | `n_step_returns` on hand-computed cases: no episode end, termination (bootstrap 0), truncation (bootstrap `V(final_obs)`), mixed within a rollout |
| T2 | Rollout over a vector env: shapes, and the final obs of a finished episode is used for the truncation bootstrap, not the next episode's first obs |
| T3 | Logger: all lists of a group have equal length, `NaN` preserved on round-trip, atomic flush leaves a valid file |
| T4 | Evaluation: greedy mode is deterministic for fixed seeds (two runs identical); `sample` mode is not argmax; eval and test seeds are disjoint |
| T5 | Best-model selection: `best.pt` is overwritten only on a strictly better `best_metric`, test loads `best.pt` |
| T6 | Smoke: 3 updates, `eval_every=2`, tiny env settings, produces `metrics.json`, `best.pt`, `last.pt`, `config.json` with the expected keys |
| T7 | Reproducibility: same seed gives identical `train/*` metrics on CPU |

Claims about results (e.g. "the agent learns") are not part of the tests; they are checked by actually running a training and reading the summary of `metrics.json`.

## 11. Boundaries
- **Always:** run the tests before reporting that they pass; keep comments and commit messages in English; seed everything from `config.seed`; read logs with `tail`/summary scripts, never in full.
- **Ask first:** adding or changing dependencies (including pinning torch in `requirements.txt`); changes to `rl_env`; changing the metrics file format after the first runs exist; wide refactors.
- **Never:** commit or push without being asked; use the env test-only `options={"state": ...}` hook in training; write the network's architecture into the trainer; delete or overwrite an existing `logs/<run_name>/` without checking.

## 12. Success criteria
1. `train.py` completes a run and produces the three files of section 1.
2. `metrics.json` follows section 6: every list within a group has the same length, group index metrics present.
3. Eval runs every `eval_every` updates; `best.pt` corresponds to the highest `eval/<best_metric>` (default `eval/sample/return_mean`).
4. Test runs once at the end on `best.pt`, with seeds disjoint from eval.
5. T1-T7 pass (to be run, not assumed).
6. A real run of at least a few hundred updates shows the training metrics moving (loss, entropy, return); whether it learns the task is reported honestly after the run, not guaranteed by this spec.

## 13. Decisions (former open questions)
1. `torch==2.14.1` added to `requirements.txt` and installed in `.venv` (import verified, nothing else run yet).
2. Metrics format: JSON with `NaN` literals.
3. Flush: at every eval and at the end, nothing more.
4. Default hyperparameters of assumption 3 accepted. `n_steps` = number of env steps each env runs per rollout before one update (so one update uses `n_envs * n_steps` transitions).
5. Logs in `logs/<run_name>/`; if the folder already exists the run fails with an error (no `--overwrite` flag). `logs/` added to `.gitignore`.
6. **Primary mode changed from greedy to stochastic** (after the 2000-update run `a2c_2000`). The greedy policy had success rate <= 0.05 for the whole run while the stochastic one reached ~0.7, so selecting `best.pt` on greedy return picked an update-850 snapshot (selection on noise) and the final test understated the trained policy. Best model is now selected on `eval/sample/return_mean` by default, configurable through `best_metric`.
7. **Optional progress printing:** `--print-every N` (config `print_every`, default 0 = silent). Every N updates one `[train]` line with that update's row; at every eval a `[eval ]` block with sample and greedy metrics (marking `(new best)`), and a `[test ]` block at the end. Printing never changes what is logged.
