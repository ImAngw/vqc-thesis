# Spec: Small classical heads at J's entropy (linear and Fourier)

## Objective
Find out why J beats the parameter-matched MLP h5 at the same entropy (plan section 12.2: at K=25, J 4.71 vs
h5-ent 2.13, +2.59 on 3/3 seeds) while the 2.7k-parameter h128 at that entropy almost catches up (4.37).

There are three candidate explanations, each with a classical head of about the same size as J (117 parameters),
trained at J's entropy:

| candidate | control head | params | if it reaches J... |
|---|---|---|---|
| h5's 5-unit bottleneck (12 -> 5 -> 8) is the problem, not being classical | **linear** `shared`: Linear(12 -> 8), no bias (exists) | 96 | ...X was an artifact of h5 |
| the periodic (Fourier) structure of a re-uploading VQC | **Fourier** `fourier4`: 4 learned frequency vectors, sin/cos, Linear(8 -> 8) | 120 | ...X = periodic structure, obtainable classically |
| something specific to the circuit (Born readout of a 3-qubit state) | none | - | J beats both: X is specific to the VQC head |

Why Fourier: J uploads each of the 12 encoder features into several rotations (3 qubits x 3 blocks for the
navigator) with trainable scales. Its Born probabilities are therefore trigonometric polynomials of linear
combinations of the features, i.e. a truncated Fourier series with learned frequencies. `fourier4` is the plainest
classical model with that ingredient: learned frequencies, then a linear map. The Born readout (logits = log P) is
the part it does not copy.

## Design
- **`classic_heads/fourier.py`, `FourierHead(n_freq=4)`:**
  - `z = x @ W.T` with `W` of shape (4, 12), learned and initialised `randn * freq_std`, `freq_std = 1.0`;
  - `phi = [sin z, cos z]` (8 values);
  - logits = `Linear(8 -> 8)(phi)`;
  - parameters: 48 + 64 + 8 = **120**.
  - Exported from `classic_heads/__init__.py` and registered as `fourier4` in `train.py` `HEADS` and in
    `rl_train/config.py` `HEADS`.
- **Linear:** the existing `shared` head (96 parameters), unchanged.
- **Nothing else changes:**
  - the adaptive entropy coefficient comes from plan 12.2: `target_entropy` 1.425 (K=25) and 1.521 (K=inf),
    `entropy_lr` 0.01;
  - pilot setup: v13 env with `window=5, n_obstacles=30`, `perception=cnn`, `detach_critic=1`, split_init=1,
    7.5k updates, outer step 0.7/0.9, seeds 0-2 with the same pools.

## Run grid (18 runs, about 40 min on mac + nodo2)
- **Entropy-matched (the decisive part), prefix `ent`:** `ent-<lin|four>-k<25|inf>-s<0-2>-7.5k-cpu-outer`, with
  `--set head=<shared|fourier4> --set target_entropy=<1.425|1.521>`. That is 12 runs.
- **Unmatched, K=25 only, prefix `gen`:** `gen-<lin|four>-k25-s<0-2>-7.5k-cpu-outer`, entropy schedule as in the
  pilot. That is 6 runs. They show whether these heads collapse on their own like the MLPs, and complete the pilot
  table.
- **Columns:** `--show sample/return_mean,greedy/return_mean,pool/sample/return_mean,sample/success_rate,entropy,entropy_coef,value_loss`.
- Names follow the existing patterns, so `summarize_gen_pilot.py` and `summarize_eval_metrics.py` work unchanged
  (cells `lin`, `four`, `lin-ent`, `four-ent`). `summarize_eval_metrics.py` reads `checkpoints/`, so you run it.

## Commands
```
Tests:    .venv/bin/python -m pytest -q
Queue:    a script like scratchpad/queue_ent.sh (queue add --prefix ... --name ..., one run per call)
Summary:  .venv/bin/python scripts/summarize_gen_pilot.py 'gen-*-k25-*' 'ent-*' 'gen-*-kinf-*' \
            --ref 'nav-id-hlr2e-3-detc-s[12]-7.5k-cpu-outer=nav-id:inf' \
            --pairs nav-id:lin-ent,nav-id:four-ent,nav-id:h5-ent,nav-id:lin,nav-id:four --extra entropy
Greedy:   (you) .venv/bin/python scripts/summarize_eval_metrics.py 'gen-*' 'ent-*' --ref ... --pairs <same>
```

## Testing Strategy
- **`FourierHead`:**
  - 120 parameters;
  - output (B, 8) for input (B, 12);
  - every parameter gets a gradient;
  - the same seed gives the same init;
  - logits equal `Linear(cat(sin(xW^T), cos(xW^T)))` computed by hand.
- **Registry:** `fourier4` builds through `build_policy` with `perception=cnn, window=5`, and a 2-update
  `A2CTrainer` run works.
- **Existing suite:** passes unchanged.
- **Smoke run** of one `ent-four` config, 2 rounds, with `train.py` before launching the rest.

## Boundaries
- **Always:**
  - same seeds, pools, budget, flags and entropy targets as plan 12.2;
  - mean of rounds 41-50;
  - `--prefix` with `--name`;
  - report the commands run and their outcome.
- **Ask first:**
  - a different `n_freq`, `freq_std` or an extra Fourier variant;
  - K=5 or K=200;
  - more seeds.
- **Never:**
  - change J, the VQC heads, the env or the eval;
  - `--no-window`;
  - commit without being asked.

## Success Criteria
- [ ] Tests and smoke run pass, outcomes reported.
- [ ] 18 runs finished; every entropy-matched run's entropy (rounds 41-50) within +-0.15 of its target, or flagged.
- [ ] Plan section 12.4: table (test sample, greedy, pool, entropy per cell) and paired J - head per seed.
  Reading rule at K=25, entropy-matched, sample test, means over seeds 0-2:
  - **Specific to the circuit:** J - lin-ent >= 1.0 and J - four-ent >= 1.0, J ahead on >= 2/3 seeds for each.
  - **Periodic structure:** four-ent within 0.5 of J, and lin-ent >= 1.0 below J.
  - **h5 artifact:** lin-ent within 0.5 of J.
  - **Otherwise:** partial, reported with the gaps.

## Out of Scope
- Classical Born-like heads, e.g. logits = log of a normalized trigonometric polynomial.
- Frequency or `freq_std` sweeps.
- K=5 and K=200.
- The full 10-seed study.

## Open Questions
1. `freq_std = 1.0` (standard random-Fourier-features scale; the encoder features lie in [-1, 1]). J starts from
   very small data scales (0.05) and a near-uniform policy. Assumed: `freq_std = 1.0` with the default init of the
   output layer. Is a near-uniform start like J's preferred instead (e.g. output layer initialised at zero)?
2. The 6 unmatched K=25 runs are not needed for the decision, but they complete the pilot table. Keep them (assumed)
   or drop them to save about 15 min?
