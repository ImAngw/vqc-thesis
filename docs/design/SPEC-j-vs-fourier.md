# Spec: J vs Fourier head at K=25 (10 seeds)

## Objective
Resolve whether the VQC head J generalizes better than its classical periodic counterpart, the Fourier head
`fourier4` (120 vs 117 parameters), with K=25 training layouts.

In plan section 12.4 (3 seeds, no entropy constraint), Fourier reached J's level on 2 seeds (4.77, 4.83 vs J 4.64,
5.18) and failed on the third (1.64). J - Fourier = +0.96 +- 0.87, unresolved.

Two questions:
1. **Mean:** is J's test return higher than Fourier's, or are they equivalent?
2. **Reliability:** does J fail less often? In 12.4, 1 of 3 Fourier seeds failed, against 0 of 3 for J.

The pilot also left an asymmetry open: J trains its head at `head_lr=2e-3`, the classical heads at the default
`lr=7e-4`. The Fourier head therefore runs at both learning rates. The comparison that matters is J against the
better of the two Fourier arms.

## Design
No code change: `fourier4`, `head_lr` (it applies to any head's parameters, `build_optimizer`), the layout pool and
the eval all exist.

Pilot setup for every run:
- v13 env, `window=5, n_obstacles=30`, `layout_pool=25`;
- `perception=cnn`, `detach_critic=1`, split_init=1;
- pilot entropy schedule, **no** `target_entropy`;
- 7.5k updates, outer step 0.7/0.9, `anneal_by=steps`;
- `env_seed = layout_pool_seed = seed`, so all three arms share each seed's pool.

| arm | head | head_lr | seeds | new runs |
|---|---|---|---|---|
| J | `vqc-nav-id` | 2e-3 | 0-9 | 7 (s3-s9; s0-s2 from the pilot) |
| Fourier | `fourier4` | default (7e-4) | 0-9 | 7 (s3-s9; s0-s2 from 12.4) |
| Fourier, lr 2e-3 | `fourier4` | 2e-3 | 0-9 | 10 |

- **Total:** 24 new runs, about 60 min on mac (4) + nodo2 (4) + nodo1 (1).
- **Names:** `gen-nav-id-k25-sN`, `gen-four-k25-sN`, `gen-four-lr2e-3-k25-sN` (+ `-7.5k-cpu-outer`), queued with
  `--prefix gen`. The summary cells are `nav-id`, `four`, `four-lr2e-3`.
- **Columns:** `--show sample/return_mean,greedy/return_mean,pool/sample/return_mean,sample/success_rate,entropy,value_loss`.
- **Order:** seeds 3-9 of all three arms are interleaved by seed, then Fourier lr 2e-3 seeds 0-2, so a partial
  result is already paired.

## Analysis (mean of rounds 41-50, test = `sample/return_mean` on fresh layouts)
- **Per arm:** mean +- SE over 10 seeds, the per-seed values, pool, gap, entropy.
- **Paired J - Fourier per seed** for both Fourier arms: mean +- SE and **t = mean / SE** (9 degrees of freedom:
  |t| > 2.26 is p < 0.05 two-sided).
- **Reliability:**
  - number of failed seeds per arm, with a failure defined beforehand as test < 3.0 (the failed Fourier seed scored
    1.64; J's worst pilot seed 4.32);
  - the per-arm SD.
- `scripts/summarize_gen_pilot.py` gets a `t` column in the paired differences (small change, test it).

## Decision rule (fixed now, before the runs)
Let F* be the Fourier arm with the higher mean.
- **J better:** J - F* >= 0.5 with t >= 2.26. The VQC beats its classical periodic counterpart at equal size: X is
  more than the Fourier structure.
- **Equivalent:** |J - F*| < 0.5, or |t| < 2.26. X = periodic structure, and the VQC is one way to get it. If J
  still has clearly fewer failures (e.g. 0-1 vs >= 3 of 10), add "the VQC realizes it more reliably" as a secondary
  claim, stated as a count, not a test.
- **Fourier better:** F* - J >= 0.5 with t >= 2.26.
- Pre-registered as written: no extra seeds or arms are added after looking at the results without saying so.

## Commands
```
Tests:    .venv/bin/python -m pytest -q
Queue:    a script like scratchpad/queue_ctl.sh (queue add --prefix gen --name ... one run per call)
Summary:  .venv/bin/python scripts/summarize_gen_pilot.py 'gen-nav-id-k25-*' 'gen-four-k25-*' 'gen-four-lr2e-3-k25-*' \
            --pairs nav-id:four,nav-id:four-lr2e-3 --extra entropy,greedy/return_mean
```

## Boundaries
- **Always:**
  - same pools and flags across arms;
  - mean of rounds 41-50;
  - report all 10 seeds, failures included;
  - `--prefix` with `--name`.
- **Ask first:**
  - more seeds, other K, other learning rates, or an entropy constraint;
  - excluding a seed for any reason.
- **Never:**
  - change J, the Fourier head, the env or the eval;
  - pick checkpoints on test layouts;
  - `--no-window`;
  - commit without being asked.

## Success Criteria
- [ ] `t` column added to the summary script, tested, suite green.
- [ ] 24 runs finished (or failures reported with their cause).
- [ ] Plan section 12.5: per-arm table, paired J - F with t, failure counts, decision per the rule above.

## Out of Scope
- Other K values (K=5, inf).
- MLP arms.
- Entropy-matched variants.
- Fourier frequency or width sweeps.
- Classical Born-like heads.

## Open Questions
1. The failure threshold of 3.0 is my choice, set before seeing the data. OK, or a different definition (e.g. pool -
   test >= 1.0, i.e. overfit)?
2. nodo1 is slow (about 35 min per run). Assumed: keep it in the pool (1 slot), at the cost of a possibly longer tail.
