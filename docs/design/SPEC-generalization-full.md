# Spec: full generalization study (part A), 10 seeds per cell

## Objective
Turn the generalization results of plan section 12 (3 seeds for most cells) into a 10-seed study, for the report to
the advisor. Four questions:
- **A1 (K=25):** does J beat the same-size MLP at the same entropy (h5-ent), and does it match the large MLP once
  entropy is matched (h128-ent)? Section 12.2 says yes / yes on 3 seeds.
- **A2 (curve in K):** for K in {5, 25, 200, inf}, where does J's advantage over the MLPs appear and disappear, and
  does the Fourier head follow J at every K?
- **A3 (functional equivalence):** do J and the Fourier head compute the same kind of policy, beyond equal returns?
- **A4 (overfitting):** does J overfit less than the Fourier head (section 12.5 observation: gap >= 1 on 2/10 vs 6/10)?

## Setup (unchanged from sections 12.1-12.6)
- v13 env, `env_kwargs={"window": 5, "n_obstacles": 30}`, `perception=cnn`, `detach_critic=1`, split_init=1.
- `layout_pool=K` (none for K=inf), `env_seed = layout_pool_seed = seed`: every head sees the same pool per seed.
- 7.5k updates: `--inner-seconds 0 --inner-steps 150 --rounds 50 --device cpu --outer-lr 0.7 --outer-momentum 0.9`,
  `--set anneal_by=steps`; pilot entropy schedule.
- J = `vqc-nav-id` with `head_lr=2e-3`; Fourier = `fourier4` at the default lr (7e-4, the better arm of 12.5);
  h5 = `mlp5`; h128 = `mlp128`; `-ent` = `target_entropy=1.425` (J's pilot entropy at K=25).
- Test = `sample/return_mean` on 100 fresh layouts (eval seeds 10000-10099); pool = the same eval on the K training
  layouts; gap = pool - test. Mean of eval rounds 41-50.

## Runs (115 new)
| arm | K=5 | K=25 | K=200 | K=inf |
|---|---|---|---|---|
| J | s3-s9 (7) | done (10) | s3-s9 (7) | reuse 11.14 + control s9 (1) |
| Fourier | s0-s9 (10) | done (10) | s0-s9 (10) | s0-s9 (10) |
| h5 | s3-s9 (7) | s3-s9 (7) | s3-s9 (7) | s3-s9 (7) |
| h128 | s3-s9 (7) | s3-s9 (7) | s3-s9 (7) | s3-s9 (7) |
| h5-ent | | s3-s9 (7) | | |
| h128-ent | | s3-s9 (7) | | |

- **J at K=inf:** `nav-id-hlr2e-3-detc-s{0..9}-7.5k-cpu-outer` (section 11.14) all ended on 2026-10-13 by 22:48,
  before the v9 commit (2026-10-14 00:26): v8 configuration. s0 is already verified identical round by round to
  `gen-nav-id-kinf-s0`. Control for the second batch (s5-s9): rerun `gen-nav-id-kinf-s9`; it must match
  `nav-id-hlr2e-3-detc-s9` round by round, otherwise J K=inf s3-s9 are rerun (7 more runs) and this is reported.
- **Not extended:** `lin` / `lin-ent` (stable at ~2.7 on 3 seeds, SE 0.02-0.20, only excludes h5's bottleneck);
  `four-ent`, `four-lr2e-3`, `k2` (closed in 12.4-12.6).
- **Names:** `gen-<nav-id|four|h5|h128>-k<K>-sN-7.5k-cpu-outer`, `ent-<h5|h128>-k25-sN-7.5k-cpu-outer`; queued with
  `--prefix gen` / `--prefix ent` (both with `--name`).
- **Columns:** `--show sample/return_mean,greedy/return_mean,pool/sample/return_mean,sample/success_rate,entropy,value_loss`
  (`ent-*`: plus `entropy_coef`).
- **Order:** J first (longest runs), then Fourier, then the MLPs; within an arm by seed, so a partial result is
  already paired.
- **Cluster:** mac (4 slots) + nodo2 (4) + nodo1 (2). Estimate ~3 h (measured medians, mac: MLP 9.5 min, Fourier
  11.3, J 19.7; nodo2 ~1.3x slower, nodo1 ~3x).

## Reporting (rewritten 2026-10-14 after the results, see the note at the end)
Standard practice of deep RL papers (Henderson et al. 2018; Colas et al. 2018; Agarwal et al. 2021, "Deep RL at
the Edge of the Statistical Precipice"): results over seeds as a mean with a 95% confidence interval, comparisons
read from the interval.
- **Per cell:** mean +- SE over 10 seeds, per-seed values, pool, gap, entropy, greedy test return
  (`scripts/summarize_gen_pilot.py`, `scripts/summarize_eval_metrics.py`).
- **Comparisons J - X:** paired per seed (same seed = same pool, which removes the pool difficulty from the
  difference): mean difference with its 95% CI, mean +- t_0.975(n-1) SE (`summarize_gen_pilot.py`).
- **Reading:** the CI excludes 0 -> "J better" / "X better"; the CI includes 0 -> "no difference detected", always
  with the width of the CI (a narrow CI around 0 is informative, a wide one is not). Never "equivalent" from a CI
  that includes 0.
- **Curve:** test return vs K per arm (mean +- SE), and J - X vs K with the CI, for X in {h5, h128, Fourier}
  (`scripts/plot_gen_curve.py` -> `docs/figures/partA_curve_k.png`).
- **Questions read this way:** A1 = J - h5-ent and J - h128-ent at K=25; A2 = J - h5, J - h128, J - Fourier per K
  ("J's advantage over the same-size MLP depends on K" if J - h5 excludes 0 at small K and includes it at K=inf);
  A4 = paired gap difference J - Fourier per K. J - Fourier at K=25 was already seen in 12.5 and is reported as such.
- **Reproducibility control** (J at K=inf, done 2026-10-14): runs are bit-identical on the same machine and diverge
  from round 2 across machines (mac vs nodo2); the reused J K=inf runs are valid; s9 enters the cell with
  `nav-id-hlr2e-3-detc-s9`.

## Entropy-matched control at K=5 (added after the results)
At K=5 every head overfits (test < 0) and the final entropies differ a lot (rounds 41-50: Fourier 0.04, h128 0.01,
h5 0.34, J 0.67), so "J better than Fourier at K=5" may be J staying stochastic, as in 12.2 for h128 at K=25.
- **Runs (20):** `ent-four-k5-s{0..9}-7.5k-cpu-outer` (fourier4, default lr) and `ent-h5-k5-s{0..9}-7.5k-cpu-outer`
  (mlp5), same flags and pools as the K=5 cells, `target_entropy=0.674` = mean entropy of J at K=5 over its 10 seeds
  (rounds 41-50; per seed 0.10-1.21). Cluster: mac + nodo2 (nodo1 removed from the pool by the user).
- **Reading:** first check that the control reached its target (entropy within +-0.1 of 0.674), otherwise it is
  reported as failed. Then J - four-ent and J - h5-ent at K=5 as above: if J - four-ent still excludes 0, J's K=5
  advantage over Fourier is not explained by entropy; if it includes 0, the K=5 result (and the K=5 overfitting
  difference) is reported as entropy-driven.

**Result (2026-10-14): control failed.** The adaptive coefficient oscillated at K=5 (e.g. four-ent s1 entropy
1.67 -> 0.93 -> 0.12 -> 1.45 -> 0.86 at rounds 10-50): mean entropy 1.00 (four-ent) and 0.95 (h5-ent), outside
0.674 +- 0.1, and the controls no longer fit their pool (pool 1.55 / 2.50 vs ~5.5). Not interpreted. Descriptive
only: four-ent - four test +1.70 (CI [0.66, 2.74]); J - four-ent +2.27 (CI [0.89, 3.65]).

**Replacement: eval-time temperature (added 2026-10-14, before running).** Trained K=5 runs of Fourier, h5 and h128,
logits divided by a temperature T chosen per run so that the mean policy entropy on the run's own visited test states
equals J's, measured the same way (mean entropy of J at T=1 on its own visited test states, mean of its 10 K=5 runs;
0.674 is J's *training* entropy on pool states, not the same quantity). T by bisection on log T, states re-collected
at each T, until the entropy is within 0.02 of the target; then the test return at that T (100 test layouts, sampled
actions, `model_latest.pt`). J is evaluated the same way at T=1 as the reference. Answers "is J better at K=5 only because it samples more at evaluation?"; says nothing about
entropy during training. Reading as above: J - X(T) per seed with its CI.

## A3: functional equivalence (offline, no RL runs)
Do J and Fourier compute the same kind of policy, beyond equal returns? At K=25 (no difference detected, wide CI)
and K=200 (no difference detected, narrow CI: the cleanest place to ask whether equal returns mean the same
function). Runs `gen-nav-id-k{25,200}-s{0..9}` and `gen-four-k{25,200}-s{0..9}`, weights `model_latest.pt`
(round 50, end of the 41-50 window). Runs from nodo1/nodo2 have light copies here: their checkpoints are fetched
first (`parallel-train fetch NAME`), read-only. The policy is feedforward (memory is part of the observation), so
any policy can be evaluated on any state.

**A3.1 Policy distance.** For each run, 2000 observations from its own rollouts with sampled actions on the 100
test layouts (eval seeds 10000-10099). Distance between two runs = mean TV distance between their action
distributions over the states of both; KL(J || F) and KL(F || J) alongside. Reported as a 2x2 table per K, because
different seeds also mean different training pools:

| | same head | different head |
|---|---|---|
| **same pool** | - (one run per cell) | J_s vs F_s |
| **different pool** | J_s vs J_s', F_s vs F_s' | J_s vs F_s' |

"Same kind of policy" if J vs F (different pool) is about as far as two seeds of the same head; the head effect is
the difference between the two, read as a number next to the seed spread.

**A3.2 Is the periodicity used?** For every J data-gate angle (w N_j + c X_j, computed on the run's own encoder
features) and every Fourier projection z_k = W_k . x: rho = (range of the angle over the states, 1st-99th
percentile) / 2 pi, i.e. how many periods the angle sweeps on the real data. Born probabilities and sin/cos are both
2 pi-periodic in these angles, so rho is comparable across heads. rho >= 0.5 (the angle sweeps >= pi, the response
can turn back) = periodic regime; below = smooth regime. If neither head reaches the periodic regime in most seeds,
reason X is reported as "a smooth, bounded, low-capacity map", not as "Fourier structure". Alongside, descriptive:
per encoder feature, the action probabilities along a sweep of that feature over its 1st-99th percentile range
(64 points, others fixed, 200 visited states), and the mean number of turning points (local extrema) per sweep:
0 = monotone, >= 1 = the response turns back. (Counted instead of an FFT, written before running: the FFT of a
monotone ramp leaks power into every harmonic and would read as periodicity.)
Amplitude (added 2026-10-14 after the first A3.2 run, before rerunning the sweeps): turning points count direction
changes whatever their size. Per sweep curve also: non-monotone amplitude = sum |steps| - |end - start| (0 for a
monotone curve, the probability that goes back and forth), and significant turning points = reversals into a
monotone stretch whose net change is >= 0.02 in probability. Reported per head next to the plain count.

**A3.3 Cross-distillation** (`quantum_heads/analysis/distill.py`, variant `frozen`: teacher encoder frozen, only the
student head trained on KL(teacher || student)). Per teacher, 20000 states from its rollouts on fresh layouts
(seeds 30000+, disjoint from eval and from the training pools), 80/20 train/val. Students J, Fourier and MLP64 (a
generic ceiling), 3 init seeds, head lr from {1e-3, 3e-3} chosen on val. Metrics: explained fraction
1 - KL(teacher || student) / KL(teacher || teacher marginal) on val, and the test return of the student (teacher
encoder + student head). Read against the same-type student (J <- J, F <- F): a cross student close to it means the
other class can represent that policy; an asymmetry says which class contains the other on these data.

**A3.3 lr control (added 2026-10-14 after the first A3.3 results, before running).** In the first run every
(K, teacher, student) chose lr 3e-3, the upper edge of the grid, so the students may be under-trained and the
asymmetry seen (J <- F frac 0.79 / 0.86 vs F <- F 0.98; F <- J 0.88 / 0.89 vs J <- J 0.95) may be optimization, not
capacity. Control: the same students (all three types, same data and init seeds) at lr 1e-2, lr then re-chosen on val
over {1e-3, 3e-3, 1e-2}. Reading: if J <- F gets within 0.05 of the F <- F ceiling, the asymmetry is reported as an
optimization limit; if the gap stays at ~0.1 or more, as a property of the two classes; if 1e-2 is chosen (edge
again), said explicitly, no further values.

**Outputs.** `quantum_heads/analysis/policy_equivalence.py` (A3.1, A3.2, with tests), the distillation through
`distill.py` adapted to J/Fourier teachers; results in `quantum_heads/analysis/runs/a3/`, summary in plan 12.7.

## Commands
```
Tests:    .venv/bin/python -m pytest -q
Queue:    scratchpad queue script (queue add --prefix gen|ent --name ... one run per call)
Summary:  .venv/bin/python scripts/summarize_gen_pilot.py 'gen-*' 'ent-*' \
            --ref 'nav-id-hlr2e-3-detc-s*-7.5k-cpu-outer=nav-id:inf' \
            --pairs nav-id:h5,nav-id:h128,nav-id:four,nav-id:h5-ent,nav-id:h128-ent --extra entropy
Greedy:   .venv/bin/python scripts/summarize_eval_metrics.py 'gen-*' 'ent-*' --ref '...'
```
(`gen-nav-id-kinf-s9` and `nav-id-hlr2e-3-detc-s9` must not both enter the K=inf cell: one is the control.)

## Boundaries
- **Always:** same pools and flags across arms; rounds 41-50; all 10 seeds reported, failures included; `--prefix`
  with `--name`; old cells recomputed with the same scripts.
- **Ask first:** more seeds, other K, other learning rates; excluding a seed.
- **Never:** change J, the Fourier head, the env or the eval; add seeds or arms after looking at the results without
  saying so.

## Note on the rewrite (2026-10-14)
The first version fixed, before the runs, a decision procedure taken from clinical statistics (equivalence margin
1.0 / TOST, Holm-Bonferroni over four primary comparisons, four outcome classes; for A3 a bootstrap with a relative
margin and fixed thresholds). The user asked for the reporting standard of RL papers instead. Rewritten after the
part A summary had been seen; checked that no conclusion changes: J - h5-ent K=25 CI [0.82, 3.44], J - Fourier
K=5 [3.21, 4.73] (both "J better" under both versions), J - Fourier K=200 [-0.79, 0.64] and K=inf [-1.13, 1.23]
(both "no difference detected"; the old version called K=200 "equivalent within 1.0"). The file is not in git;
the previous text survives only as a temporary copy in the session scratchpad.
