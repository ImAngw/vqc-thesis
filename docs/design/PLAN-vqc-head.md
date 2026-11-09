# Plan: a more expressive VQC head

Working document for the next sessions. Status as of 2026-10-10 (results of that day in sections 8 and 10).

## 1. Confirmed intent (interview, explicit yes)
- **Outcome:** replace the VQC block of `VQCHead` with a more expressive circuit. The 3 measured qubits (q0-q2) give the 8 outcomes = the 8 actions; ancillas are allowed (max 6 qubits in total). Logit = `log P(q0 q1 q2)`.
- **Why:** `VQCHead` learns but stops just above `SharedHead` (eval sample return 1.39 vs 0.99 for `shared_s0_v2`, 4.8 for `mlp128_s0_v2`).
- **Success:** eval return similar to MLP h128, multi-seed (mean and std), same training setup.
- **Fixed:** the 3 measured qubits. Encoder and post-processing must NOT change; work only on the VQC. Noise path only for the winners.
- **Out of scope:** encoder, post-processing, A2C loop, env, classic heads.
- **Implementation:** skill `/vqc-simulator:vqc-to-pytorch` (local plugin, not part of this repository), one circuit at a time, confirmation of the gate table before the PyTorch module.

The original "expressive = reach ~0/~1 on any action" was challenged by the data (section 3). Pending decision in section 6.

## 2. Done
| Ansatz | Qubits | Params | Files | Tests |
|---|---|---|---|---|
| A | 4 | 43 | `vqc_head.py`, `vqc_ref.py`, `circuit_table.md` (pre-existing) | pre-existing |
| B (A + blocks 7, 8) | 4 | 55 | `vqc_head_b.py`, `vqc_ref_b.py`, `circuit_table_b.md`, `test_vqc_head_b.py` | 16 passed |
| C (data re-uploading, features enter twice) | 4 | 61 | `vqc_head_c.py`, `vqc_ref_c.py`, `circuit_table_c.md`, `test_vqc_head_c.py` | 17 passed |
| D (C + 2 trainable-only ancillas q4, q5) | 6 | 67 | `vqc_head_d.py`, `vqc_ref_d.py`, `circuit_table_d.md`, `test_vqc_head_d.py` | 17 passed |

All in `quantum_heads/`. Classes: `VQCHeadB`, `VQCHeadC`, `VQCHeadD`. Run: `.venv/bin/python -m pytest quantum_heads/test_vqc_head_{b,c,d}.py -q` (50 passed together).
- Clean path only (no `density_matrix`, no noise tests). Qiskit equivalence in float32 and float64, parameter-shift gradients, cache after in-place updates, non-degeneracy guards (no dead parameters).
- Mutation checks done for B (4), C (5), D (5): all caught. Caveat: on macOS `sed -i ''` does not support the GNU `0,/re/` address; apply mutations with Python.
- Not verified: speed, GPU, noisy path.
- Ansatz E (entangling encoding, e.g. ZZ feature map) deferred: feature-to-pair assignment undecided.
- `train.py` / `HEADS` untouched: the new heads are not yet selectable with `--head`.

Teacher data (phase 1):
- `quantum_heads/analysis/collect_teacher_data.py` collects (encoder features, teacher logits) from rollouts of `logs/mlp128_s0_v2/best.pt` (state_dict of `Policy` with `MLPHead(128)`), encoder frozen, sampled actions.
- Output `quantum_heads/analysis/data/{train,val}.npz` (20000 / 5000 samples, seeds 30000 / 40000, disjoint from eval 10000 and test 20000). Untracked.

## 3. Findings (val set, teacher = `mlp128_s0_v2`, single seed)
- Teacher is very soft: mean entropy 1.80 nats (max 2.08), median max-prob 0.31, 3.2% of states with max-prob > 0.5, none > 0.8. Greedy returns are negative for all heads (-3.5 for h128): the sampled distribution is the policy.
- KL(teacher || uniform) = 0.283; KL(teacher || best constant marginal) = 0.220 nats. This is the ceiling of what a state-dependent head can explain.
- Features are tanh outputs in [-0.90, 0.94]. Perception features (first 8) have std 0.01-0.06; navigator features (last 8) std 0.25-0.40.
- Replacing perception with its mean changes the teacher policy by KL = 0.0003; replacing navigator with its mean by KL = 0.29. The h128 policy uses only the navigator.
- In A-D the angle is `feature + offset` (fixed gain of 1 rad per unit feature); with a frozen encoder the VQC cannot amplify small feature variations.
- Comparable runs: only `*_v2` / `vqc_s0_v3` use the current env (after commit f383f4f). Runs without suffix are on an older env.

## 4. Phase 1 harness (to write, `quantum_heads/analysis/`)
Offline, no RL, clean path, frozen encoder. Compare A, B, C, D plus calibration heads (`SharedHead`, MLP h16/h32, h128 as lower bound).
1. **Distillation (primary):** student head learns the teacher distribution on the 16 features, loss KL(teacher || student). Metric: explained fraction `1 - KL / 0.220` (also absolute KL), held-out val, several init seeds. Report parameter count.
2. **Calibration check:** the ranking must reproduce what is known (A close to `SharedHead`, h128 far better). If all ansatze are indistinguishable the proxy is not informative: stop.
3. **Diagnostics:** linear fit of teacher logits on features (is the response near linear?); Jacobian d logit / d x (rank and scale) at init and after training; gradient variance at init (barren plateau, especially D).
4. **Secondary:** random one-hot label fit (vertex reachability), perception ablation.
Decision rule to continue: a candidate beats A beyond seed variance (non-overlapping on >= 3 seeds) without degenerate gradients.

## 5. Next phases
2. Select 1-2 candidates from the harness.
3. For the winners only: noisy path (`density_matrix`, noise tests), equivalence with Aer.
4. Register in `train.py` / `rl_train/config.py` `HEADS` (e.g. `vqc_c`), without touching `vqc`.
5. RL multi-seed: winner vs `vqc`, `mlp128`, `shared`, same config and seeds; report mean and std of eval (100 episodes). Smoke test first.

## 6. Open decisions
(Status 2026-10-10: E' is implemented as G', see section 8; the phase 1 harness of section 4 is superseded by
`quantum_heads/analysis/distill.py` on the v8 teacher `mlp64_s0_v8b_n64_15k_b`; `collect_teacher_data.py` is stale,
it calls the encoder without the memory branch.)
- Move the objective from "~0/~1 on any action" to "fine modulation around a near-uniform distribution", with explained-KL fraction as primary metric and vertex reachability secondary? (proposed, not confirmed)
- Add variant E' with a trainable per-feature scale (16 parameters, inside the VQC)? Only if the gain limit shows up in the harness.
- A second teacher seed to check that `mlp128_s0_v2` is representative (costs a training run).
- Current focus requested by the user: why the perception features are almost uninformative (section 7).

## 7. Perception features investigation (in progress)
Question: why does the encoder's perception branch (CNN on a 5x5 window, 8 tanh features) end up almost constant, and is it a problem for the VQC?
Results (2026-10-09, ad-hoc scripts in the session scratchpad, not in the repo; observation sets from random actions and from teacher rollouts, 4800-6400 samples):
- Raw input is informative: per-cell std of the 5x5 window about 0.4-0.47, obstacle/outside fraction 0.19 (teacher rollouts).
- Encoder perception features are small at initialization already: std 0.016-0.020 (3 random inits, pre-tanh std about 0.017). After training: h128 0.030 (max 0.07), h16 0.006, h32 0.009-0.013, h64 0.013-0.015, shared 0.017. Training does not amplify them. Navigator features: 0.23-0.24 at init, 0.32-0.43 after training.
- The h128 head is wired to them: first-layer weight norm per input 1.66 (perception) vs 2.58 (navigator). Amplifying the perception deviations from their mean by k changes the policy by KL 0.005 (k=5), 0.022 (k=10), 0.178 (k=30), 1.21 (k=100), against a total modulation of 0.22. So the head reacts; the input variation is just about 15x too small.
- Information is partially there: a linear probe on the standardized perception features predicts neighbour occupancy above the majority baseline (e.g. 0.93 vs 0.80, 0.92 vs 0.85, 0.88 vs 0.85); the same probe on navigator features equals the baseline exactly (control).
Reading: perception is dead by scale, not (proven) by lack of information, and not specific to the VQC. For the VQC angles (`feature + offset`) a std of 0.03 rad means perception is effectively unavailable.
Not known: whether the task benefits from perception at all (no run with rescaled perception); whether the collapse is a consequence of the small init + tanh + gradient scale (hypothesis, untested).
Options: (a) RL run of h128 with perception features rescaled/normalized (touches the encoder; diagnosis only) to test whether perception helps the task; (b) accept it and treat the VQC input as the 8 navigator features, dropping the perception angles to save parameters; (c) learnable per-feature scale inside the VQC (variant E').

## 8. Results of 2026-10-10: why the VQC run has a high value loss, distillation probe
Runs compared: `vqc-mem4_s0_v8b_n64_15k` (head G, 15000 updates, test sample return of best.pt **2.05**) and
`mlp64_s0_v8b_n64_15k_b` (test **5.50**), same config. Single seed each.

### 8.1 High value loss is a symptom of the policy, not of the critic
- Binned by train policy entropy, the two runs are indistinguishable (value loss, explained variance, return mean
  and std). The MLP run also had value loss ~2.1 at entropy 1.85-1.9; it drops to ~1.0 only below entropy 1.5.
- G stays at train entropy ~1.86 (max 2.08) for the whole run, with entropy_coef already 0.001 from update 2500.
  A near-uniform policy makes the return dominated by action noise: low explained variance, high value loss.

### 8.2 Distillation probe (`quantum_heads/analysis/distill.py`)
Teacher `mlp64_s0_v8b_n64_15k_b` (val entropy 1.13, KL(teacher || marginal) 0.905). Data: teacher rollouts, sampled
actions, raw observations + teacher logits, 20000 train (seed 30000) / 5000 val (seed 40000), cached in
`quantum_heads/analysis/data/mlp64_v8/`. Student = encoder + fresh head, loss KL(teacher || student).
Variants: `frozen` (encoder frozen) and `finetune` (encoder from the checkpoint, trained with the head, lr 1e-3).
Metric: explained fraction `1 - KL / KL(teacher || marginal)` on val. Verdict rule fixed in advance: capacity
limit if mean+std of the VQC head < 0.5 * (mean-std) of an MLP64 student (in practice an absolute 50%, since the
MLP64 student has the teacher's architecture and reaches 1.0). Return: `quantum_heads/analysis/eval_students.py`
(rl_train.evaluate, 500 test episodes, seeds 20000+, sample mode). 3 init seeds everywhere; numbers are mean +- std.

| student | encoder | params | explained fraction | entropy | test sample return |
|---|---|---|---|---|---|
| MLP64 (control) | teacher, frozen | 2120 | 1.000 | 1.13 | 5.47 +- 0.02 |
| G | teacher, frozen | 51 | 0.518 +- 0.009 | 1.75 | 0.23 +- 0.50 |
| G | RL (vqc-mem4 best.pt), frozen | 51 | 0.501 +- 0.006 | 1.74 | 0.27 +- 0.14 |
| G | teacher, finetune | 51 | 0.742 +- 0.010 | 1.56 | 2.93 +- 0.62 |
| G' | teacher, frozen | 75 | 0.634 +- 0.013 (saturated) | 1.65 | not measured |
| G' | teacher, finetune | 75 | 0.756 +- 0.007 | 1.53 | not measured |
| F, logit_scale free | teacher, frozen | 57 | 0.47 | 1.66 | -0.25 +- 0.53 |
| F, logit_scale free | teacher, finetune | 57 | 0.92 | 1.22 | 4.96 +- 0.11 |
| F, logit_scale = 3 | teacher, frozen | 56 | 0.37 | 1.81 | -0.31 +- 0.15 |
| F, logit_scale = 3 | teacher, finetune | 56 | 0.86 | 1.36 | 2.30 +- 0.27 |
| G as trained by RL | its own | 51 | 0.151 | 1.86 | 2.05 (test of best.pt) |

Best head lr: 1e-2 (frozen), 3e-3 (G finetune), 1e-2 (F). Runs and logs: `quantum_heads/analysis/runs/`,
`quantum_heads/analysis/runs_*.log`.

### 8.3 Findings
1. **Optimization is not the bottleneck of G.** In distillation the head lr does not matter (G frozen 0.495 / 0.512 /
   0.518 at 1e-3 / 3e-3 / 1e-2). With RL-like settings (RMSprop, batch 1024, 8900 steps, clip 1.0, RL encoder
   frozen) G reaches 0.48 by step 3000 and 0.50 by 8900 at lr 7e-4; lr 7e-3 only gets there faster; a one-sample
   loss (cross-entropy on an action sampled from the teacher) costs at most 0.01-0.03. On the same RL encoder the RL
   head gets a far better return (2.05) than the distilled head (0.27): RL uses the 51 parameters well.
   Consequence: an RL run with a larger head lr, or warm-started from a distilled G, is not supported by the data.
2. **The RL encoder did rescale its features**: perception feature std 0.005 at init, 0.091 after RL with G,
   0.136 in the teacher; an MLP64 student on the frozen RL encoder explains 0.895. The features are informative.
3. **Explained fraction maps to return very non-linearly**: ~0.5 -> ~0, 0.74 -> 2.9, 0.86 -> 2.3, 0.92 -> 5.0,
   1.0 -> 5.5. The mean KL misses what matters (being right on the decisive actions); partial imitation also pays
   the distribution shift (no DAgger yet).
4. **G' (trainable input scale)**: +0.1 fraction at frozen encoder, nothing with finetune. Learned |fs| mean 0.7-1.4,
   max ~3 (perception below 1): the optimizer does not use larger angle ranges, so the +-1 rad range of the tanh
   features is not the main limit. Not tested: init of fs at 3 or 10.
5. **F (8 qubits, readout softmax(logit_scale * <Z_i>)) beats G only through the temperature.** Free logit_scale
   grows from 3 to 8.6-12.6 and gives 4.96; fixed at 3 F gives 2.30, comparable to G finetune (2.93). That readout
   is not a single measurement (the action is not the outcome of measuring |psi>), so it violates the constraint
   "action = measurement outcome" that holds for G (Born readout: softmax(log P) = P, one shot per step). The
   same applies to any temperature on log P. Note: greedy evaluation is not physical either (needs many shots).
6. **What the circuit must produce** (teacher on val): per state ~80% of the mass on 2 actions (max-prob mean 0.54,
   > 0.8 in 3% of states, entropy 1.13 = ~3.2 effective actions), and which 2 changes with the state (argmax share
   by action 3/22/5/20/8/16/8/17%, odd actions dominate). The students stay at entropy 1.56-1.75 (G) and 1.36 (F
   fixed): not concentrated enough, or not on the right actions per state. Working hypothesis: the limit is
   structural (each feature enters once, early, followed by data-independent blocks), not the angle range.

### 8.4 Not verified / caveats
Single teacher, single RL seed per head, no DAgger, 3 init seeds; return of G' students not measured; F's frozen
and fixed-scale rows use a single lr; the explained fraction measures similarity to one teacher, not policy quality.

### 8.5 Possible next steps (none decided)
- A Born-readout circuit with data re-uploading interleaved with the entangling blocks (more data dependence per
  state), compared in the distillation probe against G at matched parameters, with the env return as the metric.
- DAgger in the probe, to separate capacity from distribution shift.
- Classical heads with ~50-75 parameters as matched controls (e.g. MLP 24 -> 2 -> 8, 74 parameters).
- Idea raised by the user, parked: refocus the thesis on distilling a classical policy into the smallest possible
  VQC head (explained fraction and return vs parameters, against matched classical heads). Needs its own intent
  interview before any work.

### 8.6 Files added on 2026-10-10 (untracked)
- `quantum_heads/vqc_head_gs.py`, `vqc_ref_gs.py`, `test_vqc_head_gs.py`, `circuit_table_gs.md`: G' (VQCHeadGS,
  75 parameters, `fs` per feature, init 1, unconstrained). 28 tests pass (`.venv/bin/python -m pytest
  quantum_heads/test_vqc_head_gs.py -q`), 54 together with G; 3 mutations (noisy path unscaled, scale indices
  permuted, memory block unscaled) all caught. Not registered in `train.py` / `HEADS`.
- `quantum_heads/analysis/distill.py`: options `--encoder-checkpoint`, `--optimizer`, `--steps`, `--eval-steps`,
  `--loss`, `--max-grad-norm`, `--save-students`; heads `vqc-mem`, `vqc-mem-fixedscale`, `vqc-mem4`,
  `vqc-mem4-scaled`, `mlp64`.
- `quantum_heads/analysis/eval_students.py`.

## 9. Confirmed intent (2026-10-10, interview, explicit yes): a Born-readout VQC that modulates its output with the state
- **Outcome:** a VQC head with Born readout (action = outcome of measuring the 3 measured qubits, logit = log P)
  whose output distribution changes strongly with the state, through data re-uploading.
- **Candidates:** (a) G with the data re-uploaded in every block; (b) as (a) plus ancillas with dedicated encoding
  (5-6 qubits). **Control:** G widened with parameter-only blocks to the same parameter count, no re-uploading.
- **Constraints:** at most 6 qubits; soft cap of 100 trainable parameters; same 24-feature input. Each circuit goes
  through `/vqc-simulator:vqc-to-pytorch` (gate table confirmed by the user, Qiskit equivalence, tests, mutations).
- **Evaluation:** distillation probe on teacher `mlp64_s0_v8b_n64_15k_b`, variants frozen and finetune, 3 seeds.
  Primary metric: env return on the test seeds (`eval_students.py`); secondary: explained fraction, entropy.
- **Success:** finetune student return >= 4.0 (G finetune 2.93, F with logit_scale = 3: 2.30, teacher 5.47); the
  gain is attributed to the structure only if the matched-parameter control stays below.
- **Out of scope:** RL runs; noisy path (winner only, later); temperatures on the logits or <Z> readouts; ZZ-type
  feature-interaction encodings; changes to encoder, A2C or env; DAgger.

## 10. Results of the section 9 candidates (2026-10-10)
Same probe as section 8.2 (teacher `mlp64_s0_v8b_n64_15k_b`, teacher encoder, variants frozen / finetune, head lr
3e-3 and 1e-2, 3 init seeds, return = `eval_students.py`, 500 test episodes, sample mode, mean +- std over seeds).
All circuits: Born readout (logit = log P of q0-q2), clean path only, built with `/vqc-simulator:vqc-to-pytorch`
(gate table confirmed by the user, Qiskit equivalence in float32/float64, parameter-shift gradients, guards,
3 mutations each, all caught).

| head (distill name) | qubits | uploads | params | frac frozen | frac finetune | return frozen (3e-3 / 1e-2) | return finetune (3e-3 / 1e-2) |
|---|---|---|---|---|---|---|---|
| G (`vqc-mem4`) | 4 | 1 | 51 | 0.518 | 0.742 | - / 0.23 +- 0.50 | 2.93 +- 0.62 / - |
| GW (`vqc-mem4-wide`), control | 4 | 1 | 67 | 0.56 | 0.76 | 0.90 +- 0.41 / 0.66 +- 0.28 | 2.22 +- 0.29 / 2.15 +- 0.32 |
| H (`vqc-reup`) | 4 | 2 | 67 | 0.57-0.59 | 0.76-0.78 | 1.26 +- 0.28 / 0.60 +- 0.54 | 3.19 +- 0.82 / 3.15 +- 0.52 |
| HA (`vqc-reup-anc`) | 6 | 2 | 73 | 0.42-0.46 | 0.60-0.70 | 3 of 6 evaluated: -0.21, 0.15, 0.57 | 0.40 +- 0.23 / 1.43 +- 0.41 |
| GW6 (`vqc-mem4-wide6`), control | 4 | 1 | 99 | 0.59-0.62 | 0.78-0.80 | evaluation in progress | 2.65 +- 0.24 / 2.71 +- 0.32 |
| **HR3 (`vqc-reup3`)** | 4 | 3 | 99 | 0.59-0.62 | 0.80-0.82 | 0.74 +- 0.73 / 1.25 +- 0.19 | **4.44 +- 0.12 / 4.45 +- 0.37** |
| teacher MLP64 | - | - | 2120 | 1.0 | 1.0 | 5.47 | 5.47 |

Circuits (files `quantum_heads/vqc_{ref,head}_<x>.py`, `test_vqc_head_<x>.py`, `circuit_table_<x>.md`):
- **H**: G with data re-uploading. Two repetitions of [A (RY M0-3, RZ N0-3) ring, B (RY N4-7, RZ M4-7) ring,
  G (RY G0-3, RZ G4-7) ring, V (RY, RZ trainable) ring] on q0-q3, ring = CX (0,1),(1,2),(2,3),(3,0); final RY on q0-q2.
  The number of repetitions is the class attribute `reps`.
- **HR3**: H with three repetitions (`reps = (1, 2, 3)`), 99 parameters.
- **GW / GW6**: controls. G with 2 / 6 parameter-only blocks V (RY, RZ on q0-q3 + ring) after the memory block's
  ring; data uploaded once; parameter count matched to H / HR3 (class attribute `v_blocks`).
- **HA**: H on 6 qubits, the memory features moved to two ancillas q4-q5, 2 data layers per repetition; RZ of V2 on
  q0-q3 only (on q4-q5 it was dead: diagonal phase on unmeasured, never-rotated qubits, caught by the guard).

### 10.1 Findings
1. **Data re-uploading is what improves the return, not the parameter count** (finetune): at 67 parameters H beats
   GW by ~+1; at 99 parameters HR3 beats GW6 by ~+1.8, every HR3 student (worst 3.98) above every GW6 student (best
   3.06). Parameters alone: 2.2 (67) -> 2.7 (99). Uploads at matched parameters: 1 -> 2 -> 3 gives ~2.2 / 2.7 ->
   3.2 -> 4.45. Fourier reading: L uploads give frequencies 1..L in each feature, richer shapes of P(a | s) over the
   +-1 range of the tanh features (a trainable scale only stretches one frequency: G').
2. **HR3 meets the success threshold of section 9** (finetune return >= 4.0): 4.44-4.45, ~81% of the teacher, with a
   99-parameter Born-readout head; 5 of 6 students above 4.0, lowest seed std so far (+-0.12 at lr 3e-3).
3. **Only with the encoder adapted to the circuit.** With the encoder frozen, re-uploading does not stand out (H 1.26,
   HR3 0.74-1.25, GW 0.66-0.90); one GW6 frozen student reached 2.18 (evaluation incomplete).
4. **The explained fraction does not separate the circuits** (H vs GW 0.77 vs 0.76, HR3 vs GW6 0.81 vs 0.79) while
   the return differs by 1-1.8: use the env return as the primary metric, as decided in section 9.
5. **Ancillas entangled with the measured register hurt (HA).** The probability of any action is bounded by the top
   eigenvalue lambda_max of the reduced state of q0-q2. Purity Tr rho^2 / lambda_max on val states (finetune):
   H 0.65 / 0.76, GW 0.64 / 0.74, HA 0.32 / 0.47. lambda_max is below the teacher's max-prob (mean 0.535) in 66-80%
   of states for HA, 7-13% for H and GW: HA cannot concentrate as the teacher does, H is not limited by its one
   ancilla (its students stay at max-prob ~0.40, well below lambda_max). Purity does not explain H vs GW (same purity,
   return +1). Script: session scratchpad `purity.py` (not in the repo).

### 10.2 Not verified / caveats
3 init seeds, single teacher, no DAgger; distillation only (no RL run of H/HR3 yet; G in RL reached 2.05 vs 2.93
when distilled with finetune, so the gain may not transfer as is); GW6 frozen evaluation incomplete, HA frozen
evaluation stopped after 3 of 6 students; speed of HR3 not measured (3 fused matrices per forward instead of 1).

### 10.3 State and next steps
- `vqc-reup3` (HR3) is registered in `train.py` / `rl_train/config.py` `HEADS`; smoke run of 20 updates OK,
  `rl_train/tests` 40 passed. Suggested RL run, same config as the v8 runs:
  `train.py --head vqc-reup3 --run-name vqc-reup3_s0_v8b_n64_15k --n-steps 64 --total-updates 15000 --seed 0`.
- Possible next experiments: RL run of HR3 against `vqc-mem4` (2.05) and `mlp64` (5.50); a fourth upload
  (above the 100-parameter soft cap unless V blocks are trimmed); noisy path for HR3 (out of scope until now);
  more seeds for HR3 vs GW6.
- `quantum_heads/analysis/distill.py` heads: `vqc-mem`, `vqc-mem-fixedscale`, `vqc-mem4`, `vqc-mem4-scaled`,
  `vqc-mem4-wide`, `vqc-mem4-wide6`, `vqc-reup`, `vqc-reup-anc`, `vqc-reup3`, `mlp64`.

## 11. Confirmed intent (2026-10-13, interview, explicit yes): make `vqc-optim-id` reliable in RL
- **Outcome:** `vqc-optim-id` (`VQCHeadI`, 3 qubits, 3 uploads, 93 parameters) reaches the > 4 solutions it can
  already express on every seed, and keeps them (no peak-then-collapse).
- **Why:** lr and init moved the mean from ~2.8 to ~3.4 (head_lr 2e-3), but seed variance and instability remain;
  final entropy stays ~1.67 in every variant (h128 ~1.16). The user takes expressivity as established (single seeds
  reach > 4), so a distillation capacity probe is not needed.
- **Success:** eval >= 4.0 mean over 5 seeds at 7.5k updates; or, alternatively, VQC above h128 in the early part of
  the curve (to claim that at a limited budget the VQC pays off).
- **Allowed levers:** optimization dynamics of the head; the `VQCHeadI` circuit (via `/vqc-simulator:vqc-to-pytorch`).
- **Constraints:** Born readout (no temperatures / scaled logits); `ent_coef` and any other A2C-loop parameter only if
  identical for VQC and MLP (h128 rerun with the same change); same seeds, split_init, budget (7.5k updates,
  parallel-train cpu, outer step 0.7/0.9, anneal_by=steps), comparison against h128.
- **Out of scope:** distillation capacity diagnosis; encoder and post-processing; VQC-only A2C-loop hyperparameters;
  noisy path.

### 11.1 Baseline numbers (100-episode eval per round, 50 rounds = 7.5k updates, seeds 0-4)
| variant | mean of rounds 41-50 per seed | mean | first round with return >= 3 |
|---|---|---|---|
| h128-rms | 4.54 3.15 4.30 5.22 4.38 | 4.32 | 30 20 39 21 17 |
| optim-id head_lr 2e-3 | 2.00 3.60 3.15 4.47 3.21 | 3.29 | - 29 45 26 37 |
| optim-id head_lr 7e-3 | 2.65 3.05 -1.92 0.90 3.23 | 1.58 | 47 33 - - 36 |
| optim-id (head_lr = lr) | 2.56 2.74 2.70 2.62 1.79 | 2.48 | - 47 42 49 30 |

With head_lr 2e-3 the peaks fall in rounds 37-49 and are close to the last value: the curves are still rising at
7.5k. The dominant failure is slow learning (h128 reaches 3.0 around round 21, the VQC around 37 or never), not
peak-then-collapse; the clear collapse is `optim-id-s4` with head_lr = lr (4.68 at round 39 -> 1.33).

### 11.2 Phase 0 diagnosis (2026-10-13, final weights, K = 20 rollouts of 16 x 64, entropy coef 0.001)
Script: session scratchpad `phase0_summary.py` (not in the repo). Checkpoints found locally only for
`optim-id-hlr2e-3` s1-s3 and `h128-rms` s2-s4 (the others are not under `checkpoints/`, likely on nodo2).
| | VQC hlr2e-3 s1 / s2 / s3 | h128 s2 / s3 / s4 |
|---|---|---|
| return (rounds 41-50) | 3.60 / 3.15 / 4.47 | 4.30 / 5.22 / 4.38 |
| entropy | 1.70 / 1.70 / 1.66 | 1.18 / 1.14 / 1.18 |
| max-prob median, p90 | 0.33, 0.45 / 0.33, 0.47 / 0.34, 0.49 | 0.52, 0.76 / 0.53, 0.77 / 0.50, 0.71 |
| states with max-prob > 0.5 | 4% / 6% / 9% | 55% / 57% / 49% |
| spread of P(a\|s) across states | 0.10 / 0.10 / 0.11 | 0.17 / 0.18 / 0.17 |
| head gradient SNR | 0.18 / 0.32 / 0.24 | 0.21 / 0.31 / 0.30 |
| updates clipped (global norm > 1) | 60% / 90% / 85% | 80% / 85% / 75% |
- The VQC policy is state-dependent but flat: it does not concentrate (max-prob p90 < 0.5). Within the VQC the best
  seed (s3) is also the most concentrated.
- Not a VQC-specific gradient problem: head SNR (~0.2-0.3, also at init) and clipping frequency are the same for both
  heads; the critic dominates the global norm. Separate head clipping (1c) is dropped.
- |s| grew from 0.1 to 0.3-2.1, the navigator branch largest (1.1-2.1 rad per unit feature), as in the distillation
  finding that the policy uses the navigator. With RMSprop the per-step drift is ~lr x SNR (~5e-4 at head_lr 2e-3):
  the head moves at a speed set by lr and noise, and the curves are still rising at 7.5k.

### 11.3 Phase 1 results (2026-10-13, prefix `headopt`): faster head updates hurt
Mean of eval rounds 41-50 (s3/s4 stopped or not run once the trend was clear):
| variant | s0 | s1 | s2 |
|---|---|---|---|
| hlr2e-3 (reference) | 2.00 | 3.60 | 3.15 |
| `scale_lr=1e-2` (head_lr 2e-3) | 1.27 | -0.84 | -1.47 |
| `head_momentum=0.9`, head_lr 5e-4 | 0.87 | -0.37 | stopped at round 14 |
- Both below the reference on every finished seed, entropy as high or higher (1.60-1.78). With head_lr 7e-4 -> 2e-3
  -> 7e-3 giving 2.48 -> 3.29 -> 1.58, head_lr 2e-3 is near the stability limit: the phase 0 reading "the head is
  under-stepped" is wrong beyond 2e-3, and the step-size lever on the head is exhausted.
- No regression: `optim-id-hlr2e-3-ctrl-s1` (current code, mac) reproduces `optim-id-hlr2e-3-s1` digit for digit on the
  first 5 rounds (return, entropy, value loss); runs on the same machine are deterministic, so paired seeds are exact.
- Code kept (defaults unchanged): `A2CConfig.scale_lr`, `A2CConfig.head_momentum`, `rl_train.a2c.build_optimizer`
  (shared by A2CTrainer and parallel_adapter.py), test `test_scale_lr_and_head_momentum_split_the_vqc_head`.
- Loss terms (init, and value/entropy from the logs): 0.5 * value_loss ~1 dominates the loss and the encoder gradient
  (5-8x the policy term) and triggers the clip; on the head only the policy term matters (entropy at coef 0.02 is
  3-7% of it). VQC and h128 have the same entropy up to round 10 (1.81) and split afterwards (round 25: 1.74 vs 1.32).

### 11.4 Why VQCHeadI does not concentrate: tests of H1-H3 (2026-10-13)
Script: session scratchpad `hyp_test.py` (not in the repo); final weights of VQC hlr2e-3 s1-s3 and h128 s2-s4,
10240 on-policy states each. Actions are `MOVES` in compass order (N, NE, ..., NW), outcome = binary index (q0 = MSB).
| | VQC s1 / s2 / s3 | h128 s2 / s3 / s4 |
|---|---|---|
| top-2 actions compass-adjacent (random 0.29) | 0.49 / 0.39 / 0.50 | 0.70 / 0.79 / 0.91 |
| Hamming(top1, top2) binary / Gray (random 1.71) | 1.61/1.50, 1.52/1.66, 1.47/1.52 | 1.75/1.30, 1.69/1.22, 1.61/1.10 |
| polarization per qubit q0 q1 q2 (binary) | 0.70 0.59 0.60 / 0.67 0.60 0.63 / 0.72 0.60 0.64 | 0.84 0.70 0.68 / 0.89 0.72 0.66 / 0.90 0.76 0.62 |
| P_max / prod of polarizations (median) | 1.26 / 1.25 / 1.17 | 1.34 / 1.25 / 1.22 |
| features M, N, G: mean of \|mean\| / std | M ~0.3-0.5/0.05-0.11, N ~0.2-0.4/0.3-0.5, G ~0.3-0.4/0.16-0.23 | M std 0.00-0.09 |
| data-angle 5-95% range, median rad (M / N / G) | 0.08/1.50/0.11, 0.23/1.14/0.48, 0.07/0.95/0.40 | - |
- **H1 (action encoding) supported on the target side:** the good policy is compass-local (h128 top-2 adjacent in
  70-91% of states); in the binary encoding those pairs are as far apart as random actions (1.6-1.75 bits), in Gray
  code 1.1-1.3. The VQC is less compass-local (39-50%) and puts uniform-level mass on the compass neighbours of its
  argmax. Cheap test: a fixed Gray relabeling of the outcomes (0 parameters, action still = measured outcome).
- **H2 (independent qubits) refuted:** P_max / prod of polarizations is 1.2-1.3 for both heads (correlated qubits).
  The gap is the polarization of q0 and q1 (VQC 0.70 / 0.60 vs h128 0.84-0.90 / 0.70-0.76).
- **H3 partly:** G is not centred (\|mean\| >= std), M is nearly constant (also in h128 and at init: the perception
  branch carries almost no state information for either head). Only the navigator angles swing by ~1-1.5 rad (max
  3-4); M and G swing by 0.1-0.5 rad.
- **New H4:** with the branch rotation (qubit i of block r gets branch (i + r - 1) mod 3), in every block the state
  information enters on one qubit only (the one fed N); 2/3 of the data slots carry near-constant M and weak G. Harmless
  for the MLP (it ignores M), costly for the VQC. Candidate circuit change: navigator uploaded on every qubit.

### 11.5 E1 results (2026-10-13, prefix `graycmp`): Gray relabeling of the actions does not help
`vqc-optim-id-gray` (VQCHeadI action_code="gray", same parameters and init) vs hlr2e-3, mean of eval rounds 41-50
(first round with return >= 3, final entropy):
| seed | 0 | 1 | 2 | 3 | 4 | mean |
|---|---|---|---|---|---|---|
| hlr2e-3 | 2.00 (-, 1.67) | 3.60 (29, 1.69) | 3.15 (45, 1.69) | 4.47 (26, 1.63) | 3.21 (37, 1.68) | 3.29 |
| gray | 3.86 (27, 1.60) | 3.79 (19, 1.58) | 1.83 (-, 1.70) | 2.74 (26, 1.71) | 1.66 (-, 1.75) | 2.78 |
- Paired differences +1.86 +0.19 -1.32 -1.73 -1.55 (mean -0.51, t ~ -0.7): no evidence either way; entropy unchanged
  (1.58-1.75), so the relabeling does not make the policy concentrate. H1 as a lever is not supported.
- The relabeling changes which action each sample maps to, so the trajectories diverge from the first step: the
  pairing does not remove seed noise here. Seed spread of one config is ~2 points (1.66-3.86), so 5 seeds only detect
  differences of ~1.5 or more.
- Code kept: `VQCHeadI(action_code=...)`, head `vqc-optim-id-gray`, test `test_gray_action_code_relabels_the_outcomes`.

### 11.6 Ansatz J (`VQCHeadJ`, head `vqc-nav-id`): the navigator on every qubit (2026-10-13)
User's design (H4): in block r qubit i has role (i + r - 1) mod 3; A: a_k = w N_j, B: a_k = w N_j + c M_j (M as a
near-constant offset), C: a_k = w N_j + c G_j, with j = (k + i) mod 4 so every qubit reads N in a different gate order.
Same gates as I otherwise (RY RZ RY RZ data, RZ RY RZ u, CX chain, v, CX chain, RY(wf)). 117 parameters (w 36, c 24
per gate, u 27, v 27, wf 3; the user accepted going above the 100 soft cap). Files: `quantum_heads/vqc_ref_j.py`,
`vqc_head_j.py`, `circuit_table_j.md`, `test_vqc_head_j.py` (31 passed: Qiskit equivalence f32/f64, binary and Gray,
gradients, cache, guards, routing of each branch). Mutations caught: B/C swapped (22 failed), feature shift reversed
(23), CX pair swapped (19), role rotation reversed (21).
- Registered `vqc-nav-id` = VQCHeadJ(scale_init=0.05, angle_std=0.05, wf_offset=pi/2, action_code="gray"). scale 0.05
  (not I's 0.1) because N enters 9 times: on-policy init entropy, 5 seeds, J 0.1 -> 1.86-1.97, J 0.05 -> 1.97-2.03,
  I 0.1 -> 2.02-2.05 (ln 8 = 2.079).
- Sweep `nav-id-hlr2e-3-s{0..4}-7.5k-cpu-outer` (prefix navcmp), head_lr 2e-3, same config as the reference; compare
  with `optim-id-gray-hlr2e-3` (same action code, circuit effect only) and `optim-id-hlr2e-3`.
- **Result (navcmp):** mean of rounds 41-50 per seed 3.13 0.51 0.16 4.20 0.86 (mean 1.77; gray I 2.78, I 3.29);
  first round >= 3: 42 - - 33 -; final entropy 1.58-1.70 (unchanged). Paired vs gray I: -0.73 -3.28 -1.67 +1.46 -0.80.
  Mean curve below gray I at every checkpoint (rounds 10/20/30/40/50: -0.76 0.56 0.93 1.66 2.25 vs -0.68 1.03 1.94
  2.35 2.89). Bimodal (2 seeds fine, 3 fail), as I with head_lr 7e-3: possibly too large a step for J at 2e-3.
- Across every VQC variant of section 11 (lr, init, scale_lr, momentum, Gray, circuit J) the final entropy stays
  1.58-1.78, best seeds included (h128 0.87-1.31). In the distillation of section 10 the HR3 students also stayed at
  max-prob ~0.40 under a supervised target: concentration may be hard for these circuits, not only in RL.

### 11.7 Distillation of the RL heads from h128 (2026-10-13): capacity is enough for return, RL is the gap
`distill.py` extended (`--teacher-head`, `--reference`, heads vqc-optim-id(-gray), vqc-nav-id, mlp128, concentration
metrics). Teacher `h128-rms-s3` model_latest (val entropy 1.151, max-prob median 0.516, states with max-prob > 0.5:
0.541, collect return 5.23); 20k/5k states; Adam, 100 epochs, 3 seeds, head lr 3e-3 / 1e-2 (1e-2 best everywhere).
Run dir `quantum_heads/analysis/runs/distill_concentration` (summary.md, results.csv, students/); logs
`runs_distill_concentration.log`, `runs_eval_distill_concentration.log` (500 test episodes, lr 1e-2 students).
| variant | head | val frac | entropy | max-prob > 0.5 | sample return (test) |
|---|---|---|---|---|---|
| finetune | mlp128 | 0.999 | 1.15 | 0.55 | 5.18 ± 0.01 |
| finetune | vqc-nav-id (J) | 0.854 ± 0.008 | 1.41 | 0.32 | 4.97 ± 0.04 |
| finetune | vqc-optim-id (I) | 0.795 ± 0.007 | 1.47 | 0.25 | 4.76 ± 0.20 |
| finetune | vqc-optim-id-gray | 0.782 ± 0.034 | 1.47 | 0.24 | 3.85 ± 0.95 |
| frozen | mlp128 | 0.999 | 1.15 | 0.54 | 5.16 ± 0.04 |
| frozen | vqc-nav-id (J) | 0.752 ± 0.015 | 1.55 | 0.15 | 4.53 ± 0.21 |
| frozen | vqc-optim-id (I) | 0.647 ± 0.015 | 1.61 | 0.11 | 3.01 ± 0.27 |
| frozen | vqc-optim-id-gray | 0.660 ± 0.008 | 1.63 | 0.08 | 3.55 ± 0.31 |
- The pre-registered concentration criterion says "structural limit" for I (45% of the teacher's concentrated states)
  and "intermediate" for J (59%), but the return says otherwise: distilled I reaches 4.76 and J 4.97 (96% of the
  teacher) with flatter policies. Concentration was a misleading proxy: a flatter policy with the right ordering of
  actions is enough for the target (>= 4.0).
- Capacity is not the bottleneck for the success criterion of section 11: RL reaches 3.29 (I) and 1.77 (J) where the
  same circuits can hold 4.76 / 4.97. The gap is the RL dynamics (and, frozen vs finetune, the encoder adapting to
  the circuit: J frozen 4.53, I frozen 3.01).
- J > I in every distillation metric, the opposite of RL: J's RL failure (3 of 5 seeds) is an optimization problem.
- Gray: no gain (finetune worse and more variable, 3.85 ± 0.95).
- Relevant to the parked thesis idea (distil a classical policy into the smallest VQC head): 117 parameters, 96% of
  the teacher's return.

### 11.8 Warm start from the distilled students (2026-10-13, prefix `warmstart`)
`parallel_adapter.py` option `--set init_from=PATH` (full Policy state_dict, relative to the project). Inits built by
session scratchpad `make_warmstart_inits.py` into `quantum_heads/analysis/runs/warmstart_init/<head>_s<seed>.pt`:
encoder + head of the finetune lr 1e-2 student, critic of h128-rms-s3. Same RL config as the reference (head_lr 2e-3).
| run | round 1 | minimum (round) | round 17 | mean rounds 41-50 | from scratch, same seed | final entropy |
|---|---|---|---|---|---|---|
| J s0 | 4.85 | 4.17 (14) | 4.92 | 5.25 | 3.13 | 1.40 |
| J s1 | 4.72 | 3.56 (15) | 4.70 | 4.85 | 0.51 | 1.52 |
| J s2 | 4.73 | 2.08 (18) | 2.58 | 4.91 | 0.16 | 1.48 |
| I s0 | 3.18 | 2.10 (5) | 4.61 | 5.13 | 2.00 | 1.52 |
| I s1 | 4.04 | -1.45 (4) | 0.64 | 3.47 | 3.60 | 1.68 |
(I s2 removed from the queue by the user.)
- Once found, the solution holds in the second half of training (J 4.85-5.25, above h128's 4.32 from scratch): the
  main problem is reaching it from scratch. All dips are in rounds 1-18, while the entropy bonus is on (0.02 -> 0.001
  by update 2500) and the critic adapts to the student's encoder; I s1 collapsed and relearned to its from-scratch
  level (3.47 vs 3.60).
- Next (user's choice): the entropy schedule. Proposed: causal test (warm start I s1 and J s2 with entropy_start =
  0.001) and from-scratch sweep with entropy_start = 0.005 for I, J and h128 (shared loop parameter).

### 11.9 Causal test: the entropy bonus does not cause the warm-start dips (2026-10-13, prefix `entcmp`)
Same init, seed and machine (mac) as the dipping warm-start runs, entropy coefficient constant at 0.001
(`entropy_start=0.001`):
| run | rounds 1-5 | minimum (round) | round 17 | mean rounds 41-50 | value loss rounds 1-5 |
|---|---|---|---|---|---|
| I s1, bonus 0.02 -> 0.001 | 4.04 2.62 1.72 -1.45 -0.61 | -1.45 (4) | 0.64 | 3.47 | 2.04 2.36 2.52 2.23 1.73 |
| I s1, constant 0.001 | 4.30 2.44 1.03 -0.25 -1.36 | -2.00 (25) | -0.72 | 2.22 | 1.79 2.23 2.73 2.41 1.93 |
| J s2, bonus 0.02 -> 0.001 | 4.73 4.39 4.89 4.64 4.79 | 2.08 (18) | 2.58 | 4.91 | 1.66 1.59 1.50 1.51 1.57 |
| J s2, constant 0.001 | 4.66 4.63 4.63 4.46 4.71 | 1.12 (12) | 2.38 | 4.81 | 1.63 1.76 1.63 1.61 1.69 |
- Without the bonus the dips are the same or deeper: the bonus is not the cause. Entropy still rises in the first
  rounds without it (I s1: 1.71 at round 5), so the policy gradient itself flattens the policy. Remaining suspects:
  the critic taken from the teacher but fed by the student's encoder (value loss of I s1 rises 1.8 -> 2.7 in rounds
  1-3), and the step size of A2C + outer step on a good policy.

### 11.10 From-scratch sweep with entropy_start = 0.005 (2026-10-13, prefix `entcmp`)
Mean of eval rounds 41-50 per seed (first round >= 3, final entropy); reference = same config with entropy_start 0.02.
| head | s0 | s1 | s2 | s3 | s4 | mean | reference mean | paired diffs |
|---|---|---|---|---|---|---|---|---|
| I (optim-id hlr2e-3) | 2.80 (46, 1.74) | 1.43 (-, 1.68) | 3.91 (29, 1.61) | 1.93 (-, 1.70) | 1.72 (-, 1.78) | 2.36 | 3.29 | +0.80 -2.17 +0.76 -2.54 -1.49 |
| J (nav-id) | 4.25 (36, 1.62) | 1.58 (-, 1.81) | 1.98 (-, 1.69) | 2.86 (39, 1.59) | 2.86 (37, 1.67) | 2.71 | 1.77 | +1.12 +1.07 +1.82 -1.34 +2.00 |
| h128 (split_init 0) | 5.01 (21, 0.95) | 2.44 (-, 1.14) | 4.78 (22, 1.20) | 4.97 (18, 1.15) | 1.25 (20, 1.18) | 3.69 | 4.32 | +0.47 -0.71 +0.48 -0.25 -3.13 |
Mean curves (rounds 10/20/30/40/50): I -1.04 0.05 1.02 1.38 2.84 (ref -0.73 0.68 1.85 3.07 3.43); J -0.37 0.03 1.89
2.93 2.90 (ref -0.76 0.56 0.93 1.66 2.25); h128 -0.13 2.35 2.12 3.17 4.12 (ref 0.17 1.88 2.79 3.86 4.64).
- No consistent effect: I worse (-0.93), J better on 4/5 seeds (+0.93, t ~ 1.6, not significant), h128 worse
  (-0.63, one late collapse, s4 reached 3 at round 20 then fell). VQC final entropy unchanged (1.59-1.81). The entropy
  schedule is not the lever; within the noise of 5 seeds (spread ~2 points per config).
- Not run on matching machines for every seed, so pairs are not step-for-step identical.

### 11.11 Causal test: critic warmup delays but does not prevent the dips (2026-10-13, prefix `criticcmp`)
`parallel_adapter.py` option `--set critic_warmup=N`: for the first N updates encoder and head get no gradient (policy
fixed, only the critic learns); a parameter frozen by design stays frozen (flag stored on the parameter: the runtime
first steps a throwaway model; a first version keyed by id() kept the policy frozen for the whole run, runs
`ws-*-cw600-*` stopped and discarded). Runs `ws-*-cw600v2-*`: warm start, N = 600 (rounds 1-4), same seed/machine.
With the policy fixed the students score 4.99 (I s1) and 5.21 (J s2) on the eval seeds; value loss falls during the
warmup (I 1.74 -> 1.46).
| run | rounds 5-8 | minimum from round 5 (round) | round 17 | round 30 | mean rounds 41-50 | final entropy |
|---|---|---|---|---|---|---|
| I s1 original | -0.61 0.01 0.05 -0.43 | -1.30 (22) | 0.64 | 1.36 | 3.47 | 1.68 |
| I s1 no entropy bonus | -1.36 -0.16 0.68 -0.20 | -2.00 (25) | -0.72 | -0.81 | 2.22 | 1.74 |
| I s1 critic warmup | 5.14 4.89 4.64 4.23 | -1.31 (20) | -0.73 | 3.02 | 2.83 | 1.73 |
| J s2 original | 4.79 4.87 4.79 5.04 | 2.08 (18) | 2.58 | 4.40 | 4.91 | 1.48 |
| J s2 no entropy bonus | 4.71 4.69 4.13 3.10 | 1.12 (12) | 2.38 | 3.69 | 4.81 | 1.50 |
| J s2 critic warmup | 4.85 4.94 5.05 5.12 | 2.53 (12) | 3.21 | 4.47 | 4.89 | 1.47 |
- With an adapted critic the collapse of I s1 is delayed (5.14 -> 4.23 over rounds 5-8, then down to -1.31 at round 20),
  not prevented; J s2 dips to 2.53 and recovers as before. The critic mismatch is not the main cause: the A2C updates
  themselves (head_lr 2e-3 + outer step) drift a good VQC policy away, flattening it (entropy rises).
- Remaining suspects: update size (head_lr, outer Nesterov step) -> cheap diagnostics: warm start with head_lr 5e-4;
  warm start without the outer step. Then, if confirmed, a trust-region update (PPO) for both heads.
- **Correction (user's question):** the warmup test only excludes the *initial* critic mismatch, not value problems
  during training. In both runs the value loss spikes together with the collapse (I s1: 1.22 at round 6 -> 2.59 at
  round 10, return 4.89 -> 2.94; J s2: 1.45 at round 8 -> 2.85 at round 11, return 5.12 -> 2.74), entropy rising from
  round 5. Cause or consequence is not determined by these logs. A value mechanism specific to the VQC remains
  plausible: the critic shares the encoder and its gradient there is 5-8x the policy's (section 11.6 loss terms), so a
  value-loss spike reshapes the features, i.e. the data angles s * x the VQC depends on. Test: warm start with the
  critic detached from the encoder (value gradient does not reach it).

### 11.12 Causal test: the value gradient through the shared encoder causes the collapse (2026-10-13)
`Policy.detach_critic` (rl_train/policy.py, default False; test `test_detach_critic_keeps_the_value_loss_out_of_the_encoder`,
mutation caught) and adapter option `--set detach_critic=1`: the critic reads detached features, the value loss trains
only the critic, the encoder learns from the policy alone. Runs `ws-*-detc-*`: warm start, same seed/machine, no warmup.
| run | rounds 1-6 | minimum (round) | round 17 | round 30 | mean rounds 41-50 | final entropy | max value loss (round) |
|---|---|---|---|---|---|---|---|
| I s1 original | 4.04 2.62 1.72 -1.45 -0.61 0.01 | -1.45 (4) | 0.64 | 1.36 | 3.47 | 1.68 | 2.52 (3) |
| I s1 critic warmup | 4.99 x4, 5.14 4.89 | -1.31 (20) | -0.73 | 3.02 | 2.83 | 1.73 | 2.59 (10) |
| **I s1 detached critic** | 4.88 4.99 4.80 5.22 5.22 5.25 | 4.80 (3) | 5.44 | 5.41 | **5.48** | **1.28** | 1.58 (1) |
| J s2 original | 4.73 4.39 4.89 4.64 4.79 4.87 | 2.08 (18) | 2.58 | 4.40 | 4.91 | 1.48 | 2.62 (13) |
| J s2 critic warmup | 5.21 x4, 4.85 4.94 | 2.53 (12) | 3.21 | 4.47 | 4.89 | 1.47 | 2.85 (11) |
| **J s2 detached critic** | 5.10 4.92 5.10 5.14 5.14 5.13 | 4.92 (2) | 5.29 | 5.43 | **5.49** | **1.24** | 1.59 (1) |
- With the critic detached there is no dip at all: both runs improve steadily to ~5.5 (above h128 from scratch, 4.32)
  and the VQC policy concentrates (final entropy 1.24-1.28, h128 1.14-1.31); no value-loss spike.
- Cause: the critic's gradient into the shared encoder (5-8x the policy's) reshapes the features, i.e. the data
  angles of the circuit, and destroys the VQC policy. This is the first lever of section 11 that moves the entropy.
- Next: from scratch with detach_critic=1 for I, J and h128 (shared architecture change, 5 seeds each).

### 11.13 From scratch with the critic detached (2026-10-13, prefix `detccmp`)
Same configs as the references, `--set detach_critic=1` for every head (h128 with split_init=0 like its reference).
Mean of eval rounds 41-50 per seed (first round >= 3, final entropy):
| head | s0 | s1 | s2 | s3 | s4 | mean | reference | paired diffs |
|---|---|---|---|---|---|---|---|---|
| J (nav-id) | 3.88 (33, 1.52) | 3.51 (41, 1.53) | 4.98 (24, 1.41) | 5.31 (17, 1.36) | 2.97 (39, 1.58) | **4.13** | 1.77 | +0.75 +3.00 +4.82 +1.11 +2.11 |
| I (optim-id) | 2.53 (48, 1.43) | 3.54 (38, 1.66) | 1.02 (-, 1.62) | 1.73 (-, 1.51) | 2.83 (39, 1.56) | 2.33 | 3.29 | +0.53 -0.06 -2.13 -2.74 -0.38 |
| h128 | 5.48 (16, 0.37) | 4.88 (25, 0.55) | 5.25 (13, 0.47) | 5.35 (15, 0.36) | 5.53 (13, 0.36) | **5.30** | 4.32 | +0.94 +1.73 +0.95 +0.13 +1.15 |
Mean curves (rounds 5/10/15/20/30/40/50): J -> 0.15 0.28 0.55 1.49 2.66 3.95 4.65 (ref -1.25 -0.76 0.18 0.56 0.93
1.66 2.25); I -> -0.23 -0.36 -0.25 0.25 1.01 1.88 2.92 (ref ... 3.43); h128 -> -0.08 0.92 3.09 4.05 4.84 5.31 5.40
(ref -0.82 0.17 0.68 1.88 2.79 3.86 4.64).
- J improves on 5/5 seeds (+2.36 on average) and reaches the section 11 target in the mean (4.13 >= 4.0), but with
  seed std ~0.9 (SE ~0.4) and two seeds below 4: borderline, more seeds needed to claim it.
- h128 also improves on 5/5 (+1.0) and converges much faster (return >= 3 at rounds 13-25), with a far more peaked
  policy (entropy 0.36-0.55): detaching the critic is a general improvement of the setup, not VQC-specific. The second
  criterion (VQC faster than the MLP) is not met: h128 is ahead at every checkpoint.
- I gets worse (-0.96): with the navigator on one qubit per block its policy gradient into the encoder is weaker, and
  without the value signal the encoder learns too little (hypothesis, not tested). J sends the navigator through all
  qubits and benefits.
- `detach_critic` is now an `A2CConfig` field (int, default 0; `train.py --detach-critic 1`), applied to the model by
  `A2CTrainer.__init__` and by `parallel_adapter.build_model` (from the run config); test
  `test_detach_critic_config_reaches_the_model`. Seeds 5-9 of J and h128 with detach_critic queued to settle J >= 4.0.

### 11.14 Ten seeds with the critic detached (2026-10-13, prefix `detccmp`, seeds 5-9 added)
Mean of eval rounds 41-50 per seed (first round >= 3, final entropy):
- J (nav-id, head_lr 2e-3): 3.88 (33) 3.51 (41) 4.98 (24) 5.31 (17) 2.97 (39) 3.94 (39) 2.82 (43) 3.45 (41) 4.09 (34)
  5.26 (16) -> **mean 4.02, sd 0.90, SE 0.28** (mean - 2 SE = 3.45); 4/10 seeds >= 4; median first round >= 3: 36.5;
  final entropy 1.25-1.58. Seeds 5-9 alone: 3.91.
- h128: 5.48 4.88 5.25 5.35 5.53 5.51 4.10 5.01 5.49 4.55 -> **mean 5.11, sd 0.48, SE 0.15**; 10/10 >= 4; median first
  round >= 3: 17; entropy 0.36-0.55.
- Paired J - h128: -1.09 (sd 0.95, t -3.6).
- Mean curves (rounds 10/15/20/25/30/40/50): J 0.47 0.85 1.75 2.04 2.63 3.67 4.45; h128 1.13 2.75 3.31 4.14 4.61 5.02 5.20.
- Verdict against section 11: the mean sits at the 4.0 threshold (4.02) but not solidly (SE 0.28, 6 of 10 seeds below
  4); the VQC is not faster than the MLP (about 20 rounds behind to reach 3). J is still rising at round 50 (+0.8 over
  the last 10 rounds): speed, not capacity or stability, is now the limit. Next candidate: J head_lr 3e-3 / 4e-3 with
  detach_critic (step sizes that failed before were tested with the shared critic).

### 11.15 J head_lr with the critic detached (2026-10-13, prefix `navlr`) - partial
- 3e-3: s0-s2 finished, s3-s4 stopped (grid changed to log spacing, 3e-3 vs 4e-3 too close to resolve with 5 seeds).
- 7e-3: stopped by the user as too unstable (s0 41 rounds, last 5: 0.14 -0.43 -0.18 0.09 1.17; s1 32 rounds, around 0;
  s2 15 rounds, down to -2.04; s3 13 rounds; s4 removed from the queue). Even with the critic detached, 7e-3 is above
  the stability limit of J.
- Mean of eval rounds 41-50 per seed (first round >= 3):
  | head_lr | s0 | s1 | s2 | s3 | s4 | mean |
  |---|---|---|---|---|---|---|
  | 2e-3 | 3.88 (33) | 3.51 (41) | 4.98 (24) | 5.31 (17) | 2.97 (39) | 4.13 |
  | 3e-3 | 3.66 (35) | 2.59 (43) | 2.27 (46) | stopped | stopped | 2.84 (3 seeds; 2e-3 on them: 4.12) |
  | 4e-3 | 5.09 (31) | 5.25 (25) | 4.89 (33) | 4.88 (23) | 1.94 (-) | 4.41 |
  Mean curves seeds 0-4 (rounds 10/15/20/25/30/40/50): 4e-3 0.55 0.87 0.65 2.32 3.10 4.71 5.20; 2e-3 0.28 0.55 1.49
  1.94 2.66 3.95 4.65.
- 4e-3 vs 2e-3 paired: +1.21 +1.74 -0.09 -0.43 -1.03 (mean +0.28, t ~0.55): not resolved. 4 of 5 seeds at 4.9-5.3 but
  one fails (1.94); 3e-3 lower than both on its 3 seeds: the ordering is not monotonic, seed noise dominates. No
  evidence that a larger head_lr makes J faster; 7e-3 is unstable.

## 12. Confirmed intent (2026-10-14, interview, explicit yes): a regime where the VQC pays off (generalization)
- **Outcome:** show a regime where the VQC beats the MLP: training on a fixed pool of K layouts, evaluation on
  unseen layouts. Candidate reason X: the limited capacity and the angle encoding of the VQC reduce overfitting when
  training layouts are few.
- **Why:** the advisor asked for "a regime where the MLP struggles and the VQC emerges, for reason X", not for
  parity. Matching the MLP on infinite layouts (sections 9-11) was a self-imposed goal that kept moving (every fix
  also helped the MLP). Making the task harder (env v9-v11) hurts both models, the VQC more: wrong axis.
- **Success:** absolute return on test layouts as a function of K (e.g. 1, 5, 20, 100, inf); at least one K where J
  (`vqc-nav-id`, `detach_critic=1`) beats the MLP significantly over several seeds. Baselines: h128 and an MLP head
  with matched parameter count. A smaller train-test gap alone is NOT success (VQC underfitting would explain it).
- **Constraints:** less than one month to the main results; circuit J unchanged; Born readout; env = v8 configuration
  (window 5, 30 obstacles, CNN; user decision 2026-10-14, see docs/design/SPEC-generalization-pilot.md). First a
  pilot of a few days: does the MLP overfit with small K (train >> test)? Risk: even with the 5x5 window the observation
  may say little about which layout the agent is in, so neither model may overfit.
- **Plan B:** if the pilot shows no separation, the thesis becomes diagnostic (why the VQC struggles in RL: critic
  gradient through the shared encoder, distillation shows capacity suffices, residual ~1.1 gap to h128).
- **Out of scope:** closing the gap on infinite layouts; new ansatz; new env versions; non-Born readouts; noise
  robustness and parameter efficiency as the main thesis.

### 12.1 Pilot results (2026-10-14, prefix `gen`, spec docs/design/SPEC-generalization-pilot.md)
Env v13 with `window=5, n_obstacles=30` (= v8), `perception=cnn`, `detach_critic=1`, split_init=1, 7.5k updates, seeds
0-2 (`env_seed = layout_pool_seed = seed`, so every head sees the same pool per seed). J = `vqc-nav-id` head_lr 2e-3.
Test = `sample/return_mean` on 100 fresh layouts (eval seeds 10000-10099, verified disjoint from every pool); pool = the
same eval on the K training layouts. Mean of eval rounds 41-50. Control: `gen-nav-id-kinf-s0` is identical round by
round to section 11.14's `nav-id-hlr2e-3-detc-s0` (3.88), so J K=inf reuses s1-s2 from there (3.51, 4.98).
| head (params) | K | test (mean +- SE) | per seed | pool | gap | overfit seeds | entropy |
|---|---|---|---|---|---|---|---|
| h128 (2696) | 5 | -4.96 +- 0.94 | -3.12 -6.22 -5.54 | 4.91 | 9.87 | 3/3 | 0.00-0.04 |
| h128 | 25 | 1.62 +- 0.06 | | 5.63 | 4.01 | 3/3 | 0.02-0.03 |
| h128 | 200 | 4.69 +- 0.43 | 5.16 3.83 5.10 | 4.69 | -0.01 | 0/3 | 0.64-0.71 |
| h128 | inf | 4.90 +- 0.27 | 4.42 4.90 5.37 | - | - | - | 0.32-0.62 |
| h5 (113) | 5 | -2.99 +- 0.90 | -1.21 -3.70 -4.07 | 5.25 | 8.25 | 3/3 | 0.05-1.01 |
| h5 | 25 | 2.09 +- 0.63 | 1.27 3.33 1.67 | 4.27 | 2.18 | 3/3 | 0.83-1.27 |
| h5 | 200 | 2.98 +- 0.37 | 3.71 2.55 2.68 | 2.95 | -0.03 | 0/3 | 1.24-1.33 |
| h5 | inf | 3.79 +- 0.35 | 3.16 3.83 4.37 | - | - | - | 1.11-1.44 |
| J (117) | 5 | -0.73 +- 0.57 | 0.05 -1.85 -0.40 | 5.41 | 6.14 | 3/3 | 0.54-1.21 |
| J | 25 | **4.71 +- 0.25** | 4.64 5.18 4.32 | 5.02 | 0.31 | 0/3 | 1.39-1.46 |
| J | 200 | 3.79 +- 0.71 | 2.82 3.38 5.18 | 3.82 | 0.02 | 0/3 | 1.32-1.58 |
| J | inf | 4.12 +- 0.44 | 3.88 3.51 4.98 | - | - | - | 1.46-1.56 |
Paired test differences (per seed): J - h5: K=5 +2.26 +- 0.73 (3/3 > 0), K=25 +2.62 +- 0.44 (3/3), K=200 +0.82 +- 0.98
(2/3), K=inf +0.34 +- 0.33 (2/3). J - h128: K=5 +4.23 +- 0.57 (3/3), K=25 +3.09 +- 0.23 (3/3), K=200 -0.90 +- 0.73 (1/3),
K=inf -0.78 +- 0.31 (0/3).
- **Decision: GO** (spec rule: MLP overfit at K=5 on 3/3 seeds for both MLPs; J test >= h5 and >= h128 - 0.5 at K=5 and
  K=25). With few layouts (K=5, 25) J generalizes clearly better than both MLPs, also than the parameter-matched h5;
  at K=200 / inf the MLPs no longer overfit and h128 is ahead again (as in section 11).
- J memorizes 5 layouts (pool 5.41) but not 25 (gap 0.31; test 4.71 = its K=inf level): low memorization capacity
  forces a general strategy. The MLPs still overfit at K=25 (gap 4.0 / 2.2).
- Confound: entropy. MLP policies collapse at small K (h128 ~0), J stays stochastic, and `sample/*` is a stochastic
  eval. Weakened at K=25 (h5 s2 at entropy 1.27 scores 1.67 vs J 4.3-5.2 at ~1.4; no entropy-test correlation within
  h5) but not removed. Greedy returns were not in the logged columns.
- Caveats: 3 seeds; J K=200 has a large spread (2.82-5.18); non-monotonic J (K=25 above K=200/inf) within noise.
- Next (user, 2026-10-14): entropy-matched MLP control (adaptive entropy coefficient with a target entropy = J's
  measured entropy per K, recommended) and a Fourier-structure control (MLP on sin/cos features, VQC = truncated
  Fourier series) to pin down reason X; then the full study (10 seeds, finer K grid). Nothing started yet.

### 12.2 Entropy-matched MLP control (2026-10-14, prefix `ent`, spec docs/design/SPEC-entropy-matched-control.md)
h128 and h5 retrained with an adaptive entropy coefficient (`target_entropy`, `entropy_lr` 0.01: SGD on log alpha)
holding the training entropy at J's pilot entropy (K=25: 1.425, K=inf: 1.521); everything else as the pilot (same
seeds and pools). Off path verified unchanged (2-round rerun of gen-h128-k25-s0 identical). Calibration
(ent-h128-k25-s0): entropy 41-50 = 1.40, within 0.1 of target from round 5, coefficient never at its bounds.
Mean of rounds 41-50, J from the pilot:
| head | K | test (sample) | greedy (test) | pool | gap | entropy | test of the same head in the pilot |
|---|---|---|---|---|---|---|---|
| h128-ent | 25 | 4.37 +- 0.36 (4.70 4.75 3.66) | -0.43 (-0.27 0.43 -1.46) | 5.17 | 0.80 | 1.32-1.40 | 1.62 |
| h128-ent | inf | 4.53 +- 0.32 (4.44 4.02 5.13) | 1.01 (0.47 -1.13 3.69) | - | - | 1.47-1.55 | 4.90 |
| h5-ent | 25 | 2.13 +- 0.27 (2.35 2.45 1.59) | -4.77 (-4.77 -4.37 -5.17) | 3.85 | 1.73 | 1.26*-1.41 | 2.09 |
| h5-ent | inf | 2.73 +- 0.49 (3.71 2.28 2.20) | -4.96 (-4.04 -5.59 -5.24) | - | - | 1.23*-1.64 | 3.79 |
| J | 25 | 4.71 +- 0.25 | not logged | 5.02 | 0.31 | 1.39-1.46 | |
| J | inf | 4.12 +- 0.44 | not logged | - | - | 1.46-1.56 | |
(* outside the +-0.15 tolerance: h5 K=25 s1 1.26, h5 K=inf s2 1.23; h5 K=25 s0 is on target at 1.41 and scores 2.35.)
Paired test J - matched MLP: K=25 J - h128-ent +0.34 +- 0.21 (2/3 > 0; pilot gap was +3.09), J - h5-ent +2.59 +- 0.15
(3/3; pilot gap +2.62). K=inf: J - h128-ent -0.40 +- 0.13, J - h5-ent +1.39 +- 0.76.
- **Split result.** Large MLP: entropy explains most of J's advantage (h128 at J's entropy goes 1.62 -> 4.37 at K=25,
  89% of the gap closed, within 0.5 of J). Parameter-matched MLP: entropy explains none of it (h5 at J's entropy:
  2.09 -> 2.13; still overfits, gap 1.73; J +2.59 on 3/3 seeds).
- So at ~115 parameters J generalizes like a 2.7k-parameter MLP that has been regularized to the right entropy, and the
  same-size MLP cannot get there by entropy alone. Candidate reason X: per-parameter efficiency of the VQC policy in
  the low-data regime, given a stochastic (Born) policy; plus J reaching a good entropy without a tuned target.
- Greedy: the MLPs' argmax policies are poor on fresh layouts (h128-ent -0.43, h5-ent -4.8 at K=25): their test
  return comes from sampling. J's greedy return is unknown (not in the pilot logs; needs an offline eval of the
  checkpoints, permission pending).
- Forcing entropy costs at K=inf: h128 4.90 -> 4.53, h5 3.79 -> 2.73.
- Caveats: 3 seeds; per-round entropy is a single-batch value (noisy); two h5 runs off target.

### 12.3 Greedy returns (2026-10-14, from checkpoints/<run>/metrics.jsonl via scripts/summarize_eval_metrics.py)
metrics.jsonl has every eval metric per round (the round logs only the --show columns). Sample values match the logs.
Test return, sample / greedy, mean of rounds 41-50 over seeds 0-2:
| head | K=5 | K=25 | K=200 | K=inf |
|---|---|---|---|---|
| J | -0.73 / -5.94 | 4.71 / -3.71 | 3.79 / -4.40 | 4.12 / -4.25 |
| h5 | -2.99 / -5.44 | 2.09 / -4.71 | 2.98 / -4.91 | 3.79 / -5.04 |
| h5-ent | | 2.13 / -4.77 | | 2.73 / -4.96 |
| h128 | -4.96 / -5.42 | 1.62 / 0.79 | 4.69 / 3.19 | 4.90 / 4.05 |
| h128-ent | | 4.37 / -0.43 | | 4.53 / 1.01 |
- J's greedy (argmax) policy is poor at every K, even at K=inf: its competence lies in the shape of the action
  distribution (an oriented random walk), not in its mode. Same for h5. Only h128 learns a good deterministic policy,
  and only with many layouts.
- Not a flaw of the eval: in a POMDP with limited memory the best stationary policy can be stochastic (Singh,
  Jaakkola, Jordan 1994, to verify before citing). Sampling is the right eval and is the definition of a Born-readout
  policy. Greedy is therefore not the discriminator between "general strategy" and "just stochastic".
- The decisive comparison stays 12.2: same size and same entropy (h5-ent), J +2.59 sample (+1.06 greedy) on 3/3 at K=25.
- Open: is h5's 5-unit bottleneck the issue rather than being classical? Proposed next controls at J's entropy, ~115
  params: linear head (12->8 softmax, 104 params) and a Fourier-feature head (sin/cos of the inputs). K=25 and inf.

### 12.4 Small classical heads: linear (96) and Fourier (120) (2026-10-14, spec docs/design/SPEC-small-classical-controls.md)
`shared` = Linear(12 -> 8) no bias; `fourier4` = 4 learned frequency vectors -> sin/cos -> Linear(8 -> 8) (new,
classic_heads/fourier.py). Pilot setup; `ent-*` at J's entropy (adaptive coefficient), `gen-*` with the pilot schedule.
Test (sample) per seed, mean of rounds 41-50, K=25 (J: 4.64 5.18 4.32 = 4.71):
| head | test per seed | mean +- SE | pool | entropy | J - head (paired) |
|---|---|---|---|---|---|
| lin-ent | 2.76 2.68 2.70 | 2.71 +- 0.02 | 4.06 | 1.20 1.15 1.64 (all off +-0.15) | +2.00 +- 0.26 (3/3) |
| lin | 2.51 2.41 3.06 | 2.66 +- 0.20 | 3.70 | 1.36 1.13 1.09 | +2.05 +- 0.44 (3/3) |
| four-ent | 1.28 3.70 1.05 | 2.01 +- 0.84 | 3.75 | 1.31 1.53 1.23 | +2.70 +- 0.61 (3/3) |
| four | **4.77 4.83** 1.64 | 3.75 +- 1.06 | 4.81 | **0.60 0.46** 1.37 | +0.96 +- 0.87 (-0.13 +0.34 +2.68) |
K=inf: lin-ent 3.26 (J +0.86), four-ent 2.87 (J +1.26).
- Pre-registered rule (entropy-matched, K=25): J - lin-ent and J - four-ent both >= 1.0 on 3/3 -> "specific to the
  circuit". BUT the rule assumed entropy matching is neutral, and it is not for Fourier: unconstrained, two Fourier
  seeds reach J's level (4.77, 4.83 vs J 4.64, 5.18) at LOW entropy (0.46-0.60, pool 5.6, gap ~0.8); forcing J's
  entropy drops them to 1.28 / 3.70. The third seed fails in both conditions.
- Linear: ~2.7 whatever the entropy, ~2 below J on 3/3 seeds -> h5's bottleneck is not the explanation.
- Reading: the periodic structure is not ruled out; it may reach J on most seeds, less reliably (1/3 failures), and
  at a much lower entropy, i.e. by a different route than J (not "stochasticity"). Not resolved with 3 seeds.
- Caveats: the adaptive coefficient tracks the linear head poorly (all three off target); J keeps head_lr 2e-3 vs 7e-4
  for the classical heads.
- Next candidate: Fourier unconstrained with more seeds (and K=5 / inf) to resolve J vs Fourier; the head_lr asymmetry.

### 12.5 J vs Fourier head, K=25, 10 seeds (2026-10-14, spec docs/design/SPEC-j-vs-fourier.md)
No entropy constraint (pilot coefficient schedule for all arms), same pools per seed. Test = sample return on fresh
layouts, mean of rounds 41-50:
| arm | test per seed (s0..s9) | mean +- SE | sd | failures (< 3.0) | overfit (gap >= 1) | entropy |
|---|---|---|---|---|---|---|
| J (vqc-nav-id, head_lr 2e-3) | 4.64 5.18 4.32 3.81 4.87 5.27 0.57 4.31 2.33 5.18 | 4.05 +- 0.47 | 1.50 | 2/10 | 2/10 | 1.19-1.48 |
| Fourier (fourier4, lr 7e-4) | 4.77 4.83 1.64 2.79 3.19 4.84 3.95 4.39 2.98 3.41 | 3.68 +- 0.34 | 1.07 | 3/10 | 6/10 | 0.46-1.37 |
| Fourier, head_lr 2e-3 | 1.18 3.91 2.07 4.76 4.70 1.65 -1.42 4.84 3.81 2.06 | 2.76 +- 0.64 | 2.02 | 5/10 | 7/10 | 0.66-1.18 |
Paired: J - Fourier = +0.37 +- 0.53, t 0.70 (95% CI -0.83 .. +1.57); J - Fourier lr 2e-3 = +1.29 +- 0.60, t 2.16.
- **Decision (pre-registered rule): EQUIVALENT.** F* = Fourier at the default lr (3.68); |t| < 2.26. No secondary
  reliability claim: J fails on 2/10 seeds vs 3/10 (the rule asked for clearly fewer, e.g. 0-1 vs >= 3).
- Not pre-registered, observation only: J overfits on fewer seeds (gap >= 1: 2/10 vs 6/10), mean gap 0.84 vs 1.13.
- head_lr 2e-3 hurts the Fourier head (2.76, 5/10 failures): the lr asymmetry favoured the classical head, not J.
- J's K=25 mean drops from 4.71 (s0-s2) to 4.05 over 10 seeds (s6 0.57, s8 2.33), consistent with its known seed
  variance (section 11).
- Reading: at K=25 and equal size, a classical head with learned Fourier features generalizes as well as J. The
  advantage of J over same-size MLPs (h5, linear) is explained by the periodic (Fourier) structure, which the VQC is
  one way to obtain; no evidence of a circuit-specific advantage. Consistent with the view of re-uploading VQCs as
  Fourier series and with their classical approximation by Fourier features (Schuld et al. 2021; Landman et al. 2022;
  to verify before citing). A difference up to ~1.5 in J's favour is not excluded by 10 seeds.

### 12.6 Ansatz K (`VQCHeadK`, head `vqc-k`), K=25, 10 seeds (2026-10-14, prefix `gen`)
User's design: 4 qubits, q0-q2 measured (Born, Gray code), q3 an ancilla that is only a CX control. Three blocks r;
on every qubit i three data gates RY RZ RY, each with the navigator plus a rotating second branch:
r = 1: N, N+M, N+G; r = 2: N+G, N, N+M; r = 3: N+M, N+G, N (angle d_r[3i+k] N_i + c_r[2i+j] X_i, independent weights),
then RZ RY RZ (u_r), CX (3,2) (3,1) (3,0) (2,1) (2,0) (1,0), then RZ RY RZ (v_r) on q0-q2 and RY RZ RY on q3; RY(wf)
on q0-q2 at the end. Gates with no effect on the output are dropped (last RZ(u3) of q3, the q3 rotation after the last
CX chain). 131 parameters (d 36, c 24, u 35, v 33, wf 3). Files: `quantum_heads/vqc_ref_k.py`, `vqc_head_k.py`,
`circuit_table_k.md`, `test_vqc_head_k.py` (33 passed: Qiskit equivalence f32/f64, binary and Gray, gradients,
cache, guards, routing of each branch, the dropped gates have no effect in Qiskit). Mutations caught: M/G swapped in
block 2, gate positions in block 3, c weights swapped, navigator read from M, feature of the wrong qubit.
- Registered `vqc-k` = VQCHeadK(scale_init=0.05, angle_std=0.03, wf_offset=pi/2, action_code="gray"): J's init but
  angle_std 0.03, since with 0.05 the start was slightly less uniform (init entropy 2.019 mean, 2/20 seeds < 2.0; 1/20
  even with the data off, i.e. caused by the extra free angles). With 0.03: 2.018-2.038 over 50 seeds.
- An earlier K (data gates with one branch each, 107 params) ran as `gen-k-k25-s{0..4}` and was stopped at rounds
  25-38 by the user (wrong direction): not evaluated, not comparable.
- Runs `gen-k2-k25-s{0..9}-7.5k-cpu-outer`, same config as J's K=25 runs (head_lr 2e-3, detach_critic, pools per seed).

Test = sample return on fresh layouts, mean of rounds 41-50:
| arm | test per seed (s0..s9) | mean +- SE | pool | gap | failures (< 3.0) | overfit (gap >= 1) | entropy |
|---|---|---|---|---|---|---|---|
| K (vqc-k) | 2.17 3.99 5.09 2.53 4.57 3.84 4.10 3.33 3.76 1.16 | 3.45 +- 0.37 | 4.40 | 0.95 | 3/10 | 4/10 | 1.44 |
| J (vqc-nav-id) | 4.64 5.18 4.32 3.81 4.87 5.27 0.57 4.31 2.33 5.18 | 4.05 +- 0.47 | 4.88 | 0.84 | 2/10 | 2/10 | 1.38 |
| Fourier (fourier4) | 4.77 4.83 1.64 2.79 3.19 4.84 3.95 4.39 2.98 3.41 | 3.68 +- 0.34 | 4.81 | 1.13 | 3/10 | 6/10 | 1.06 |
Paired: K - J = -0.59 +- 0.67, t -0.89 (K lower on 7/10 seeds; s9 -4.01, s6 +3.52 dominate); K - Fourier = -0.22 +-
0.56, t -0.40.
- Reading: no gain from K. Not resolved against J (|t| < 2.26), but the point estimate is below J and K has the lowest
  pool return of the three (4.40): the extra navigator inputs, the ancilla and 14 more parameters do not help
  generalization at K=25.
- Caveats: K differs from J in the circuit AND in angle_std (0.03 vs 0.05); head_lr 2e-3 was tuned for J, not for K.

### 12.7 Full generalization study, part A: 10 seeds per cell (2026-10-14, spec docs/design/SPEC-generalization-full.md)
Reporting as in RL papers (Henderson et al. 2018; Colas et al. 2018; Agarwal et al. 2021): mean +- SE over 10 seeds,
comparisons as paired differences per seed (same seed = same pool) with their 95% CI; the CI excludes 0 -> better,
includes 0 -> "no difference detected" (with its width), never "equivalent". This supersedes the "EQUIVALENT" of 12.5:
J - Fourier at K=25 is "no difference detected, wide CI". (A first version of the spec used Holm / TOST decision rules;
rewritten at the user's request after the summary, no conclusion changed, see the spec's note.)
Test = sample return on 100 fresh layouts, mean of rounds 41-50 (`scripts/summarize_gen_pilot.py`); figure
`docs/figures/partA_curve_k.png` (`scripts/plot_gen_curve.py`, run with an interpreter that has matplotlib).

**Cells** (mean +- SE; gap = pool - test):
| head | K=5 | K=25 | K=200 | K=inf |
|---|---|---|---|---|
| J (vqc-nav-id) | -1.13 +- 0.37 | 4.05 +- 0.47 | 4.26 +- 0.30 | 4.02 +- 0.28 |
| Fourier (fourier4) | -5.10 +- 0.25 | 3.68 +- 0.34 | 4.33 +- 0.23 | 3.97 +- 0.30 |
| h5 (mlp5) | -3.79 +- 0.50 | 2.27 +- 0.28 | 3.73 +- 0.30 | 3.42 +- 0.53 |
| h128 (mlp128) | -5.33 +- 0.40 | 1.25 +- 0.46 | 5.02 +- 0.15 | 5.18 +- 0.13 |
| h5-ent / h128-ent | | 1.92 +- 0.38 / 4.29 +- 0.23 | | |
Gap at K=5: 6.6 (J) to 10.7 (Fourier, h128): every head overfits, nobody generalizes. K=200: gap ~0 for all.

**Paired differences J - X** (mean, 95% CI):
| X | K=5 | K=25 | K=200 | K=inf |
|---|---|---|---|---|
| h5 | +2.66 [1.82, 3.50] | +1.78 [0.58, 2.97] | +0.53 [-0.52, 1.58] | +0.60 [-0.78, 1.98] |
| h128 | +4.20 [3.44, 4.96] | +2.79 [1.40, 4.18] | -0.77 [-1.39, -0.14] | -1.16 [-1.81, -0.51] |
| Fourier | +3.97 [3.21, 4.73] | +0.37 [-0.82, 1.56] | -0.07 [-0.79, 0.64] | +0.05 [-1.13, 1.23] |
| h5-ent | | **+2.13 [0.82, 3.44]** | | |
| h128-ent | | -0.24 [-1.17, 0.70] | | |
Gap difference J - Fourier (A4): K=5 -4.05 [-4.85, -3.26]; K=25 -0.29 [-1.01, 0.42]; K=200 -0.01 [-0.15, 0.13].

- **A1 (K=25):** J beats the same-size MLP at the same entropy (h5-ent, CI above 0): the 12.2 result holds at 10
  seeds. Against the large MLP at the same entropy (h128-ent): no difference detected.
- **A2 (curve in K):** J's advantage over h5 is clear at K=5 and K=25 and not detected at K=200 / inf; the large MLP
  overtakes J from K=200 on (CI below 0). J and Fourier: no difference detected at K=25, 200, inf (narrow CI at 200,
  wide at 25 and inf); J ahead at K=5 (but see the entropy control below).
- **A4 (overfitting):** J's gap is smaller than Fourier's only at K=5, where it is the same fact as the test return.
- **Greedy** (argmax actions, from metrics.jsonl, `scripts/summarize_eval_metrics.py`, context only): every head
  collapses on fresh layouts (J -4.1 / -4.5 / -3.8 at K=25 / 200 / inf, Fourier -3.0 / -3.4 / -4.3, h5 < -3.7) except
  h128 at K >= 200 (3.9 / 4.3) and partly at K=25 (0.3); as in 12.3, sampling is the right evaluation in this POMDP.

**Reproducibility control (J at K=inf).** The reused runs `nav-id-hlr2e-3-detc-s*` (section 11.14) enter the K=inf
cell; control `gen-nav-id-kinf-s9` (nodo2) matched only round 1 and diverged from round 2. A 3-round rerun on the mac
with the current code matched the reference exactly (rounds 1-3, 4 digits): runs are bit-identical on one machine and
diverge across machines (float differences, amplified by RL); final test 5.26 vs 5.32. Reused runs valid.
J K=5 s6 / s9 were stopped on nodo1 (2.1 steps/s) and rerun as `...-r` on nodo2.

**Entropy at K=5.** Final entropies differ (Fourier 0.04, h128 0.01, h5 0.34, J 0.67), so J's K=5 advantage could be
stochasticity, as for h128 at K=25 in 12.2.
- Training-time control `ent-{four,h5}-k5` (target 0.674): **failed**. The adaptive coefficient oscillated (entropy
  1.00 / 0.95 mean, per seed 0.15-1.97) and the controls stopped fitting their pool (1.55 / 2.50 vs ~5.5). Not
  interpreted.
- Eval-time temperature (`quantum_heads/analysis/policy_equivalence.py temperature`): logits / T per run so that the
  entropy on its own visited test states equals J's (1.12; higher than J's training entropy 0.67, measured on pool
  states). Test: Fourier -5.24 -> -2.34 (T 3.6), h5 -3.91 -> -1.87 (T 4.2), h128 -5.51 -> -2.16 (T 15.5), J -1.27.
  J - X(T): Fourier +1.07 [-0.09, 2.23], h5 +0.60 [-0.47, 1.67], h128 +0.89 [-0.13, 1.91]: no difference detected.
- Reading: most of J's K=5 advantage (+3.97 over Fourier) is eval-time stochasticity: the other heads collapse to
  near-deterministic policies that fail on unseen layouts, J stays stochastic. A residual ~+1 (positive for all three
  heads) is not resolved by 10 seeds. Says nothing about entropy during training.

**A3: do J and Fourier compute the same kind of policy?** (offline, `model_latest.pt`, K=25 and 200;
`quantum_heads/analysis/policy_equivalence.py`, `cross_distill.py`; results in `quantum_heads/analysis/runs/a3/`)
- *A3.1 policy distance* (TV between action distributions on the visited test states of both runs):
  | pair | K=25 | K=200 |
  |---|---|---|
  | J vs J, different seeds | 0.54 [0.33, 0.68] | 0.49 [0.29, 0.63] |
  | F vs F, different seeds | 0.67 [0.55, 0.84] | 0.55 [0.35, 0.72] |
  | J vs F, same seed | 0.60 | 0.47 |
  | J vs F, different seeds | 0.61 | 0.53 |
  Head effect (J vs F different seeds - mean within-head) +0.007 / +0.008: J and Fourier differ as much as two seeds
  of the same head. The distances themselves are large: many different policies reach the same return. "Same seed"
  also means same encoder init (split_init=1) and env stream, not only the same pool.
- *A3.2 periodicity on the data* (rho = range of each data angle over the states / 2 pi; >= 0.5 = periodic regime):
  both heads have at least one periodic angle in 10/10 seeds at K=25 and 200, so reason X stays "Fourier structure".
  They use it differently: Fourier 4/4 projections periodic (median rho 0.5-0.9); J 2-8 of 36 gate angles periodic
  (median rho 0.1-0.2), yet J's action probabilities turn back more often along a single feature, each feature
  entering several gates. Sweeps of one encoder feature over its q1-q99 range (200 states, 8 action probabilities,
  mean over seeds; K=25 / K=200):
  | head | turning points | significant (stretch >= 0.02) | non-monotone amplitude |
  |---|---|---|---|
  | J | 0.65 / 0.62 | 0.39 / 0.40 | 0.049 / 0.055 |
  | Fourier | 0.38 / 0.37 | 0.27 / 0.25 | 0.047 / 0.038 |
  About 40% of J's plain turning points are tiny wiggles; J still has more significant reversals than Fourier, but
  the probability that goes back and forth is similar (0.05 per sweep, slightly more for J at K=200). Both heads
  are mostly monotone along a single feature with modest non-monotone structure.
- *A3.3 cross-distillation* (frozen teacher encoder, fresh student head trained on KL(teacher || student), 20k
  states of the teacher's rollouts on fresh layouts, 3 init seeds, lr chosen on val; mean over 10 teachers):
  | teacher <- student | K=25 frac | K=25 test (teacher) | K=200 frac | K=200 test (teacher) |
  |---|---|---|---|---|
  | F <- F | 0.986 | 3.81 (3.83) | 0.991 | 4.47 (4.55) |
  | F <- **J** | **0.810** | 3.87 (3.83) | **0.871** | 4.23 (4.55) |
  | F <- MLP64 | 0.995 | 3.82 | 0.996 | 4.52 |
  | J <- J | 0.956 | 4.16 (4.18) | 0.958 | 4.30 (4.43) |
  | J <- **F** | **0.881** | 4.05 (4.18) | **0.895** | 4.30 (4.43) |
  | J <- MLP64 | 0.967 | 4.21 | 0.971 | 4.36 |
  (frac = explained fraction 1 - KL(teacher || student) / KL(teacher || marginal) on val, SE 0.001-0.014.)
  lr control: the first run chose lr 3e-3 (grid edge) everywhere; adding 1e-2, it is chosen everywhere (edge again,
  no further values, as fixed in the spec), but the gain is small (+0.01 / +0.02 on the cross students: F <- J 0.794
  -> 0.810 and 0.863 -> 0.871): the students are near saturation. Reading (rule fixed before the control): J <- F
  stays 0.12-0.18 below the F <- F ceiling, F <- J 0.06-0.08 below J <- J -> the asymmetry is a property of the two
  classes, not an optimization limit. Fourier (and MLP64) represent J's policies better than J represents Fourier's;
  in return the difference hardly matters (every student within 0.3 of its teacher, J <- F at K=25 even equal).
**B2 noise robustness** (`scripts/eval_noise.py`, J K=25, 10 seeds, final weights): delta vs clean (4.18) -0.00 /
+0.04 / -0.15 / -0.79 at depolarizing p2 = 0.001 / 0.003 / 0.01 / 0.03 (p1 = p2 / 10); readout 2% +0.09; calibrated
fake_torino -0.07, fake_brisbane -0.29. Aer check of the effective noise model (s0, 200 states, 20k shots): TVD to a
full Aer simulation of the transpiled circuit 0.010 (torino) / 0.028 (brisbane), shot floor 0.007, vs 0.08 / 0.15 for
the clean model; the brisbane residual is likely idle T1/T2 (depth 204 vs 129), not in the effective model.
**B3 full Aer** (`scripts/eval_aer.py`, spec docs/design/SPEC-noise-robustness.md): every action of the 100 test episodes
from one shot of the transpiled circuit on `AerSimulator.from_backend` (per-qubit / per-pair errors, T1 / T2, readout),
10 seeds. Noiseless check first: transpiled circuit vs head TVD 0.0057 = shot floor, decoding correct.
| backend | full Aer (mean +- SE) | effective model | clean | full Aer - effective [CI95] | full Aer - clean [CI95] |
|---|---|---|---|---|---|
| fake_torino | 4.09 +- 0.38 | 4.11 | 4.18 | -0.03 [-0.22, 0.17] | -0.10 [-0.28, 0.09] |
| fake_brisbane | 3.80 +- 0.41 | 3.89 | 4.18 | -0.09 [-0.26, 0.07] | -0.38 [-0.62, -0.14] |
Per seed (torino / brisbane): 4.40/4.32 5.05/4.63 4.09/4.17 3.92/3.29 4.92/4.34 5.30/5.04 1.17/0.75 4.02/4.00 3.13/2.63
4.86/4.80. The effective model is adequate for the return (no difference detected against full Aer, narrow CI). On
torino no loss is detected; on brisbane J loses 0.38 (9%), still at the level of the noiseless Fourier head (3.83) and
far above h5 at J's entropy (2.06; references: `scripts/eval_clean_returns.py`, same weights and eval). Figure
`docs/figures/partB_noise.png` (`scripts/plot_noise.py`). Cost: 35-80 ms per circuit, ~1 h CPU in 5 processes.

**Overall reading (draft).** J generalizes better than a same-size MLP at the same entropy at K=25, and that is the
result that survives every control. At K=5 the advantage over every head is mostly J's stochasticity at evaluation.
With enough layouts (K >= 200) a large MLP is best. A classical Fourier-feature head of the same size matches J in
return, in behaviour (as close as two seeds) and uses periodic structure on the data too: the VQC is one way to get
this inductive bias, with no evidence of a circuit-specific advantage.
