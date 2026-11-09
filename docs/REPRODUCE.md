# Retraining

How to rebuild the results of [the report](report_testa_vqc.md). Sections 1-3 are about retraining, 4 onwards about the analyses.

## One run (example: MLP h5, K=25 layouts, seed 0)

```
python train.py --run-name gen-h5-k25-s0 --log-dir logs \
  --head mlp5 --detach-critic 1 --perception cnn --env-kwargs '{"window": 5, "n_obstacles": 30}' \
  --layout-pool 25 --layout-pool-seed 0 --env-seed 818069386 --seed 0 \
  --n-envs 16 --n-steps 64 --total-updates 7500 \
  --inner-steps 150 --outer-lr 0.7 --outer-momentum 0.9 --outer-dtype bf16 --eval-every 150
```

- Schedule: 50 rounds of 150 updates. At the end of each round the weights take an outer SGD step
  (lr 0.7, Nesterov momentum 0.9) on the pseudo-gradient "weights at the start of the round - weights now".
  The pseudo-gradient is rounded to bf16 (`--outer-dtype bf16`): this is what the report's runs used, and with
  fp32 the trajectories differ from the second round on.
- The run writes `logs/<run-name>/node_0.log` (one block per round, read by `scripts/summarize_gen_pilot.py` and
  `scripts/plot_gen_curve.py`) and the file metrics.json in the same directory (every metric of every update and evaluation).
- Use `--device cpu`; the report's runs were not run on GPU.

## Every run of the report

`configs/runs.json` has the parameters of the 289 runs whose logs are in `data/logs/`; `scripts/cell_command.py` prints
the `train.py` command of a run, with the seed mapping below applied:

```
python scripts/cell_command.py --list                              # the 45 cells
python scripts/cell_command.py 'gen-nav-id-k25-s*-7.5k-cpu-outer'  # the 10 seeds of one cell
```

What was checked, against the logs of the original runs (first three rounds, every column; one of them all 50 rounds):

| run | machine of the original run | result |
|---|---|---|
| `gen-h5-k25-s0` | same Mac | identical over all 50 rounds |
| `nav-id-hlr2e-3-detc-s0` (J, K=inf, environment v8) | same Mac | identical, rounds 1-3 |
| the other runs | another machine (nodo1, nodo2) or not retrained | not comparable bit by bit, see Reproducibility |

Runs from before 2026-10-14 had no `perception`/`env_kwargs` in their record: they ran on environment v8 (CNN
perception, window 5, 30 obstacles) and the command passes it. The warm-started runs (`ws-*`) cannot be reproduced.

## Seeds

`--seed` sets the initial weights. The environments and the action sampling follow `--env-seed`, and the K training
layouts follow `--layout-pool-seed`.

The report's runs were launched with a seed `e` for the environments and the layouts, and the runtime derived the
seed actually given to the vector environment as

```python
torch.randint(0, 2**31 - 1, (1,), generator=torch.Generator().manual_seed(e)).item()
```

`train.py` uses `--env-seed` directly, so to reproduce a run with seed `s` and environment seed `e` pass
`--seed s --layout-pool-seed e --env-seed <drawn value of e>`. In the report `s = e`.

| e | `--env-seed` |
|---|---|
| 0 | 818069386 |
| 1 | 1422616888 |
| 2 | 245121889 |
| 3 | 740111726 |
| 4 | 1437444519 |
| 5 | 2143903636 |
| 6 | 1001570786 |
| 7 | 1631897122 |
| 8 | 1107986653 |
| 9 | 1654461528 |

## Reproducibility

Checked on one machine (macOS, CPU, torch 2.14.1) with seed 0 of the example above: all 50 rounds agree with the log
of the original run (evaluation return on fresh layouts 1.27, on the training layouts 4.88, success rate 0.68, mean of
rounds 41-50), and the first three rounds agree on every column of the log.

Runs are bit-identical only on the same machine. Seeds 1 and 2 of the same cell were originally run on a different
machine; retrained here they diverge from the originals (train entropy differs from round 3 and round 6 respectively;
return on fresh layouts, rounds 41-50: 0.91 vs 3.33 and 1.24 vs 1.67). Three seeds are too few to say whether the
distributions match: the across-seed standard deviation of the original runs is already about 1.1.

## Entropy-target sweep (thesis, chapter on the entropy control)

The sweep holds the training entropy of MLP h5 and h128 at K=25 at four targets (0.4, 0.8, 1.2, 1.8 nats, fixed before
the runs) with the adaptive entropy coefficient (`--target-entropy`), seeds 0-9: 80 runs named
`ent-sweep-<h5|h128>-t<target>-k25-s<seed>-7.5k-cpu-outer`, with the same configuration as the `ent-h5-k25` and
`ent-h128-k25` cells (only the target changes). All of them were trained on the same machine (macOS, CPU).

```
python scripts/sweep_entropy.py --list                 # the 80 runs
python scripts/sweep_entropy.py --write                # add them to configs/runs.json (already done in the repository)
python scripts/sweep_entropy.py --run mac --jobs 4     # train them, 4 at a time; completed runs are skipped
python scripts/summarize_entropy_sweep.py --logs data/logs --logs logs --tex thesis/generated --fig thesis/figures/entropy_sweep.pdf
```

A run is single-threaded in practice: the same run gives the same numbers with 1, 2 or the default number of torch
threads (checked on 4 rounds), so several runs can share a machine. An interrupted run is renamed `<run>.partial` and
restarted from scratch. The logs of the finished runs are copied to `data/logs/<run>/node_0.log`.

## Results from the shipped data (no retraining)

```
make tables     # tables 4-7 (return per head and K, entropy control, paired differences J - X, overfitting gap) and 11-12 (noise)
make figures    # figure of the curve in K (section 6) and of the noise robustness (section 9)
```

Table 8 (greedy return) reads the per-round evaluation files `data/checkpoints/<run>/metrics.jsonl`, written by
`scripts/pack_artifacts.sh`:

```
python scripts/summarize_eval_metrics.py 'gen-*' 'ent-*' --checkpoints data/checkpoints \
  --ref 'nav-id-hlr2e-3-detc-s*-7.5k-cpu-outer=nav-id:inf'
```

## Analyses on trained runs (need `make fetch`)

```
# section 8: distance between policies, periods of the data gates, temperature control, cross-distillation
python quantum_heads/analysis/policy_equivalence.py distance --k 25
python quantum_heads/analysis/policy_equivalence.py periods --k 25
python quantum_heads/analysis/policy_equivalence.py temperature --k 5 --arms four,h5,h128
python quantum_heads/analysis/cross_distill.py --k 25 200 --workers 8

# section 9: noise robustness of J at K = 25
python scripts/eval_noise.py 'gen-nav-id-k25-s*-7.5k-cpu-outer' --out quantum_heads/analysis/runs/noise
python scripts/eval_clean_returns.py 'gen-four-k25-s*-7.5k-cpu-outer=fourier4' \
  'ent-h5-k25-s*-7.5k-cpu-outer=mlp5' --out quantum_heads/analysis/runs/noise/references_k25.json
python scripts/eval_aer.py 'gen-nav-id-k25-s*-7.5k-cpu-outer' --out quantum_heads/analysis/runs/noise/aer   # ~1-1.5 h per backend
```

Section 4, phase 2 (distillation of a teacher MLP h128 into the VQC heads I and J; 3 seeds, Adam, 100 epochs, 20 000
training and 5 000 validation states, head lr 3e-3 and 1e-2):

```
python quantum_heads/analysis/distill.py --teacher checkpoints/h128-rms-s3-7.5k-cpu-outer/model_latest.pt \
  --teacher-head mlp128 --reference mlp128 --data-dir quantum_heads/analysis/data/h128rms_s3 \
  --heads mlp128 vqc-optim-id vqc-optim-id-gray vqc-nav-id --lrs 0.003 0.01 --save-students \
  --out quantum_heads/analysis/runs/distill_concentration
python quantum_heads/analysis/eval_students.py quantum_heads/analysis/runs/distill_concentration/students
```

The results are in `quantum_heads/analysis/runs/distill_concentration/`. The scripts build the policies and the
environment of the report's setting (CNN perception, window 5, 30 obstacles). The teacher is the run
`h128-rms-s3-7.5k-cpu-outer`: its logits on the 5 000 validation states stored in `data/h128rms_s3/val.pt` are
identical to the stored ones (maximum difference 0), while the other h128 runs with seed 3 differ by 10-12. The
distillation was rerun from this repository (same Mac): the validation fraction of all 48 students (4 heads x 2
variants x 3 seeds x 2 learning rates) is identical to the original `results.csv`, and the evaluation of the students
(500 test episodes) reproduces the rows of table 3, phase 2, to the printed precision (for example J 4.968 +- 0.036
with the encoder adapted, 4.527 +- 0.214 with it frozen; teacher mlp128 5.184 +- 0.009).

Tables 11 and 12 (noise) are printed by `scripts/summarize_noise.py` from the JSON files that `scripts/eval_noise.py` and
`scripts/eval_aer.py` write.

## What cannot be rerun from this repository

Section 4, phase 3 (warm start from distilled students, entropy bonus, critic warm-up) used options of the original
training runner that are not in `train.py`; only its logs (`data/logs/ws-*`, `*-ent5e-3-*`) and the initial weights
(`quantum_heads/analysis/runs/warmstart_init/`) are kept.
