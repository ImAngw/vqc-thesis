# Artifacts outside git

The trained weights and the raw data of the distillation analyses (about 100 MB) are not in the repository. They are
published as one archive (`vqc-thesis-artifacts.tar.gz`, DOI to be added here after the upload) and fetched with

```
ARTIFACTS_URL=<archive url> make fetch
```

which checks the archive against `SHA256SUMS` and unpacks it in place:

| content | needed by |
|---|---|
| `checkpoints/<run>/model_latest.pt` for J, Fourier, h5, h128 (K = 5, 25, 200) and the entropy-matched h5 (K = 25) | part B (`scripts/eval_noise.py`, `eval_aer.py`, `eval_clean_returns.py`), A3 (`policy_equivalence.py`, `cross_distill.py`) |
| `quantum_heads/analysis/data/a3/`, `data/h128rms_s3/` | cross-distillation (Table 10) and distillation (Table 3, phase 2) |
| `quantum_heads/analysis/runs/{a3,distill_concentration}/students/` | trained student heads of those analyses |

`make figures` and `make tables` do not need them: the logs and results they read are in git (`data/`,
`quantum_heads/analysis/runs/`).

The archive is built with `scripts/pack_artifacts.sh` from the directory where the experiments were run.
