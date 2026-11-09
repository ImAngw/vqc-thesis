# Spec: noise robustness of J (part B)

## Objective
The advisor asked to check that the trained VQC policy is robust on real hardware, by simulating the noise of the
gates. Question: how much test return does J (`vqc-nav-id`, K=25, the head of the generalization study) lose when its
circuit runs with the noise of current IBM processors instead of the exact statevector? Evaluation only: the policies
are trained without noise. Training under noise and runs on real hardware are out of scope here (real hardware is a
later step: same code path as B3, a real backend instead of a fake one).

## B2: effective noise model (done 2026-10-14, `scripts/eval_noise.py`)
- 10 runs `gen-nav-id-k25-s{0..9}-7.5k-cpu-outer`, `model_latest.pt`; test eval as in training (100 fresh layouts,
  eval seeds 10000-10099, sampled actions); the head's density-matrix path replaces the statevector.
- Synthetic: depolarizing p2 after every CX in {0, 0.1, 0.3, 1, 3}% with p1 = p2 / 10 after every RY / RZ; readout
  error 2% alone.
- Calibrated: `fake_torino`, `fake_brisbane` (qiskit-ibm-runtime fake backends: topology, native gates and a snapshot
  of real calibrations). J is transpiled on the backend; effective p1, p2 and readout from the calibration of the
  physical qubits and the transpiled gate counts (see `calibrated_noise`). Idle decoherence (T1 / T2) is not in it.
- Validation of the effective model on s0: TVD of the outcome distribution against a full Aer simulation
  (`AerSimulator.from_backend`, 200 visited states, 20k shots).
- Results in plan 12.7; figure `docs/figures/partB_noise.png` (`scripts/plot_noise.py`).

## B3: episodes with full Aer (added 2026-10-15, before running)
Removes the approximation of B2 where it showed a residual (brisbane TVD 0.028 vs shot floor 0.007).
- Same 10 runs, same test eval, but every action comes from the circuit run on Aer: at each step, for every active
  episode, J with the state's encoder features is transpiled on the backend (`optimization_level=3`, numeric
  angles: the circuit that would be sent to the hardware), simulated by `AerSimulator.from_backend` (per-qubit and
  per-pair gate errors, T1 / T2 relaxation with the gate durations, readout error), **1 shot**; the measured outcome
  is the action (Gray code inverted, as in the head). Steps are batched over the active episodes.
- Backends: `fake_torino`, `fake_brisbane`. Cost measured on s0: ~25 ms per circuit (13 transpile + 12 Aer at batch
  100), ~7100 steps per eval: ~1-1.5 h for 10 seeds x 2 backends.
- Reported next to B2: per seed and mean +- SE, and the paired difference full Aer - effective model with its 95% CI
  (same seeds, same layouts; the action sampling differs, so a difference within the CI means the effective model
  is adequate for the return).
- Check before running: with a noiseless Aer simulator and many shots, the outcome distribution of the transpiled
  circuit matches the head's probabilities (TVD at the shot floor), and the outcome -> action decoding matches the
  head's action code.
