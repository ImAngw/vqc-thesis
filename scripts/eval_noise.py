"""Robustness of trained J policies (head vqc-nav-id) to gate and readout noise (plan part B2).

Loads the final weights (model_latest.pt, round 50) of the runs matched by the globs, swaps the head's clean
statevector path for its density-matrix path with a NoiseParams, and re-runs the test evaluation of training
(sampled actions, eval_episodes fresh layouts from eval_seed): same envs and action generator as the clean eval.

Noise settings:
  - synthetic sweep: depolarizing p2 after every CX, p1 = p2 / 10 after every RY / RZ, no readout error; plus
    readout error alone;
  - calibrated: effective parameters from an IBM fake backend's calibration on the 3 physical qubits the
    transpiler picks for J (see calibrated_noise), checked against a full Aer simulation of the transpiled circuit
    with NoiseModel.from_backend (thermal relaxation included) on states visited by the policy (--validate).

    python scripts/eval_noise.py 'gen-nav-id-k25-s*-7.5k-cpu-outer' --out quantum_heads/analysis/runs/noise
"""
from __future__ import annotations

import argparse
import functools
import glob
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "quantum_heads")]

from rl_train.config import A2CConfig  # noqa: E402
from rl_train.envs import make_env  # noqa: E402
from rl_train.evaluate import evaluate  # noqa: E402
from rl_train.policy import build_policy, encoder_kwargs  # noqa: E402
from train import HEADS  # noqa: E402
from vqc_core import noise as nz  # noqa: E402
import vqc_ref_j as ref  # noqa: E402

# the configuration of every generalization run (docs/design/SPEC-generalization-full.md); env_seed only sets the pool
RUN_CONFIG = dict(head="vqc-nav-id", perception="cnn", detach_critic=1,
                  env_kwargs={"window": 5, "n_obstacles": 30})
P2_SWEEP = (0.0, 0.001, 0.003, 0.01, 0.03)
READOUT_ONLY = 0.02
BACKENDS = ("fake_torino", "fake_brisbane")


def load_policy(run: str, checkpoints: str) -> tuple[torch.nn.Module, A2CConfig]:
    cfg = A2CConfig(**RUN_CONFIG)
    model = build_policy(HEADS[cfg.head], **encoder_kwargs(cfg))
    state = torch.load(os.path.join(checkpoints, run, "model_latest.pt"), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.detach_critic = True
    return model, cfg


def set_noise(model: torch.nn.Module, noise: nz.NoiseParams | None) -> None:
    """Route the head through its density-matrix path (noise given) or its clean statevector path (None)."""
    head = model.head
    head.forward = functools.partial(type(head).forward, head, noise=noise) if noise is not None else \
        functools.partial(type(head).forward, head)


def test_return(model, cfg) -> dict:
    env_fn = lambda: make_env(**cfg.env_kwargs)  # noqa: E731
    return evaluate(model, env_fn, cfg.eval_episodes, cfg.eval_seed, greedy=False)


# --- calibrated noise --------------------------------------------------------------------------------------------

def _backend(name: str):
    from qiskit_ibm_runtime import fake_provider
    cls = "Fake" + "".join(part.capitalize() for part in name.removeprefix("fake_").split("_"))
    return getattr(fake_provider, cls)()


def transpiled_j(backend, params: dict, features: np.ndarray, layout=None):
    """J with measurements (Qiskit qubit ref.q(k) -> clbit m - 1 - k, so the outcome reads q0 q1 q2), transpiled."""
    from qiskit import ClassicalRegister, transpile
    qc = ref.build_circuit(params, features)
    m = ref.N_MEASURED
    qc.add_register(ClassicalRegister(m))
    for k in range(m):
        qc.measure(ref.q(k), m - 1 - k)
    return transpile(qc, backend, optimization_level=3, seed_transpiler=0, initial_layout=layout)


def calibrated_noise(backend, params: dict) -> tuple[nz.NoiseParams, dict]:
    """Effective NoiseParams of J on `backend`. The logical circuit has 18 CX and 93 RY / RZ; after transpilation
    the two-qubit gates (with SWAPs) and the physical sx / x pulses (RZ is virtual, error free) carry the error.
    Depolarizing parameter from the average gate error e of a d-dimensional gate: p = e d / (d - 1), spread over
    the logical gates so that the total depolarizing budget matches:
      p2 = (n_2q_phys / 18) * 4/3 * mean e_2q,   p1 = (n_sx_phys / 93) * 2 * mean e_sx,   readout p01 = p10 = e_meas.
    Idle decoherence (T1 / T2) is not in this model: --validate measures what that leaves out."""
    rng = np.random.default_rng(0)
    t = transpiled_j(backend, params, rng.uniform(-1, 1, ref.FEATURES))
    phys = [t.layout.final_index_layout()[ref.q(k)] for k in range(ref.N_MEASURED)]  # description qubit k -> physical
    ops = t.count_ops()
    two_q = next(g for g in ("cz", "ecr", "cx") if g in ops)
    target = backend.target
    e2 = np.mean([props.error for qargs, props in target[two_q].items()
                  if props is not None and props.error is not None and set(qargs) <= set(phys)])
    esx = np.mean([target["sx"][(q,)].error for q in phys])
    emeas = [target["measure"][(q,)].error for q in phys]
    n_sx = ops.get("sx", 0) + ops.get("x", 0)
    p2 = ops[two_q] / 18 * 4 / 3 * e2
    p1 = n_sx / 93 * 2 * esx
    noise = nz.NoiseParams(p1=float(p1), p2=float(p2), p01=torch.tensor(emeas), p10=torch.tensor(emeas))
    info = dict(backend=backend.name, physical_qubits=phys, two_qubit_gate=two_q, n_2q=int(ops[two_q]), n_sx=int(n_sx),
                depth=t.depth(), e_2q=float(e2), e_sx=float(esx), e_meas=[float(e) for e in emeas], p1=float(p1),
                p2=float(p2))
    return noise, info


def head_params(head) -> dict:
    return {k: p.detach().double().numpy() for k, p in head.named_parameters()}


def visited_features(model, cfg, n: int) -> torch.Tensor:
    """Encoder features of the first n states of the clean sampled test eval (the head's inputs)."""
    seen = []
    hook = model.head.register_forward_pre_hook(lambda mod, args: seen.append(args[0].detach().clone()))
    try:
        test_return(model, cfg)
    finally:
        hook.remove()
    feats = torch.cat(seen)
    idx = torch.randperm(len(feats), generator=torch.Generator().manual_seed(0))[:n]
    return feats[idx]


def validate(model, cfg, backend, noise: nz.NoiseParams, phys: list[int], n: int, shots: int) -> dict:
    """TVD between the outcome distribution of a full Aer simulation (transpiled J, NoiseModel.from_backend) and
    (a) the effective density-matrix model, (b) the clean model, over n visited states."""
    from qiskit_aer import AerSimulator
    sim = AerSimulator.from_backend(backend)
    params = head_params(model.head)
    feats = visited_features(model, cfg, n)
    with torch.no_grad():
        p_eff = model.head.measured_probabilities(feats, noise).numpy()
        p_clean = model.head.measured_probabilities(feats).numpy()
    m = ref.N_MEASURED
    layout = [phys[k] for k in reversed(range(m))]  # Qiskit virtual qubit j = description qubit m - 1 - j
    circuits = [transpiled_j(backend, params, f.numpy(), layout=layout) for f in feats]
    counts = sim.run(circuits, shots=shots, seed_simulator=0).result().get_counts()
    tvd_eff, tvd_clean = [], []
    for c, pe, pc in zip(counts, p_eff, p_clean):
        freq = np.zeros(2 ** m)
        for key, v in c.items():
            freq[int(key.replace(" ", ""), 2)] = v / shots
        tvd_eff.append(0.5 * np.abs(freq - pe).sum())
        tvd_clean.append(0.5 * np.abs(freq - pc).sum())
    # sampling floor: expected TVD of `shots` samples from the exact distribution itself
    floor = np.mean([0.5 * np.abs(np.random.default_rng(i).multinomial(shots, pe / pe.sum()) / shots - pe).sum()
                     for i, pe in enumerate(p_eff)])
    return dict(states=n, shots=shots, tvd_aer_vs_effective=float(np.mean(tvd_eff)),
                tvd_aer_vs_clean=float(np.mean(tvd_clean)), tvd_shot_floor=float(floor))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run name globs under --checkpoints")
    ap.add_argument("--checkpoints", default=os.path.join(ROOT, "checkpoints"))
    ap.add_argument("--out", default=os.path.join(ROOT, "quantum_heads", "analysis", "runs", "noise"))
    ap.add_argument("--episodes", type=int, default=0, help="override eval_episodes (smoke tests only)")
    ap.add_argument("--backends", default=",".join(BACKENDS))
    ap.add_argument("--validate", type=int, default=200, help="states for the Aer check on the first run (0 = skip)")
    ap.add_argument("--shots", type=int, default=20_000)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    runs = sorted({os.path.basename(p) for g in args.runs for p in glob.glob(os.path.join(args.checkpoints, g))})
    if not runs:
        sys.exit("no run matches")
    os.makedirs(args.out, exist_ok=True)
    settings = [("clean", None)]
    settings += [(f"p2={p2:g}", nz.NoiseParams(p1=p2 / 10, p2=p2)) for p2 in P2_SWEEP]
    settings += [(f"readout={READOUT_ONLY:g}", nz.NoiseParams(p01=READOUT_ONLY, p10=READOUT_ONLY))]
    backends = [_backend(b) for b in args.backends.split(",") if b]

    for run in runs:
        path = os.path.join(args.out, f"{run}.json")
        if os.path.exists(path):
            print(f"{run}: done, skipped")
            continue
        model, cfg = load_policy(run, args.checkpoints)
        if args.episodes:
            cfg.eval_episodes = args.episodes
        row = {"run": run, "episodes": cfg.eval_episodes, "eval_seed": cfg.eval_seed, "results": {}, "calibration": {}}
        run_settings = list(settings)
        for be in backends:
            noise, info = calibrated_noise(be, head_params(model.head))
            row["calibration"][be.name] = info
            run_settings.append((be.name, noise))
            if args.validate and run == runs[0]:
                set_noise(model, None)
                row["calibration"][be.name]["validation"] = validate(model, cfg, be, noise, info["physical_qubits"],
                                                                     args.validate, args.shots)
        for name, noise in run_settings:
            t0 = time.time()
            set_noise(model, noise)
            res = test_return(model, cfg)
            row["results"][name] = res
            print(f"{run} {name:16s} return {res['return_mean']:6.2f} success {res.get('success_rate', float('nan')):.2f}"
                  f" ({time.time() - t0:.0f}s)", flush=True)
        set_noise(model, None)
        with open(path, "w") as f:
            json.dump(row, f, indent=1)


if __name__ == "__main__":
    main()
