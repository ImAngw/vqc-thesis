"""Test return of trained J policies with every action taken by a full Aer simulation of the transpiled circuit
(part B3 of docs/design/SPEC-noise-robustness.md).

At each step, for every active episode, J with the state's encoder features is transpiled on the backend (numeric
angles, optimization_level=3, the physical qubits of eval_noise.calibrated_noise) and run once (1 shot) on
AerSimulator.from_backend: gate errors per qubit / pair, T1 / T2 relaxation, readout error. The measured outcome is
the action (the head's action code inverted). Same envs and layouts as the clean test eval (100 fresh layouts, eval
seeds 10000-10099); steps are batched over the active episodes.

    python scripts/eval_aer.py 'gen-nav-id-k25-s*-7.5k-cpu-outer' --out quantum_heads/analysis/runs/noise/aer
    python scripts/eval_aer.py 'gen-nav-id-k25-s0-7.5k-cpu-outer' --check   # noiseless check of circuit + decoding
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "quantum_heads"), os.path.join(ROOT, "scripts")]

from eval_noise import BACKENDS, _backend, calibrated_noise, head_params, load_policy, visited_features  # noqa: E402
import vqc_ref_j as ref  # noqa: E402
from rl_train.envs import make_env  # noqa: E402
from rl_train.rollout import obs_to_tensor  # noqa: E402


def measured_circuit(params: dict, features: np.ndarray):
    """J with measurements; Qiskit qubit ref.q(k) -> clbit m - 1 - k, so the outcome string reads q0 q1 q2."""
    from qiskit import ClassicalRegister
    qc = ref.build_circuit(params, features)
    m = ref.N_MEASURED
    qc.add_register(ClassicalRegister(m))
    for k in range(m):
        qc.measure(ref.q(k), m - 1 - k)
    return qc


def action_of_outcome(head) -> list[int]:
    """outcome index -> action, the inverse of the head's outcome_of_action (Gray code for vqc-nav-id)."""
    out = head.outcome_of_action.tolist()
    return [out.index(o) for o in range(len(out))]


def outcomes(counts: list[dict]) -> list[int]:
    """1-shot counts -> outcome index (bit string q0 q1 q2, big-endian as in the head)."""
    return [int(next(iter(c)).replace(" ", ""), 2) for c in counts]


def aer_episodes(model, backend, layout, sim, episodes: int, seed0: int, env_kwargs: dict) -> dict:
    from qiskit import transpile
    params, decode = head_params(model.head), action_of_outcome(model.head)
    envs = [make_env(**env_kwargs) for _ in range(episodes)]
    returns, lengths = np.zeros(episodes), np.zeros(episodes, dtype=int)
    successes = np.zeros(episodes, dtype=bool)
    obs = [env.reset(seed=seed0 + i)[0] for i, env in enumerate(envs)]
    active, step, n_circuits, t0 = list(range(episodes)), 0, 0, time.time()
    while active:
        batch = {k: np.stack([obs[i][k] for i in active]) for k in obs[active[0]]}
        with torch.no_grad():
            o = obs_to_tensor(batch)
            feats = model.encoder(o["perception"], o["navigator"], o["memory"]).double().numpy()
        circuits = transpile([measured_circuit(params, f) for f in feats], backend, optimization_level=3,
                             seed_transpiler=0, initial_layout=layout)
        counts = sim.run(circuits, shots=1, seed_simulator=seed0 * 1000 + step).result().get_counts()
        counts = counts if isinstance(counts, list) else [counts]
        n_circuits += len(circuits)
        still = []
        for i, outcome in zip(active, outcomes(counts)):
            obs[i], reward, terminated, truncated, info = envs[i].step(decode[outcome])
            returns[i] += reward
            lengths[i] += 1
            if terminated or truncated:
                successes[i] = bool(info["is_success"])
            else:
                still.append(i)
        active, step = still, step + 1
    for env in envs:
        env.close()
    return {"return_mean": float(returns.mean()), "return_std": float(returns.std()),
            "length_mean": float(lengths.mean()), "success_rate": float(successes.mean()), "circuits": n_circuits,
            "seconds": time.time() - t0}


def check(model, cfg, backend_name: str, n: int = 20, shots: int = 20000) -> None:
    """Noiseless Aer on the transpiled circuit vs the head: outcome distribution and outcome -> action decoding."""
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    backend = _backend(backend_name)
    params = head_params(model.head)
    _, info = calibrated_noise(backend, params)
    layout = [info["physical_qubits"][k] for k in reversed(range(ref.N_MEASURED))]
    feats = visited_features(model, cfg, n)
    with torch.no_grad():
        p_out = model.head.measured_probabilities(feats).numpy()
        p_act = torch.softmax(model.head(feats), -1).numpy()
    decode = action_of_outcome(model.head)
    assert np.allclose(p_act[:, decode], p_out, atol=1e-6), "outcome -> action decoding does not match the head"
    circuits = transpile([measured_circuit(params, f) for f in feats.double().numpy()], backend,
                         optimization_level=3, seed_transpiler=0, initial_layout=layout)
    counts = AerSimulator().run(circuits, shots=shots, seed_simulator=0).result().get_counts()
    tvd, floor = [], []
    for i, (c, p) in enumerate(zip(counts, p_out)):
        freq = np.zeros(len(p))
        for key, v in c.items():
            freq[int(key.replace(" ", ""), 2)] = v / shots
        tvd.append(0.5 * np.abs(freq - p).sum())
        floor.append(0.5 * np.abs(np.random.default_rng(i).multinomial(shots, p / p.sum()) / shots - p).sum())
    print(f"check {backend_name}: decoding ok; noiseless TVD transpiled circuit vs head {np.mean(tvd):.4f} "
          f"(shot floor {np.mean(floor):.4f}) over {n} visited states")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run name globs under --checkpoints")
    ap.add_argument("--checkpoints", default=os.path.join(ROOT, "checkpoints"))
    ap.add_argument("--out", default=os.path.join(ROOT, "quantum_heads", "analysis", "runs", "noise", "aer"))
    ap.add_argument("--backends", default=",".join(BACKENDS))
    ap.add_argument("--episodes", type=int, default=0, help="override eval_episodes (smoke tests only)")
    ap.add_argument("--check", action="store_true", help="only the noiseless check, on the first run")
    args = ap.parse_args()
    torch.set_num_threads(1)
    from qiskit_aer import AerSimulator

    runs = sorted({os.path.basename(p) for g in args.runs for p in glob.glob(os.path.join(args.checkpoints, g))})
    if not runs:
        sys.exit("no run matches")
    if args.check:
        model, cfg = load_policy(runs[0], args.checkpoints)
        for name in args.backends.split(","):
            check(model, cfg, name)
        return
    os.makedirs(args.out, exist_ok=True)
    for run in runs:
        path = os.path.join(args.out, f"{run}.json")
        row = json.load(open(path)) if os.path.exists(path) else {"run": run, "results": {}}
        model, cfg = load_policy(run, args.checkpoints)
        episodes = args.episodes or cfg.eval_episodes
        for name in args.backends.split(","):
            if name in row["results"] and row["results"][name].get("episodes") == episodes:
                continue  # already done
            backend = _backend(name)
            _, info = calibrated_noise(backend, head_params(model.head))
            layout = [info["physical_qubits"][k] for k in reversed(range(ref.N_MEASURED))]
            result = aer_episodes(model, backend, layout, AerSimulator.from_backend(backend), episodes,
                                  cfg.eval_seed, cfg.env_kwargs)
            row["results"][name] = {**result, "episodes": episodes, "physical_qubits": info["physical_qubits"]}
            with open(path, "w") as f:
                json.dump(row, f, indent=1)
            print(f"{run} {name:14s} return {result['return_mean']:6.2f} success {result['success_rate']:.2f} "
                  f"({result['circuits']} circuits, {result['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
