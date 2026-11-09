"""
Helpers shared by the equivalence tests of the generated VQCs (copied next to the tests of the target
project, like core/). Qiskit and Aer are needed here, never in the modules under test.

Contents: tolerances, random parameters and model construction, flatten/unflatten of parameter dicts, the
Aer NoiseModel with labelled gates, the exact density matrix from Aer, and the statistical comparison of
sampled outcome frequencies with exact probabilities.
"""
from __future__ import annotations

import numpy as np
import torch
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

TOL = {torch.float32: 1e-5, torch.float64: 1e-10}      # clean-path tolerance on states and probabilities
SHOTS = 400_000


# --- parameters and models ----------------------------------------------------------------------------
def random_params(rng: np.random.Generator, sizes: dict, scale: float) -> dict:
    """{name: normal(0, scale) array of the given size}; size 0 means a scalar (0-d array)."""
    return {k: rng.normal(size=n or None) * scale for k, n in sizes.items()}


def make_model(cls, params: dict, dtype, **ctor):
    """cls(**ctor) in the given dtype with every named parameter set to params[name]."""
    model = cls(**ctor).to(dtype)
    with torch.no_grad():
        for k, v in params.items():
            getattr(model, k).copy_(torch.tensor(np.asarray(v), dtype=dtype))
    return model


def flatten(params: dict, names: list[str]) -> np.ndarray:
    return np.concatenate([np.atleast_1d(params[k]) for k in names])


def unflatten(theta: np.ndarray, sizes: dict) -> dict:
    out, i = {}, 0
    for k, n in sizes.items():
        out[k], i = (theta[i:i + n] if n else theta[i]), i + (n or 1)
    return out


# --- Aer ----------------------------------------------------------------------------------------------
def noise_model(one_qubit_labels, two_qubit_labels, p1, p2, readout=None) -> NoiseModel:
    """Depolarizing noise attached to gate labels (the reference circuits wrap every gate in a `unitary` with
    label=<gate name>, so one noise event follows each nominal gate; Aer's density_matrix method rejects
    non-native gates such as CRY and a transpile would decompose them). readout: list of
    (qiskit_qubit, p01, p10) for the measured qubits only."""
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), list(one_qubit_labels))
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), list(two_qubit_labels))
    for qubit, p01, p10 in readout or []:
        nm.add_readout_error(ReadoutError([[1 - p01, p01], [p10, 1 - p10]]), [qubit])
    return nm


def aer_density_matrix(qc: QuantumCircuit, nm: NoiseModel) -> np.ndarray:
    qc = qc.copy()
    qc.save_density_matrix()
    res = AerSimulator(method="density_matrix", noise_model=nm).run(qc).result()
    assert res.success, res.status
    return np.array(res.data(0)["density_matrix"])


def aer_register_frequencies(qc: QuantumCircuit, nm: NoiseModel, measured: list[int], seed: int,
                             shots: int = SHOTS) -> np.ndarray:
    """Outcome frequencies of the Qiskit qubits `measured` (most significant first), 2^len(measured) values.
    Readout errors in `nm` act on these measurements, and only on them."""
    qc = qc.copy()
    qc.add_register(ClassicalRegister(len(measured)))
    for k, qubit in enumerate(measured):
        qc.measure(qubit, len(measured) - 1 - k)
    counts = AerSimulator(method="density_matrix", noise_model=nm, seed_simulator=seed) \
        .run(qc, shots=shots).result().get_counts()
    freq = np.zeros(2 ** len(measured))
    for key, c in counts.items():
        freq[int(key, 2)] = c / shots
    return freq


def assert_frequencies_match(freq: np.ndarray, probs: np.ndarray, shots: int = SHOTS) -> None:
    """Each sampled frequency within 5 standard errors (+1e-4) of the exact probability."""
    sigma = np.sqrt(probs * (1 - probs) / shots)
    assert np.all(np.abs(freq - probs) < 5 * sigma + 1e-4), np.abs(freq - probs).max()


# --- non-degeneracy guards ---------------------------------------------------------------------------
# Matching Qiskit is vacuous when the output hardly depends on the inputs or the parameters (two vectors of
# zeros are equal). These helpers measure that on the PyTorch model.

def output_std(model_factory, rng: np.random.Generator, feature_size: int, draws: int = 200) -> np.ndarray:
    """Standard deviation of every output over random parameters and inputs (batch of `draws` inputs, one
    random parameter set per draw)."""
    outs = []
    for _ in range(draws):
        model = model_factory()
        x = torch.tensor(rng.uniform(-np.pi, np.pi, size=(1, feature_size)), dtype=torch.float64)
        with torch.no_grad():
            outs.append(model(x)[0].numpy())
    return np.array(outs).std(axis=0)


def blind_feature_slices(model_factory, rng: np.random.Generator, slices: dict, draws: int = 20,
                         delta: float = 0.7, tol: float = 1e-6) -> list[str]:
    """Names of the feature slices whose change never moves the outputs (max |change| < tol over the draws)."""
    biggest = {name: 0.0 for name in slices}
    for _ in range(draws):
        model = model_factory()
        x = torch.tensor(rng.uniform(-np.pi, np.pi, size=(1, max(s.stop for s in slices.values()))), dtype=torch.float64)
        with torch.no_grad():
            base = model(x)
            for name, sl in slices.items():
                y = x.clone()
                y[:, sl] += delta
                biggest[name] = max(biggest[name], float((model(y) - base).abs().max()))
    return [name for name, v in biggest.items() if v < tol]


def dead_parameter_entries(model_factory, rng: np.random.Generator, feature_size: int, draws: int = 5,
                           tol: float = 1e-8) -> dict[str, list[int]]:
    """{parameter: [flat indices]} of the entries whose gradient stays below tol on every draw (random
    parameters, inputs and output weights). A dead entry is not a bug of the simulator, but the author must
    know about it."""
    biggest = {}
    for _ in range(draws):
        model = model_factory()
        x = torch.tensor(rng.uniform(-np.pi, np.pi, size=(4, feature_size)), dtype=torch.float64)
        out = model(x)
        w = torch.tensor(rng.normal(size=out.shape[-1]), dtype=torch.float64)
        model.zero_grad()
        (out * w).sum().backward()
        for name, p in model.named_parameters():
            g = p.grad.abs().reshape(-1).numpy()
            biggest[name] = np.maximum(biggest.get(name, 0.0), g)
    dead = {name: [int(i) for i in np.where(g < tol)[0]] for name, g in biggest.items()}
    return {name: idx for name, idx in dead.items() if idx}
