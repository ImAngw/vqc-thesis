"""
Noise channels on density matrices (core/density.py), with the conventions of qiskit_aer.noise.

Depolarizing channel on a set S of k qubits, d = 2^k, as in qiskit_aer.noise.depolarizing_error(p, k):
    rho -> (1 - p) rho + p Tr_S[rho] (x) I_S / d
(checked numerically against Aer in tests/test_noise.py; p up to d^2 / (d^2 - 1) is a valid channel).
In Aer the noise attaches to a gate name, after the gate: here depolarize() is called right after the gate
it belongs to, on the qubits of that gate.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from vqc_core import density as dm


def depolarize(rho: torch.Tensor, qubits: tuple, p) -> torch.Tensor:
    """Depolarizing channel with probability p on the given qubits. rho: (B, D, D); p is a float or a
    scalar tensor (differentiable)."""
    B, D, _ = rho.shape
    n = D.bit_length() - 1
    qubits = tuple(qubits)
    k = len(qubits)
    d = 2 ** k
    rows = [1 + s for s in qubits]
    cols = [1 + n + s for s in qubits]
    others = [a for a in range(1, 2 * n + 1) if a not in rows + cols]
    perm = [0] + others + rows + cols
    t = rho.reshape(B, *([2] * (2 * n))).permute(perm)
    rest = t.shape[1:1 + len(others)]
    t = t.reshape(B, *rest, d, d)
    traced = torch.diagonal(t, dim1=-2, dim2=-1).sum(-1)                         # Tr_S rho: (B, *rest)
    eye = torch.eye(d, dtype=rho.dtype, device=rho.device) / d
    out = (1 - p) * t + p * traced[..., None, None] * eye
    out = out.reshape(B, *rest, *([2] * k), *([2] * k))
    inverse = sorted(range(len(perm)), key=perm.__getitem__)
    return out.permute(inverse).reshape(B, D, D)


def depolarizing_1q(rho: torch.Tensor, qubit: int, p) -> torch.Tensor:
    return depolarize(rho, (qubit,), p)


def depolarizing_2q(rho: torch.Tensor, a: int, b: int, p) -> torch.Tensor:
    return depolarize(rho, (a, b), p)


def readout_matrices(p01, p10, m: int, like: torch.Tensor) -> torch.Tensor:
    """Per-qubit confusion matrices M[i] = [[1 - p01_i, p10_i], [p01_i, 1 - p10_i]], shape (m, 2, 2), acting
    on the probability vector of qubit i (M[i] @ [P(0), P(1)]). p01 = P(read 1 | true 0), p10 = P(read 0 |
    true 1); each is a float, a scalar tensor or a tensor of shape (m,) (one value per qubit)."""
    p01 = torch.as_tensor(p01, dtype=like.dtype, device=like.device).expand(m)
    p10 = torch.as_tensor(p10, dtype=like.dtype, device=like.device).expand(m)
    return torch.stack((torch.stack((1 - p01, p10), -1), torch.stack((p01, 1 - p10), -1)), -2)


def apply_readout(probs: torch.Tensor, p01, p10) -> torch.Tensor:
    """Readout error on the exact outcome probabilities of the measured qubits: p_noisy = (M_0 x ... x
    M_{m-1}) p, applied one qubit at a time (no 2^m x 2^m matrix). probs: (B, 2^m), qubit 0 the most
    significant bit, as everywhere in core/. Differentiable in the error probabilities too.

    In Aer the readout error acts on the classical result of a measurement, not on rho: the exact
    counterpart of ReadoutError([[1 - p01, p01], [p10, 1 - p10]]) is this function."""
    B, K = probs.shape
    m = K.bit_length() - 1
    M = readout_matrices(p01, p10, m, probs)
    for i in range(m):
        v = probs.reshape(B, 2 ** i, 2, -1)
        v0, v1 = v[:, :, 0], v[:, :, 1]
        probs = torch.stack((M[i, 0, 0] * v0 + M[i, 0, 1] * v1, M[i, 1, 0] * v0 + M[i, 1, 1] * v1), dim=2)
    return probs.reshape(B, K)


@dataclass
class NoiseParams:
    """Synthetic noise model: depolarizing probability after every 1-qubit gate (p1) and 2-qubit gate (p2),
    and readout errors p01 = P(read 1 | true 0), p10 = P(read 0 | true 1). Each value is a float or a tensor
    (readout: also one value per measured qubit). All zero = noiseless."""
    p1: float | torch.Tensor = 0.0
    p2: float | torch.Tensor = 0.0
    p01: float | torch.Tensor = 0.0
    p10: float | torch.Tensor = 0.0


def noisy_gate(rho: torch.Tensor, gate, qubits: tuple, params: tuple, p) -> torch.Tensor:
    """A nominal gate followed by its depolarizing event: gate = one of the gates of core/statevector.py,
    applied as dm.apply(rho, gate, qubits, *params), then depolarize(rho, qubits, p). Use p1 for 1-qubit
    gates and p2 for 2-qubit gates."""
    return depolarize(dm.apply(rho, gate, qubits, *params), qubits, p)


def noisy_permutation(rho: torch.Tensor, perm: torch.Tensor, qubits: tuple, p) -> torch.Tensor:
    """Same for a gate given as a basis-index permutation (CX, SWAP), acting on `qubits`."""
    return depolarize(dm.apply_permutation(rho, perm.to(rho.device)), qubits, p)
