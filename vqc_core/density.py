"""
Density-matrix simulation on top of the statevector primitives of core/statevector.py.

A density matrix rho (B, D, D), D = 2^n, is handled as the pure state vec(rho) of 2n qubits: the row
index of rho gives the qubits 0..n-1 and the column index the qubits n..2n-1 (big-endian, as in
core/statevector.py). A unitary U then acts as
    rho -> U rho U^dagger = (U on the row qubits) and (conj(U) on the column qubits),
so every statevector gate with per-sample angles works unchanged. conj(U) v = conj(U conj(v)) gives the
column action without a second implementation of each gate.

Cost is 4^n in memory and time: practical up to about 10-12 qubits.
"""
from __future__ import annotations

import torch

import vqc_core.statevector as sv


def from_state(state: torch.Tensor) -> torch.Tensor:
    """|psi><psi| for every sample. state: (B, D) -> rho: (B, D, D), complex."""
    state = sv.as_complex(state)
    return state[:, :, None] * state[:, None, :].conj()


def zero_density(batch: int, n: int, like: torch.Tensor) -> torch.Tensor:
    """|0...0><0...0| for `batch` samples on n qubits, complex, with the precision and device of `like`."""
    state = torch.zeros(batch, 2 ** n, dtype=like.dtype, device=like.device)
    state[:, 0] = 1
    return from_state(state)


def _n(rho: torch.Tensor) -> int:
    return rho.shape[1].bit_length() - 1


def apply(rho: torch.Tensor, gate, qubits: tuple, *params) -> torch.Tensor:
    """rho -> U rho U^dagger, with U = gate(state, *qubits, *params) one of the gates of
    core/statevector.py (ry, rx, rz, h, x, z, cry, crz, cz, ...). Angles may be scalar or (B,)."""
    B, D, _ = rho.shape
    n = _n(rho)
    vec = gate(rho.reshape(B, D * D), *qubits, *params)                          # U on the row qubits
    vec = gate(vec.conj(), *(q + n for q in qubits), *params).conj()             # conj(U) on the columns
    return vec.resolve_conj().reshape(B, D, D)


def apply_permutation(rho: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Gate that permutes the basis states (CX, SWAP: core.statevector.cx_permutation / swap_permutation)
    applied as rho[perm][:, perm], consistent with state[:, perm]."""
    return rho[:, perm][:, :, perm]


def apply_matrix(rho: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """Block of data-independent gates given as the matrix of core.statevector.block_matrix (state @ M):
    U = M^T, so rho -> M^T rho conj(M)."""
    M = M.to(rho.dtype)
    return M.transpose(-1, -2) @ rho @ M.conj()


def probabilities(rho: torch.Tensor) -> torch.Tensor:
    """Born probabilities of every basis state: the real diagonal. (B, D, D) -> (B, D)."""
    return torch.diagonal(rho, dim1=-2, dim2=-1).real


def expval_z(rho: torch.Tensor) -> torch.Tensor:
    """<Z_i> for every qubit. (B, D, D) -> (B, n)."""
    p = probabilities(rho)
    return p @ sv._z_signs(_n(rho), str(rho.device), p.dtype).T


def register_probabilities(rho: torch.Tensor, n_out: int) -> torch.Tensor:
    """Born probabilities of the first n_out qubits, summed over the others. (B, D, D) -> (B, 2^n_out)."""
    p = probabilities(rho)
    return p.view(rho.shape[0], 2 ** n_out, -1).sum(dim=-1)
