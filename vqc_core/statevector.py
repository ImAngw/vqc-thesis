"""
Pure-PyTorch statevector primitives for variational quantum circuits.

Conventions
-----------
- State: tensor (B, 2^n). Real (float32/float64) for circuits made only of real gates (RY, CRY, CX);
  complex (complex64/complex128) once complex gates appear. Every function here accepts both dtypes.
- Qubit order is big-endian: qubit 0 is the most significant bit of the basis index. Qiskit is
  little-endian: qubit i here is qubit n-1-i in Qiskit. Applying every gate of qubit i to Qiskit qubit
  n-1-i makes the basis indices of the two statevectors coincide, so they can be compared directly.
- A gate (or block of gates) that does not depend on the data is a 2^n x 2^n matrix shared by the whole
  batch: block_matrix builds it by applying the gates to the identity matrix (one row per basis state),
  and it is then applied with a single product, state @ matrix.

Ported from QML_Thesis/quantum/statevector.py.
"""
from __future__ import annotations

from functools import lru_cache

import torch


def n_qubits(state: torch.Tensor) -> int:
    return state.shape[1].bit_length() - 1


def product_state(phi: torch.Tensor) -> torch.Tensor:
    """RY(phi_0) x ... x RY(phi_{n-1}) |0...0>, built qubit by qubit. phi: (B, n) -> (B, 2^n)."""
    B = phi.shape[0]
    state = phi.new_ones(B, 1)
    for k in range(phi.shape[1]):
        h = phi[:, k:k + 1] / 2
        # the new qubit becomes the least significant bit: qubit 0 stays the most significant one
        state = torch.stack((state * torch.cos(h), state * torch.sin(h)), dim=2).reshape(B, -1)
    return state


def ry(state: torch.Tensor, qubit: int, angle: torch.Tensor) -> torch.Tensor:
    """RY(angle) on one qubit; angle is a scalar (shared by the batch) or (B,)."""
    B = state.shape[0]
    psi = state.view(B, 2 ** qubit, 2, -1)
    c, s = torch.cos(angle / 2).reshape(-1, 1, 1), torch.sin(angle / 2).reshape(-1, 1, 1)
    a0, a1 = psi[:, :, 0], psi[:, :, 1]
    return torch.stack((c * a0 - s * a1, s * a0 + c * a1), dim=2).reshape(B, -1)


def ry_layer(state: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """RY(phi_k) on every qubit k, with a different angle per sample. phi: (B, n).
    Two implementations of the same function. Without gradient (data collection, small batches): a
    broadcast matmul per qubit, the fastest. With gradient (updates, large batches): amplitude pairs,
    because the backward of a broadcast matmul is very slow on CUDA."""
    B, n = phi.shape
    c, s = torch.cos(phi / 2), torch.sin(phi / 2)
    if torch.is_grad_enabled():
        c, s = c[:, :, None, None], s[:, :, None, None]
        for q in range(n):
            psi = state.view(B, 2 ** q, 2, -1)
            a0, a1 = psi[:, :, 0], psi[:, :, 1]
            state = torch.stack((c[:, q] * a0 - s[:, q] * a1, s[:, q] * a0 + c[:, q] * a1), dim=2).reshape(B, -1)
        return state
    R = torch.stack((torch.stack((c, -s), -1), torch.stack((s, c), -1)), -2).unsqueeze(2)   # (B, n, 1, 2, 2)
    R = R.to(state.dtype)
    for q in range(n):
        state = (R[:, q] @ state.view(B, 2 ** q, 2, -1)).reshape(B, -1)
    return state


def cry(state: torch.Tensor, control: int, target: int, angle: torch.Tensor) -> torch.Tensor:
    """Controlled-RY(angle): RY on the target if the control is |1>, identity otherwise.
    Valid for control < target and control > target. angle is a scalar or (B,)."""
    B = state.shape[0]
    n = n_qubits(state)
    lo, hi = sorted((control, target))
    psi = state.view(B, 2 ** lo, 2, 2 ** (hi - lo - 1), 2, 2 ** (n - hi - 1))
    c_ax, t_ax = (2, 4) if control < target else (4, 2)
    off, on = psi.select(c_ax, 0), psi.select(c_ax, 1)
    t = t_ax if t_ax < c_ax else t_ax - 1           # target axis once the control axis is removed
    a0, a1 = on.select(t, 0), on.select(t, 1)
    c, s = torch.cos(angle / 2).reshape(-1, 1, 1, 1), torch.sin(angle / 2).reshape(-1, 1, 1, 1)
    on = torch.stack((c * a0 - s * a1, s * a0 + c * a1), dim=t)
    return torch.stack((off, on), dim=c_ax).reshape(B, -1)


def cry_layer(state: torch.Tensor, pairs, angles: torch.Tensor) -> torch.Tensor:
    """CRY on the (control, target) pairs in the given order, angle angles[k] for pair k."""
    for k, (c, t) in enumerate(pairs):
        state = cry(state, c, t, angles[k])
    return state


def cx_permutation(pairs, n: int) -> torch.Tensor:
    """CX on the (control, target) pairs, in the given order, as a permutation of the basis indices:
    state[:, perm] applies the whole sequence."""
    perm = torch.arange(2 ** n)
    for c, t in pairs:
        # after the CX, the amplitude of index i is the one of index j, the target bit flipped if the control is 1
        idx = torch.arange(2 ** n)
        src = torch.where((idx >> (n - 1 - c)) & 1 == 1, idx ^ (1 << (n - 1 - t)), idx)
        perm = perm[src]
    return perm


def block_matrix(apply_gates, n: int, like: torch.Tensor) -> torch.Tensor:
    """2^n x 2^n matrix of a block of data-independent gates: apply_gates applied to the identity
    (row j = the block applied to the basis state |j>). The block is then applied as state @ matrix."""
    return apply_gates(torch.eye(2 ** n, dtype=like.dtype, device=like.device))


def probabilities(state: torch.Tensor) -> torch.Tensor:
    """Born probabilities of every basis state. (B, 2^n) -> (B, 2^n), real for real and complex states."""
    return state.real ** 2 + state.imag ** 2 if state.is_complex() else state ** 2


def register_probabilities(state: torch.Tensor, n_out: int) -> torch.Tensor:
    """Born probabilities of the states of the first n_out qubits (output register), summed over all the
    states of the other qubits (ancillas). (B, 2^n) -> (B, 2^n_out)."""
    return probabilities(state).view(state.shape[0], 2 ** n_out, -1).sum(dim=-1)


@lru_cache
def _z_signs(n: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    idx = torch.arange(2 ** n)
    bits = (idx[None, :] >> (n - 1 - torch.arange(n)[:, None])) & 1            # (n, 2^n)
    return (1 - 2 * bits).to(dtype=dtype, device=device)


def expval_z_from_probabilities(probs: torch.Tensor) -> torch.Tensor:
    """<Z_i> = P(qubit i = 0) - P(qubit i = 1) from outcome probabilities (B, 2^n) -> (B, n); also valid
    for probabilities already affected by readout error."""
    return probs @ _z_signs(probs.shape[1].bit_length() - 1, str(probs.device), probs.dtype).T


def expval_z(state: torch.Tensor) -> torch.Tensor:
    """<Z_i> for every qubit of a pure state. (B, 2^n) -> (B, n)."""
    return expval_z_from_probabilities(probabilities(state))


# ---------------------------------------------------------------------------------------------------
# Complex and multi-qubit gates
# ---------------------------------------------------------------------------------------------------

def as_complex(state: torch.Tensor) -> torch.Tensor:
    """Promote a real state to the complex dtype of the same precision; complex states are returned as is."""
    if state.is_complex():
        return state
    return state.to(torch.complex64 if state.dtype == torch.float32 else torch.complex128)


def kron(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-sample Kronecker product: a (B, m, n), b (B, p, q) -> (B, m*p, n*q). For state vectors use
    columns (n = q = 1). The first factor corresponds to the most significant qubits."""
    B = max(a.shape[0], b.shape[0])
    return (a[:, :, None, :, None] * b[:, None, :, None, :]).reshape(B, a.shape[1] * b.shape[1],
                                                                       a.shape[2] * b.shape[2])


def kron_chain(mats: torch.Tensor) -> torch.Tensor:
    """Per-sample Kronecker product of the factors along dimension 1: (B, n, p, q) -> (B, p^n, q^n);
    factor 0 is the most significant qubit. With the 2x2 matrices of one gate on each of n qubits it
    gives the matrix of the whole layer."""
    out = mats[:, 0]
    for j in range(1, mats.shape[1]):
        out = kron(out, mats[:, j])
    return out


def gate_1q(state: torch.Tensor, qubit: int, U: torch.Tensor) -> torch.Tensor:
    """Apply a 2x2 matrix U (shared, shape (2, 2), or per sample, shape (B, 2, 2)) to one qubit.
    A real state is promoted to complex if U is complex."""
    if U.is_complex():
        state = as_complex(state)
    U = U.to(state.dtype)
    B = state.shape[0]
    U = U.expand(B, 2, 2) if U.dim() == 2 else U
    psi = state.view(B, 2 ** qubit, 2, -1)
    a0, a1 = psi[:, :, 0], psi[:, :, 1]
    u = U[:, :, :, None, None]
    return torch.stack((u[:, 0, 0] * a0 + u[:, 0, 1] * a1, u[:, 1, 0] * a0 + u[:, 1, 1] * a1), dim=2).reshape(B, -1)


def _rx_matrices(angle: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(angle / 2).reshape(-1), torch.sin(angle / 2).reshape(-1)
    z = torch.zeros_like(c)
    return torch.stack((torch.stack((torch.complex(c, z), torch.complex(z, -s)), -1),
                        torch.stack((torch.complex(z, -s), torch.complex(c, z)), -1)), -2)


def _rz_matrices(angle: torch.Tensor) -> torch.Tensor:
    angle = angle.reshape(-1)
    one, z = torch.ones_like(angle), torch.zeros_like(angle)
    d0, d1 = torch.polar(one, -angle / 2), torch.polar(one, angle / 2)
    zc = torch.complex(z, z)
    return torch.stack((torch.stack((d0, zc), -1), torch.stack((zc, d1), -1)), -2)


def rx(state: torch.Tensor, qubit: int, angle: torch.Tensor) -> torch.Tensor:
    """RX(angle) on one qubit, as in Qiskit; angle is a scalar or (B,). The result is complex."""
    return gate_1q(state, qubit, _rx_matrices(angle))


def rz(state: torch.Tensor, qubit: int, angle: torch.Tensor) -> torch.Tensor:
    """RZ(angle) = diag(exp(-i angle/2), exp(+i angle/2)) on one qubit, as in Qiskit; angle is a scalar
    or (B,). The result is complex."""
    return gate_1q(state, qubit, _rz_matrices(angle))


_SQRT_HALF = 0.5 ** 0.5
H_MATRIX = torch.tensor([[_SQRT_HALF, _SQRT_HALF], [_SQRT_HALF, -_SQRT_HALF]], dtype=torch.float64)
X_MATRIX = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
Z_MATRIX = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.float64)   # float64 constants, cast to the state dtype


def h(state: torch.Tensor, qubit: int) -> torch.Tensor:
    return gate_1q(state, qubit, H_MATRIX.to(state.device))


def x(state: torch.Tensor, qubit: int) -> torch.Tensor:
    return gate_1q(state, qubit, X_MATRIX.to(state.device))


def z(state: torch.Tensor, qubit: int) -> torch.Tensor:
    return gate_1q(state, qubit, Z_MATRIX.to(state.device))


T_MATRIX = torch.tensor([[1.0, 0.0], [0.0, 0.7071067811865476 + 0.7071067811865476j]], dtype=torch.complex128)


def t(state: torch.Tensor, qubit: int) -> torch.Tensor:
    """T gate = diag(1, exp(i pi/4)), as in Qiskit (the global phase is kept). The result is complex."""
    return gate_1q(state, qubit, T_MATRIX.to(state.device))


def _bit(n: int, qubit: int, device) -> torch.Tensor:
    """Value (0 or 1) of the given qubit for every basis index, shape (2^n,)."""
    return (torch.arange(2 ** n, device=device) >> (n - 1 - qubit)) & 1


def cz(state: torch.Tensor, a: int, b: int) -> torch.Tensor:
    """Controlled-Z (symmetric in a and b): sign -1 on the basis states where both qubits are 1."""
    n = n_qubits(state)
    sign = 1 - 2 * (_bit(n, a, state.device) & _bit(n, b, state.device))
    return state * sign.to(state.dtype)


def crz(state: torch.Tensor, control: int, target: int, angle: torch.Tensor) -> torch.Tensor:
    """Controlled-RZ(angle), as in Qiskit: RZ on the target if the control is |1>. angle is a scalar or
    (B,). The result is complex."""
    state = as_complex(state)
    n = n_qubits(state)
    bc, bt = _bit(n, control, state.device), _bit(n, target, state.device)
    exponent = angle.reshape(-1, 1).to(state.real.dtype) / 2 * (bc * (2 * bt - 1)).to(state.real.dtype)
    return state * torch.polar(torch.ones_like(exponent), exponent)


def swap_permutation(pairs, n: int) -> torch.Tensor:
    """SWAP on the (a, b) pairs, in the given order, as a permutation of the basis indices:
    state[:, perm] applies the whole sequence."""
    perm = torch.arange(2 ** n)
    for a, b in pairs:
        idx = torch.arange(2 ** n)
        ba, bb = (idx >> (n - 1 - a)) & 1, (idx >> (n - 1 - b)) & 1
        src = idx ^ ((ba ^ bb) * ((1 << (n - 1 - a)) | (1 << (n - 1 - b))))
        perm = perm[src]
    return perm


def rz_ry_matrices(ry_angles: torch.Tensor, rz_angles: torch.Tensor) -> torch.Tensor:
    """Matrices RZ(beta) . RY(alpha) (RY first, then RZ) for every sample and qubit: angles (B, n) ->
    (B, n, 2, 2), complex. RZ(beta) = diag(exp(-i beta/2), exp(+i beta/2)), as in Qiskit."""
    c, s = torch.cos(ry_angles / 2), torch.sin(ry_angles / 2)
    one = torch.ones_like(rz_angles)
    e_minus, e_plus = torch.polar(one, -rz_angles / 2), torch.polar(one, rz_angles / 2)
    return torch.stack((torch.stack((e_minus * c, -e_minus * s), -1),
                        torch.stack((e_plus * s, e_plus * c), -1)), -2)


def single_qubit_layer(state: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """Apply the 2x2 matrix U[:, j] (one per sample) to qubit n-k+j for j = 0..k-1: one gate on each of
    the last k qubits (the least significant ones). U: (B, k, 2, 2).
    Two implementations of the same function. Without gradient (data collection, small batches): the
    whole layer as a 2^k x 2^k matrix per sample (kron_chain) and one product, the fastest. With gradient
    (updates, large batches): one gate at a time on amplitude pairs, faster in the backward."""
    B, k = state.shape[0], U.shape[1]
    n = n_qubits(state)
    if U.is_complex():
        state = as_complex(state)
    U = U.to(state.dtype)
    if not torch.is_grad_enabled():
        return (state.view(B, -1, 2 ** k) @ kron_chain(U).transpose(1, 2)).reshape(B, -1)
    for j in range(k):
        state = gate_1q(state, n - k + j, U[:, j])
    return state
