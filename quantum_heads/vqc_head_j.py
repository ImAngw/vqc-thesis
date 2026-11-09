"""
VQCHeadJ in pure PyTorch (ansatz J, registered as "vqc-nav-id": 3 qubits, three data uploads, the navigator on every
qubit of every block; generated from the description, checked against vqc_ref_j.py).

Conventions: big-endian qubit order (qubit 0 = most significant bit of the basis index), as in vqc_core.
Qiskit qubit = 2 - qubit here. Features arrive as [perception M (4) | navigator N (4) | memory G (4)]. All three
qubits are measured, no ancilla. The state is complex (RZ).

Circuit: three blocks r = 1, 2, 3. In block r qubit i has role (i + r - 1) mod 3: A = navigator only, B = navigator
+ perception, C = navigator + memory. Gate k = 0..3 of qubit i reads feature j = (k + i) mod 4 of each branch:
  A: a_k = w_r[4i+k] N_j      B: a_k = w_r[4i+k] N_j + c_r[k] M_j      C: a_k = w_r[4i+k] N_j + c_r[4+k] G_j
and the qubit runs RY(a0) RZ(a1) RY(a2) RZ(a3) RZ(u_r[3i]) RY(u_r[3i+1]) RZ(u_r[3i+2]); then CX (2,1), (2,0), (1,0);
RZ RY RZ (v_r) on every qubit; CX (2,1), (2,0), (1,0). Finally RY(wf) on every qubit.
Clean path: as in VQCHeadI, the seven data gates of each qubit collapse into one 2x2 matrix per sample and the
data-independent stretch of each block (plus RY(wf) after block 3) into one cached 8 x 8 matrix.
Noisy path: density_matrix applies every nominal gate separately, each followed by its depolarizing event (p1 after
RY / RZ, p2 after CX), in the order of the description; readout error on the three measured qubits (VQCBase).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from vqc_core import density as dm
from vqc_core import noise as nz
from vqc_core import statevector as sv
from vqc_core.vqc_base import VQCBase

N_QUBITS, N_MEASURED = 3, 3
FEATURES = 4  # per branch
CX_CHAIN = [(2, 1), (2, 0), (1, 0)]
REPS = (1, 2, 3)
ROLE_A, ROLE_B, ROLE_C = 0, 1, 2
_ACTIONS = lambda: torch.arange(2 ** N_MEASURED)  # noqa: E731
ACTION_CODES = {"binary": _ACTIONS, "gray": lambda: _ACTIONS() ^ (_ACTIONS() >> 1)}


def _complex_dtype(x: torch.Tensor) -> torch.dtype:
    return torch.complex64 if x.dtype == torch.float32 else torch.complex128


class VQCHeadJ(VQCBase):
    """117 parameters: w{1,2,3} (12, navigator weights) and c{1,2,3} (8, weights of M on the B qubit then of G on the
    C qubit), init scale_init; u{1,2,3} and v{1,2,3} (9 each) and wf (3), init randn * angle_std (wf shifted by
    wf_offset). logit[a] = log P(q0 q1 q2 = code(a)), code = action_code (see VQCHeadI).
    Small scale_init and angle_std with wf_offset = pi/2 start near |+++>, a near-uniform Born policy."""
    n_qubits, n_measured = N_QUBITS, N_MEASURED

    def __init__(self, scale_init: float = 1.0, angle_std: float = 0.5, wf_offset: float = 0.0,
                 action_code: str = "binary"):
        super().__init__()
        if action_code not in ACTION_CODES:
            raise ValueError(f"action_code must be one of {tuple(ACTION_CODES)}, got {action_code!r}")
        # registered before the parameters: it draws no random numbers
        self.register_buffer("outcome_of_action", ACTION_CODES[action_code](), persistent=False)
        for r in REPS:
            setattr(self, f"w{r}", nn.Parameter(torch.full((N_QUBITS * FEATURES,), scale_init)))
            setattr(self, f"c{r}", nn.Parameter(torch.full((2 * FEATURES,), scale_init)))
            setattr(self, f"u{r}", nn.Parameter(torch.randn(3 * N_QUBITS) * angle_std))
            setattr(self, f"v{r}", nn.Parameter(torch.randn(3 * N_QUBITS) * angle_std))
        self.wf = nn.Parameter(torch.randn(N_QUBITS) * angle_std + wf_offset)
        self.register_buffer("cx", sv.cx_permutation(CX_CHAIN, N_QUBITS), persistent=False)
        # shift[i, k] = (k + i) mod 4: the feature of each branch read by gate k of qubit i
        shift = (torch.arange(FEATURES)[None, :] + torch.arange(N_QUBITS)[:, None]) % FEATURES
        self.register_buffer("shift", shift, persistent=False)

    def fixed_blocks(self) -> list[torch.Tensor]:
        """[W1, W2, W3]: CX chain, RZ RY RZ (v_r), CX chain; W3 also ends with RY(wf)."""
        def stretch(r):
            def apply(state):
                state = state[:, self.cx]
                v = getattr(self, f"v{r}")
                for k in range(N_QUBITS):
                    state = sv.rz(state, k, v[3 * k])
                    state = sv.ry(state, k, v[3 * k + 1])
                    state = sv.rz(state, k, v[3 * k + 2])
                state = state[:, self.cx]
                if r == REPS[-1]:
                    for k in range(N_QUBITS):
                        state = sv.ry(state, k, self.wf[k])
                return state
            return apply
        dtype = _complex_dtype(self.v1)
        return [sv.block_matrix(stretch(r), N_QUBITS, self.v1).to(dtype) for r in REPS]

    def data_angles(self, features: torch.Tensor, r: int) -> torch.Tensor:
        """(B, 3, 4): the four data angles of every qubit in block r."""
        m, nav, g = (features[:, FEATURES * b: FEATURES * (b + 1)][:, self.shift] for b in range(3))  # (B, 3, 4) each
        w = getattr(self, f"w{r}").view(N_QUBITS, FEATURES)
        c = getattr(self, f"c{r}").view(2, FEATURES)  # row 0: the B qubit, row 1: the C qubit
        angles = []
        for i in range(N_QUBITS):
            a = w[i] * nav[:, i]
            role = (i + r - 1) % 3
            if role == ROLE_B:
                a = a + c[0] * m[:, i]
            elif role == ROLE_C:
                a = a + c[1] * g[:, i]
            angles.append(a)
        return torch.stack(angles, dim=1)

    def data_matrices(self, features: torch.Tensor, r: int) -> torch.Tensor:
        """(B, 3, 2, 2): the seven single-qubit gates of block r on each qubit, as one matrix per sample."""
        a = self.data_angles(features, r)
        u = getattr(self, f"u{r}").view(N_QUBITS, 3)
        first = sv.rz_ry_matrices(a[..., 0], a[..., 1])
        second = sv.rz_ry_matrices(a[..., 2], a[..., 3] + u[:, 0])
        third = sv.rz_ry_matrices(u[:, 1].expand_as(a[..., 0]), u[:, 2].expand_as(a[..., 0]))
        return third @ second @ first

    def state(self, features: torch.Tensor) -> torch.Tensor:
        """Complex statevector (B, 8) on the features (B, 12)."""
        state = None
        for r, W in zip(REPS, self.blocks()):
            U = self.data_matrices(features, r)
            if state is None:  # block 1 on |000>: product state
                state = sv.kron_chain(U[..., 0:1]).squeeze(-1)
            else:
                state = sv.single_qubit_layer(state, U)
            state = state @ W
        return state

    def density_matrix(self, features: torch.Tensor, noise: nz.NoiseParams) -> torch.Tensor:
        """rho (B, 8, 8); one depolarizing event per nominal gate, in the order of the description."""
        rho = dm.zero_density(features.shape[0], N_QUBITS, features)

        def rot(rho, gate, qubit, angle):
            return nz.noisy_gate(rho, gate, (qubit,), (angle,), noise.p1)

        def chain(rho):
            for c, t in CX_CHAIN:
                rho = nz.noisy_permutation(rho, sv.cx_permutation([(c, t)], N_QUBITS), (c, t), noise.p2)
            return rho

        for r in REPS:
            a = self.data_angles(features, r)
            u, v = getattr(self, f"u{r}"), getattr(self, f"v{r}")
            for i in range(N_QUBITS):
                for k, gate in enumerate((sv.ry, sv.rz, sv.ry, sv.rz)):
                    rho = rot(rho, gate, i, a[:, i, k])
                for t, gate in enumerate((sv.rz, sv.ry, sv.rz)):
                    rho = rot(rho, gate, i, u[3 * i + t])
            rho = chain(rho)
            for i in range(N_QUBITS):
                for t, gate in enumerate((sv.rz, sv.ry, sv.rz)):
                    rho = rot(rho, gate, i, v[3 * i + t])
            rho = chain(rho)
        for i in range(N_QUBITS):
            rho = rot(rho, sv.ry, i, self.wf[i])
        return rho

    @staticmethod
    def readout(probs: torch.Tensor) -> torch.Tensor:
        """logit[a] = log P(a), with P floored at 1e-12."""
        return torch.log(probs.clamp_min(1e-12))

    def forward(self, features: torch.Tensor, noise=None) -> torch.Tensor:
        """Logits indexed by action: logit[a] = log P(outcome_of_action[a])."""
        return super().forward(features, noise)[:, self.outcome_of_action]
