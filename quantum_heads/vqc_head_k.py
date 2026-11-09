"""
VQCHeadK in pure PyTorch (ansatz K: 4 qubits, three data uploads, the navigator in every data gate plus a rotating
second branch, Born readout on q0-q2; generated from the description, checked against vqc_ref_k.py).

Conventions: big-endian qubit order (qubit 0 = most significant bit of the basis index), as in vqc_core.
Qiskit qubit = 3 - qubit here. Features arrive as [perception M (4) | navigator N (4) | memory G (4)]; qubit i reads
index i of each branch. q0-q2 are measured, q3 is an ancilla. The state is complex (RZ).

Circuit: three blocks r = 1, 2, 3. Data gate k = 0, 1, 2 of qubit i has angle a_k = d_r[3i+k] N_i, plus
c_r[2i+j] X_i on the j-th gate with a second branch X: r = 1: (N, N+M, N+G), r = 2: (N+G, N, N+M), r = 3: (N+M, N+G, N).
Qubit i runs RY(a_0) RZ(a_1) RY(a_2) RZ(u_r[3i]) RY(u_r[3i+1]) RZ(u_r[3i+2])
then CX (3,2), (3,1), (3,0), (2,1), (2,0), (1,0); then RZ RY RZ (v_r) on q0-q2 and RY RZ RY (v_r[9:12]) on q3.
Block 3 leaves out the gates without effect on the output: the last RZ(u3) of q3 and the rotation of q3 after the
last CX chain (so u3 has 11 entries, v3 9). Finally RY(wf) on q0-q2.
Clean path: the data gates and RZ(u_r[3i]) of each qubit collapse into one 2x2 matrix per sample; RY RZ (u_r), the
CX chain and the v_r rotations (plus RY(wf) after block 3) into one cached 16 x 16 matrix per block.
No noisy path (density_matrix not implemented).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from vqc_core import statevector as sv
from vqc_core.vqc_base import VQCBase

N_QUBITS, N_MEASURED = 4, 3
FEATURES = 4  # per branch
ANCILLA = 3
CX_CHAIN = [(3, 2), (3, 1), (3, 0), (2, 1), (2, 0), (1, 0)]
REPS = (1, 2, 3)
BRANCH_M, BRANCH_N, BRANCH_G = 0, 1, 2
# (data gate k, branch) of the two second-branch terms of each qubit, in the order of c_r[2i], c_r[2i+1]
SECOND_BRANCH = {1: ((1, BRANCH_M), (2, BRANCH_G)), 2: ((0, BRANCH_G), (2, BRANCH_M)), 3: ((0, BRANCH_M), (1, BRANCH_G))}
U_SIZES = {1: 12, 2: 12, 3: 11}
V_SIZES = {1: 12, 2: 12, 3: 9}
_ACTIONS = lambda: torch.arange(2 ** N_MEASURED)  # noqa: E731
ACTION_CODES = {"binary": _ACTIONS, "gray": lambda: _ACTIONS() ^ (_ACTIONS() >> 1)}


def _complex_dtype(x: torch.Tensor) -> torch.dtype:
    return torch.complex64 if x.dtype == torch.float32 else torch.complex128


class VQCHeadK(VQCBase):
    """131 parameters: d{1,2,3} (12, navigator weights) and c{1,2,3} (8, weights of M / G), init scale_init; u1, u2
    (12), u3 (11), v1, v2 (12), v3 (9) and
    wf (3), init randn * angle_std (wf shifted by wf_offset). logit[a] = log P(q0 q1 q2 = code(a)), code =
    action_code (gray by default: a ^ (a >> 1); binary: a)."""
    n_qubits, n_measured = N_QUBITS, N_MEASURED

    def __init__(self, scale_init: float = 1.0, angle_std: float = 0.5, wf_offset: float = 0.0,
                 action_code: str = "gray"):
        super().__init__()
        if action_code not in ACTION_CODES:
            raise ValueError(f"action_code must be one of {tuple(ACTION_CODES)}, got {action_code!r}")
        # registered before the parameters: it draws no random numbers
        self.register_buffer("outcome_of_action", ACTION_CODES[action_code](), persistent=False)
        for r in REPS:
            setattr(self, f"d{r}", nn.Parameter(torch.full((3 * N_QUBITS,), scale_init)))
            setattr(self, f"c{r}", nn.Parameter(torch.full((2 * N_QUBITS,), scale_init)))
            setattr(self, f"u{r}", nn.Parameter(torch.randn(U_SIZES[r]) * angle_std))
            setattr(self, f"v{r}", nn.Parameter(torch.randn(V_SIZES[r]) * angle_std))
        self.wf = nn.Parameter(torch.randn(N_MEASURED) * angle_std + wf_offset)
        self.register_buffer("cx", sv.cx_permutation(CX_CHAIN, N_QUBITS), persistent=False)

    def fixed_blocks(self) -> list[torch.Tensor]:
        """[W1, W2, W3]: RY RZ (u_r, the part after the data), CX chain, v_r rotations; W3 also ends with RY(wf)."""
        def stretch(r):
            def apply(state):
                u, v = getattr(self, f"u{r}"), getattr(self, f"v{r}")
                for k in range(N_QUBITS):
                    state = sv.ry(state, k, u[3 * k + 1])
                    if not (r == 3 and k == ANCILLA):
                        state = sv.rz(state, k, u[3 * k + 2])
                state = state[:, self.cx]
                for k in range(N_QUBITS if r < 3 else N_MEASURED):
                    first, mid = (sv.ry, sv.rz) if k == ANCILLA else (sv.rz, sv.ry)
                    state = first(state, k, v[3 * k])
                    state = mid(state, k, v[3 * k + 1])
                    state = first(state, k, v[3 * k + 2])
                if r == REPS[-1]:
                    for k in range(N_MEASURED):
                        state = sv.ry(state, k, self.wf[k])
                return state
            return apply
        dtype = _complex_dtype(self.v1)
        return [sv.block_matrix(stretch(r), N_QUBITS, self.v1).to(dtype) for r in REPS]

    def data_matrices(self, features: torch.Tensor, r: int) -> torch.Tensor:
        """(B, 4, 2, 2): RY(a0) RZ(a1) RY(a2) RZ(u_r[3i]) on each qubit i of block r, one matrix per sample."""
        branches = features.view(-1, 3, FEATURES)                                  # (B, branch, qubit)
        a = branches[:, BRANCH_N, :, None] * getattr(self, f"d{r}").view(N_QUBITS, 3)  # (B, qubit, gate)
        c = getattr(self, f"c{r}").view(N_QUBITS, 2)
        extra = torch.zeros_like(a)
        for j, (k, b) in enumerate(SECOND_BRANCH[r]):
            extra[..., k] = c[:, j] * branches[:, b]
        a = a + extra
        u0 = getattr(self, f"u{r}")[0::3][:N_QUBITS].expand_as(a[..., 0])
        return sv.rz_ry_matrices(a[..., 2], u0) @ sv.rz_ry_matrices(a[..., 0], a[..., 1])

    def state(self, features: torch.Tensor) -> torch.Tensor:
        """Complex statevector (B, 16) on the features (B, 12)."""
        state = None
        for r, W in zip(REPS, self.blocks()):
            U = self.data_matrices(features, r)
            if state is None:  # block 1 on |0000>: product state
                state = sv.kron_chain(U[..., 0:1]).squeeze(-1)
            else:
                state = sv.single_qubit_layer(state, U)
            state = state @ W
        return state

    @staticmethod
    def readout(probs: torch.Tensor) -> torch.Tensor:
        """logit[a] = log P(a), with P floored at 1e-12."""
        return torch.log(probs.clamp_min(1e-12))

    def forward(self, features: torch.Tensor, noise=None) -> torch.Tensor:
        """Logits indexed by action: logit[a] = log P(outcome_of_action[a])."""
        return super().forward(features, noise)[:, self.outcome_of_action]
