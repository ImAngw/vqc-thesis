"""
VQCHeadI in pure PyTorch (ansatz I, registered as "vqc-optim": 3 qubits, three data uploads with multiplicative
feature weights; generated from the description, checked against vqc_ref_i.py).

Conventions: big-endian qubit order (qubit 0 = most significant bit of the basis index), as in vqc_core.
Qiskit qubit = 2 - qubit here. Features arrive as [perception M (4) | navigator N (4) | memory G (4)]. All three
qubits are measured, no ancilla. The state is complex (RZ).

Circuit: three blocks r = 1, 2, 3. In block r qubit i receives branch X = (i + r - 1) mod 3 of (M, N, G) and runs
RY(s_r[4i] X0) RZ(s_r[4i+1] X1) RY(s_r[4i+2] X2) RZ(s_r[4i+3] X3) RZ(u_r[3i]) RY(u_r[3i+1]) RZ(u_r[3i+2]);
then CX (2,1), (2,0), (1,0); RZ RY RZ (v_r) on every qubit; CX (2,1), (2,0), (1,0). Finally RY(wf) on every qubit.
Clean path: the seven data gates of each qubit collapse into one 2x2 matrix per sample (the two adjacent RZ merge),
RZ(u2) RY(u1) . RZ(u0 + s3 X3) RY(s2 X2) . RZ(s1 X1) RY(s0 X0); block 1 on |000> is then a product state. The
data-independent stretch (CX chain, v_r, CX chain, plus RY(wf) after block 3) is fused into one 8 x 8 matrix per
block, rebuilt at every forward with gradient and cached without gradient until a parameter changes.
No noisy path (density_matrix not implemented).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from vqc_core import statevector as sv
from vqc_core.vqc_base import VQCBase

N_QUBITS, N_MEASURED = 3, 3
FEATURES = 4  # per branch
CX_CHAIN = [(2, 1), (2, 0), (1, 0)]
REPS = (1, 2, 3)
_ACTIONS = lambda: torch.arange(2 ** N_MEASURED)  # noqa: E731
ACTION_CODES = {"binary": _ACTIONS, "gray": lambda: _ACTIONS() ^ (_ACTIONS() >> 1)}


def _complex_dtype(x: torch.Tensor) -> torch.dtype:
    return torch.complex64 if x.dtype == torch.float32 else torch.complex128


class VQCHeadI(VQCBase):
    """93 parameters: s{1,2,3} (12, feature weights, init scale_init), u{1,2,3} and v{1,2,3} (9 each) and wf (3),
    init randn * angle_std (wf shifted by wf_offset). logit[a] = log P(q0 q1 q2 = code(a)), code = action_code.
    Small scale_init and angle_std with wf_offset = pi/2 start near |+++>: a uniform Born policy that barely
    depends on the features (as an MLP head's small output layer does)."""
    n_qubits, n_measured = N_QUBITS, N_MEASURED

    def __init__(self, scale_init: float = 1.0, angle_std: float = 0.5, wf_offset: float = 0.0,
                 action_code: str = "binary"):
        super().__init__()
        if action_code not in ACTION_CODES:
            raise ValueError(f"action_code must be one of {tuple(ACTION_CODES)}, got {action_code!r}")
        # action a is the measured outcome code(a): a fixed relabeling, the action stays a measurement outcome.
        # "gray": a ^ (a >> 1), so actions with adjacent indices (compass neighbours in rl_env MOVES, 7 -> 0
        # included) are outcomes one bit apart. Registered before the parameters: it draws no random numbers
        self.register_buffer("outcome_of_action", ACTION_CODES[action_code](), persistent=False)
        for r in REPS:
            setattr(self, f"s{r}", nn.Parameter(torch.full((N_QUBITS * FEATURES,), scale_init)))
            setattr(self, f"u{r}", nn.Parameter(torch.randn(3 * N_QUBITS) * angle_std))
            setattr(self, f"v{r}", nn.Parameter(torch.randn(3 * N_QUBITS) * angle_std))
        self.wf = nn.Parameter(torch.randn(N_QUBITS) * angle_std + wf_offset)
        self.register_buffer("cx", sv.cx_permutation(CX_CHAIN, N_QUBITS), persistent=False)

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

    def data_matrices(self, features: torch.Tensor, r: int) -> torch.Tensor:
        """(B, 3, 2, 2): the seven single-qubit gates of block r on each qubit, as one matrix per sample."""
        branches = features.view(-1, 3, FEATURES)
        x = torch.roll(branches, shifts=-(r - 1), dims=1)  # x[:, i] = branch (i + r - 1) mod 3
        s = getattr(self, f"s{r}").view(N_QUBITS, FEATURES)
        u = getattr(self, f"u{r}").view(N_QUBITS, 3)
        a = s * x                                          # (B, 3, 4) data angles
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

    @staticmethod
    def readout(probs: torch.Tensor) -> torch.Tensor:
        """logit[a] = log P(a), with P floored at 1e-12."""
        return torch.log(probs.clamp_min(1e-12))

    def forward(self, features: torch.Tensor, noise=None) -> torch.Tensor:
        """Logits indexed by action: logit[a] = log P(outcome_of_action[a])."""
        return super().forward(features, noise)[:, self.outcome_of_action]
