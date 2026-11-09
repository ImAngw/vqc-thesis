"""
Qiskit reference of ansatz K (4 qubits, three data uploads, the navigator in every data gate plus a rotating second
branch, Born readout on q0-q2),
written from the description only (never from the PyTorch module).

Qubit i of the description is Qiskit qubit N - 1 - i (the PyTorch code is big-endian, Qiskit little-endian).

Input layout: features = [perception M (4) | navigator N (4) | memory G (4)], qubit i reads index i of each branch.
Circuit: three blocks r = 1, 2, 3. Data gate k = 0, 1, 2 (RY, RZ, RY) of qubit i has angle
  a_k = d{r}[3i+k] N[i] (+ c{r}[2i+j] X[i] for the j-th gate of the qubit that has a second branch X), with
  r = 1: N, N+M, N+G      r = 2: N+G, N, N+M      r = 3: N+M, N+G, N
           q_i: RY(a_0) RZ(a_1) RY(a_2) RZ(u{r}[3i]) RY(u{r}[3i+1]) RZ(u{r}[3i+2])
           CX (3,2), (3,1), (3,0), (2,1), (2,0), (1,0)   [(control, target)]
           q0-q2: RZ(v{r}[3i]) RY(v{r}[3i+1]) RZ(v{r}[3i+2]);  q3: RY(v{r}[9]) RZ(v{r}[10]) RY(v{r}[11])
         then RY(wf[i]) on q0, q1, q2.
Gates without effect on the output are left out of block 3: the last RZ(u3) of q3 (q3 is only a control in the
last CX chain, which an RZ commutes with) and the whole rotation of q3 after the last CX chain (never measured).
Parameters: d{r} (12, navigator weights) and c{r} (8, weights of M / G) for r = 1..3; u1, u2 (12), u3 (11);
v1, v2 (12), v3 (9); wf (3): 131.
Qubit 3 is an ancilla (never measured).
Readout: Born probabilities of q0-q2; action a is the outcome code(a) (gray by default: a ^ (a >> 1); binary: a),
logits = log P(code(a)).
"""
from __future__ import annotations

import sys

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate, RYGate, RZGate
from qiskit.quantum_info import Operator, Statevector

N = 4
N_MEASURED = 3
FEATURES = 12
REPS = (1, 2, 3)
SECOND_BRANCH = {1: "-MG", 2: "G-M", 3: "MG-"}  # per data gate k, "-" = navigator only
CX_CHAIN = [(3, 2), (3, 1), (3, 0), (2, 1), (2, 0), (1, 0)]
PARAM_SIZES = {"d1": 12, "c1": 8, "u1": 12, "v1": 12, "d2": 12, "c2": 8, "u2": 12, "v2": 12,
               "d3": 12, "c3": 8, "u3": 11, "v3": 9, "wf": 3}
PARAM_NAMES = [k for k in PARAM_SIZES if PARAM_SIZES[k]]
ACTION_CODES = {"binary": [0, 1, 2, 3, 4, 5, 6, 7], "gray": [0, 1, 3, 2, 6, 7, 5, 4]}

GATES = {"ry": RYGate, "rz": RZGate, "cx": CXGate}


def q(i: int) -> int:
    return N - 1 - i


def _gate(qc: QuantumCircuit, name: str, angle: float | None, qubits: list[int], labelled: bool) -> None:
    gate = GATES[name]() if angle is None else GATES[name](angle)
    if labelled:
        qc.unitary(Operator(gate).data, [q(i) for i in qubits], label=name)
    else:
        qc.append(gate, [q(i) for i in qubits])


def ops(params: dict, features: np.ndarray):
    """The gates of the description in order: (name, angle or None, qubits, symbolic angle for the table)."""
    branch = {"M": features[0:4], "N": features[4:8], "G": features[8:12]}
    for r in REPS:
        for i in range(N):
            j = 0  # index of the next second-branch weight of this qubit
            for k, (axis, b) in enumerate(zip(("ry", "rz", "ry"), SECOND_BRANCH[r])):
                angle = params[f"d{r}"][3 * i + k] * branch["N"][i]
                sym = f"d{r}[{3 * i + k}] * N[{i}]"
                if b != "-":
                    angle = angle + params[f"c{r}"][2 * i + j] * branch[b][i]
                    sym += f" + c{r}[{2 * i + j}] * {b}[{i}]"
                    j += 1
                yield axis, angle, [i], sym
            for t, axis in enumerate(("rz", "ry", "rz")):
                if r == 3 and i == 3 and t == 2:
                    continue  # dead: RZ on a control-only, unmeasured qubit
                yield axis, params[f"u{r}"][3 * i + t], [i], f"u{r}[{3 * i + t}]"
        for c, t in CX_CHAIN:
            yield "cx", None, [c, t], "-"
        for i in range(N if r < 3 else N_MEASURED):
            axes = ("ry", "rz", "ry") if i == 3 else ("rz", "ry", "rz")
            for t, axis in enumerate(axes):
                yield axis, params[f"v{r}"][3 * i + t], [i], f"v{r}[{3 * i + t}]"
    for i in range(N_MEASURED):
        yield "ry", params["wf"][i], [i], f"wf[{i}]"


def build_circuit(params: dict, features: np.ndarray, labelled: bool = False) -> QuantumCircuit:
    qc = QuantumCircuit(N)
    for name, angle, qubits, _ in ops(params, features):
        _gate(qc, name, angle, qubits, labelled)
    return qc


def statevector(params: dict, features: np.ndarray) -> np.ndarray:
    return Statevector(build_circuit(params, features)).data


def register_probabilities(sv_data: np.ndarray) -> np.ndarray:
    """P(register state a), qubit 0 the most significant bit (Qiskit: qargs[0] is the least significant)."""
    return Statevector(sv_data).probabilities([q(k) for k in reversed(range(N_MEASURED))])


def logits(params: dict, features: np.ndarray, action_code: str = "gray") -> np.ndarray:
    """logit[a] = log P(outcome code(a)), P floored at 1e-12."""
    log_p = np.log(np.maximum(register_probabilities(statevector(params, features)), 1e-12))
    return log_p[ACTION_CODES[action_code]]


def symbolic_table() -> str:
    """Gate-by-gate table: the artifact the user confirms before the PyTorch module is written."""
    rows = ["| # | gate | qubits (description) | Qiskit qubits | parameter |", "|---|---|---|---|---|"]
    demo = {k: np.zeros(n or 1) for k, n in PARAM_SIZES.items()}
    for i, (name, _, qubits, sym) in enumerate(ops(demo, np.zeros(FEATURES)), start=1):
        rows.append(f"| {i} | {name.upper()} | {', '.join(f'q{x}' for x in qubits)} | "
                    f"{', '.join(f'q_{q(x)}' for x in qubits)} | {sym} |")
    return "\n".join(rows)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--table":
        print(symbolic_table())
    else:
        rng = np.random.default_rng(0)
        demo = {k: np.round(rng.normal(size=n or None) * 0.5, 2) for k, n in PARAM_SIZES.items()}
        qc = build_circuit(demo, np.round(rng.normal(size=FEATURES), 2))
        print(qc.draw(output="text", fold=200))
        print(dict(qc.count_ops()), "parameters:", sum(PARAM_SIZES.values()))
