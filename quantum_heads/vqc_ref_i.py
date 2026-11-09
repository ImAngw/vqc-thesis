"""
Qiskit reference of ansatz I (3 qubits, three data uploads with multiplicative feature weights), written from the
description only (never from the PyTorch module).

Qubit i of the description is Qiskit qubit N - 1 - i (the PyTorch code is big-endian, Qiskit little-endian).

Input layout: features = [perception M (4) | navigator N (4) | memory G (4)], branch b = features[4b : 4b + 4].
Circuit: three blocks r = 1, 2, 3; in block r qubit i receives branch X = (i + r - 1) mod 3 (M -> N -> G cyclic, so
after block 3 every qubit has seen all 12 features):
           q_i: RY(s{r}[4i] X[0]) RZ(s{r}[4i+1] X[1]) RY(s{r}[4i+2] X[2]) RZ(s{r}[4i+3] X[3])
                RZ(u{r}[3i]) RY(u{r}[3i+1]) RZ(u{r}[3i+2])
           CX (2,1), (2,0), (1,0)   [(control, target)]
           q_i: RZ(v{r}[3i]) RY(v{r}[3i+1]) RZ(v{r}[3i+2])
           CX (2,1), (2,0), (1,0)
         then RY(wf[i]) on every qubit.
Parameters: s{r} (12), u{r} (9), v{r} (9) for r = 1..3, wf (3): 93.
Readout: Born probabilities of q0-q2 (8 strings = 8 actions), logits = log P.
"""
from __future__ import annotations

import sys

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate, RYGate, RZGate
from qiskit.quantum_info import Operator, Statevector

N = 3
N_MEASURED = 3
FEATURES = 12
REPS = (1, 2, 3)
BRANCHES = "MNG"
CX_CHAIN = [(2, 1), (2, 0), (1, 0)]
PARAM_SIZES = {**{f"{p}{r}": n for r in REPS for p, n in (("s", 12), ("u", 9), ("v", 9))}, "wf": 3}
PARAM_NAMES = [k for k in PARAM_SIZES if PARAM_SIZES[k]]

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
            b = BRANCHES[(i + r - 1) % 3]
            for k, axis in enumerate(("ry", "rz", "ry", "rz")):
                w = params[f"s{r}"][4 * i + k]
                yield axis, w * branch[b][k], [i], f"s{r}[{4 * i + k}] * {b}[{k}]"
            for j, axis in enumerate(("rz", "ry", "rz")):
                yield axis, params[f"u{r}"][3 * i + j], [i], f"u{r}[{3 * i + j}]"
        for c, t in CX_CHAIN:
            yield "cx", None, [c, t], "-"
        for i in range(N):
            for j, axis in enumerate(("rz", "ry", "rz")):
                yield axis, params[f"v{r}"][3 * i + j], [i], f"v{r}[{3 * i + j}]"
        for c, t in CX_CHAIN:
            yield "cx", None, [c, t], "-"
    for i in range(N):
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


def logits(params: dict, features: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(register_probabilities(statevector(params, features)), 1e-12))


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
