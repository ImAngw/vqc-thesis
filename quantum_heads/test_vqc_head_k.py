"""
VQCHeadK (vqc_head_k.py) against the Qiskit reference (vqc_ref_k.py), written from the description.

  1. clean path vs Qiskit: statevector, register probabilities and logits (Gray and binary action codes) on
     N_TRIALS random parameter sets and inputs, both precisions
  2. grad and no-grad paths agree; the block cache follows in-place updates
  3. gradient of the register probabilities w.r.t. every parameter vs Qiskit: exact parameter-shift for the free
     angles (u, v, wf), central finite differences for the feature weights d, c (angle = d * N + c * M or G)
  4. guards: outputs vary, every input branch matters, no dead parameter, N reaches every data gate of its qubit,
     M and G the gates given by the block
  5. the gates left out of block 3 (last RZ of q3, rotation of q3 after the last CX chain) have no effect in Qiskit
No noisy path: density_matrix is not implemented for this ansatz.

Run:  .venv/bin/python -m pytest quantum_heads/test_vqc_head_k.py -q
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch
from qiskit.quantum_info import Statevector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import vqc_testkit as kit
import vqc_ref_k as ref
from vqc_head_k import VQCHeadK

N_TRIALS = 50
TOL = kit.TOL
FEATURES = 12
rng = np.random.default_rng(0)


@pytest.fixture(params=[torch.float32, torch.float64], ids=["f32", "f64"])
def dtype(request):
    return request.param


def random_params(scale: float) -> dict:
    return kit.random_params(rng, ref.PARAM_SIZES, scale)


def make_model(params: dict, dtype, **ctor) -> VQCHeadK:
    return kit.make_model(VQCHeadK, params, dtype, **ctor)


def test_parameter_count():
    model = VQCHeadK()
    assert sum(p.numel() for p in model.parameters()) == 131
    assert {k: p.numel() for k, p in model.named_parameters()} == ref.PARAM_SIZES


def test_init():
    model = VQCHeadK(scale_init=0.1)
    for r in ref.REPS:
        assert torch.allclose(getattr(model, f"d{r}"), torch.full((12,), 0.1))
        assert torch.allclose(getattr(model, f"c{r}"), torch.full((8,), 0.1))
    torch.manual_seed(0)
    model = VQCHeadK(angle_std=0.05, wf_offset=np.pi / 2)
    torch.manual_seed(0)
    for r in ref.REPS:
        assert torch.equal(getattr(model, f"u{r}"), torch.randn(ref.PARAM_SIZES[f"u{r}"]) * 0.05)
        assert torch.equal(getattr(model, f"v{r}"), torch.randn(ref.PARAM_SIZES[f"v{r}"]) * 0.05)
    assert torch.equal(model.wf, torch.randn(3) * 0.05 + np.pi / 2)
    assert torch.equal(model.outcome_of_action, torch.tensor(ref.ACTION_CODES["gray"]))  # gray is the default


@pytest.mark.parametrize("seed", range(5))
def test_near_identity_init_starts_from_a_near_uniform_policy(seed):
    x = torch.as_tensor(rng.uniform(-1, 1, (64, FEATURES)), dtype=torch.float32)
    torch.manual_seed(seed)
    # the registered init ("vqc-k"): as VQCHeadJ's "vqc-nav-id" but angle_std 0.03 instead of 0.05, since K has more
    # free angles (with 0.05, 2 of 20 seeds started below 2.0)
    probs = VQCHeadK(scale_init=0.05, angle_std=0.03, wf_offset=np.pi / 2)(x).softmax(-1)
    assert (-(probs * probs.log()).sum(-1)).mean() > 2.0  # ln 8 = 2.079


@pytest.mark.parametrize("code", ["gray", "binary"])
def test_clean_path_matches_qiskit(dtype, code):
    for _ in range(N_TRIALS):
        params = random_params(rng.choice([0.5, 1.5]))
        features = rng.uniform(-1, 1, size=(2, FEATURES))
        model = make_model(params, dtype, action_code=code)
        x = torch.tensor(features, dtype=dtype)
        with torch.no_grad():
            state, probs, logits = model.state(x), model.measured_probabilities(x), model(x)
        for b in range(2):
            ref_state = ref.statevector(params, features[b])
            ref_probs = ref.register_probabilities(ref_state)
            assert np.abs(state[b].numpy() - ref_state).max() < TOL[dtype]
            assert np.abs(probs[b].numpy() - ref_probs).max() < TOL[dtype]
            # logits are log P: weight the error by P (of the same action), where P is not negligible
            ref_logits = ref.logits(params, features[b], code)
            err = np.abs(logits[b].numpy() - ref_logits) * np.exp(ref_logits)
            assert err.max() < 10 * TOL[dtype]


def test_grad_path_matches_qiskit():
    """The gradient branch of single_qubit_layer (gate by gate) against Qiskit directly."""
    params = random_params(1.0)
    features = rng.uniform(-1, 1, size=(3, FEATURES))
    model = make_model(params, torch.float64)
    state = model.state(torch.tensor(features))
    for b in range(3):
        assert np.abs(state[b].detach().numpy() - ref.statevector(params, features[b])).max() < 1e-10


def test_grad_and_no_grad_paths_agree(dtype):
    model = make_model(random_params(1.0), dtype)
    x = torch.tensor(rng.uniform(-1, 1, size=(4, FEATURES)), dtype=dtype)
    with torch.no_grad():
        a = model(x)
    assert torch.allclose(a, model(x), atol=10 * TOL[dtype])


@pytest.mark.parametrize("name", ["u1", "v1", "v2", "u3", "v3", "wf", "d3", "c2"])
def test_block_cache_follows_parameter_updates(name):
    params = random_params(1.0)
    model = make_model(params, torch.float64)
    x = torch.tensor(rng.uniform(-1, 1, size=(2, FEATURES)))
    with torch.no_grad():
        model(x)                                   # fill the cache
        state_before = model.state(x)
        getattr(model, name).add_(0.3)             # in-place, as optimizer.step does
        after, state_after = model.measured_probabilities(x), model.state(x)
    assert not torch.allclose(state_before, state_after)
    params[name] = params[name] + 0.3
    for b in range(2):
        ref_probs = ref.register_probabilities(ref.statevector(params, x[b].numpy()))
        assert np.abs(after[b].numpy() - ref_probs).max() < 1e-9


@pytest.mark.parametrize("trial", range(3))
def test_probability_gradient_matches_qiskit(trial):
    """u, v, wf: each entry is the angle of exactly one RY/RZ -> exact two-term parameter-shift.
    d, c: each entry scales a feature inside an angle -> central finite differences, h = 1e-5."""
    params = random_params(1.0)
    names = list(ref.PARAM_SIZES)
    features = rng.uniform(-1, 1, size=(3, FEATURES))
    w = rng.normal(size=2 ** ref.N_MEASURED)

    def loss_qiskit(theta):
        p = kit.unflatten(theta, ref.PARAM_SIZES)
        return sum(float(w @ ref.register_probabilities(ref.statevector(p, f))) for f in features)

    theta = kit.flatten(params, names)
    is_scale = np.concatenate([np.full(ref.PARAM_SIZES[k], k[0] in "dc") for k in names])
    h = 1e-5
    expected = np.array([
        (loss_qiskit(theta + h * e) - loss_qiskit(theta - h * e)) / (2 * h) if scale
        else (loss_qiskit(theta + np.pi / 2 * e) - loss_qiskit(theta - np.pi / 2 * e)) / 2
        for e, scale in zip(np.eye(len(theta)), is_scale)])
    model = make_model(params, torch.float64)
    (model.measured_probabilities(torch.tensor(features)) * torch.tensor(w)).sum().backward()
    ours = np.concatenate([np.atleast_1d(getattr(model, k).grad.numpy()) for k in names])
    assert np.abs(ours - expected)[~is_scale].max() < 1e-9
    assert np.abs(ours - expected)[is_scale].max() < 1e-6


def test_gray_action_code_relabels_the_outcomes():
    torch.manual_seed(3)
    binary = VQCHeadK(scale_init=0.7, action_code="binary")
    torch.manual_seed(3)
    gray = VQCHeadK(scale_init=0.7)
    for p, q in zip(binary.parameters(), gray.parameters()):
        assert torch.equal(p, q)
    x = torch.rand(16, FEATURES) * 2 - 1
    assert torch.equal(gray(x), binary(x)[:, ref.ACTION_CODES["gray"]])
    with pytest.raises(ValueError, match="action_code"):
        VQCHeadK(action_code="hex")


def test_removed_gates_have_no_effect():
    """The full description, with RZ(u3[11]) on q3 before the last CX chain and the rotation RY RZ RY of q3 after it,
    gives the same register probabilities as the reference without them, for any angles."""
    for _ in range(10):
        params = random_params(1.0)
        qc = ref.build_circuit(params, rng.uniform(-1, 1, size=FEATURES))
        cx_at = [k for k, inst in enumerate(qc.data) if inst.operation.name == "cx"]
        first, last = cx_at[12], cx_at[17]             # the last chain: CX 13 to 18
        a, b, c, e = rng.uniform(-np.pi, np.pi, size=4)
        full = qc.copy_empty_like()
        for k, inst in enumerate(qc.data):
            if k == first:
                full.rz(a, ref.q(3))
            full.append(inst)
            if k == last:
                full.ry(b, ref.q(3))
                full.rz(c, ref.q(3))
                full.ry(e, ref.q(3))
        assert len(cx_at) == 18 and len(full.data) == len(qc.data) + 4
        assert np.abs(ref.register_probabilities(Statevector(full).data)
                      - ref.register_probabilities(Statevector(qc).data)).max() < 1e-12


# --- guards: the equivalence tests above are vacuous if the output ignores inputs or parameters ---------

FEATURE_SLICES = {"M": slice(0, 4), "N": slice(4, 8), "G": slice(8, 12)}
EXPECTED_DEAD = {}


def test_outputs_are_not_degenerate():
    assert kit.output_std(lambda: VQCHeadK().double(), rng, FEATURES).min() > 1e-3


def test_every_feature_branch_matters():
    assert kit.blind_feature_slices(lambda: VQCHeadK().double(), rng, FEATURE_SLICES) == []


def test_dead_parameters_are_the_known_ones():
    assert kit.dead_parameter_entries(lambda: VQCHeadK().double(), rng, FEATURES) == EXPECTED_DEAD


@pytest.mark.parametrize("branch", ["M", "N", "G"])
def test_routing_of_each_branch(branch):
    """Feature i of a branch enters linearly, angle += weight * x_i, on qubit i only: N in all three data gates
    (weight d_r[3i + k]), M and G in the gate of block r that lists them (weight c_r[2i + j], j-th such gate:
    r = 1: -, M, G; r = 2: G, -, M; r = 3: M, G, -). Hence x_i dL/dx_i = sum of weight * dL/dweight over those
    gates; every term must be non-zero: the feature really reaches each of them."""
    second = {1: "-MG", 2: "G-M", 3: "MG-"}
    params = random_params(1.0)
    model = make_model(params, torch.float64)
    x = torch.tensor(rng.uniform(-1, 1, size=(1, FEATURES)), requires_grad=True)
    w = torch.tensor(rng.normal(size=8))
    (model.measured_probabilities(x) * w).sum().backward()
    offset = {"M": 0, "N": 4, "G": 8}[branch]
    for i in range(4):
        terms = []
        for r in ref.REPS:
            if branch == "N":
                gates = [(f"d{r}", 3 * i + k) for k in range(3)]
            else:
                gates = [(f"c{r}", 2 * i + second[r].replace("-", "").index(branch))]
            for name, idx in gates:
                p = getattr(model, name)
                terms.append(float(p.detach()[idx] * p.grad[idx]))
        assert len(terms) == (9 if branch == "N" else 3)
        assert abs(float(x.detach()[0, offset + i] * x.grad[0, offset + i]) - sum(terms)) < 1e-9
        assert all(abs(t) > 1e-8 for t in terms)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
