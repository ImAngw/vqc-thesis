"""
VQCHeadJ ("vqc-nav-id", vqc_head_j.py) against the Qiskit reference (vqc_ref_j.py), written from the description.

  1. clean path vs Qiskit: statevector and logits (binary and Gray action codes) on N_TRIALS random parameter sets
     and inputs, both precisions
  2. grad and no-grad paths agree; the block cache follows in-place updates
  3. gradient of the register probabilities w.r.t. every parameter vs Qiskit: exact parameter-shift for the free
     angles (u, v, wf), central finite differences for the feature weights w and c (angle = w * N + c * M or G)
  4. noisy path: density matrix vs Aer, noise off equals clean (both action codes), readout wiring, gradient flows,
     sampled probabilities with depolarizing + readout error vs Aer shots
  5. guards: outputs vary, every input branch matters, no dead parameter, the navigator reaches every qubit of every
     block and M / G only their B / C qubit

Run:  .venv/bin/python -m pytest quantum_heads/test_vqc_head_j.py -q
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import vqc_testkit as kit
from vqc_core import noise as nz
import vqc_ref_j as ref
from vqc_head_j import VQCHeadJ

N_TRIALS = 50
TOL = kit.TOL
FEATURES = 12
rng = np.random.default_rng(0)


@pytest.fixture(params=[torch.float32, torch.float64], ids=["f32", "f64"])
def dtype(request):
    return request.param


def random_params(scale: float) -> dict:
    return kit.random_params(rng, ref.PARAM_SIZES, scale)


def make_model(params: dict, dtype, **ctor) -> VQCHeadJ:
    return kit.make_model(VQCHeadJ, params, dtype, **ctor)


def test_parameter_count():
    model = VQCHeadJ()
    assert sum(p.numel() for p in model.parameters()) == 117
    assert {k: p.numel() for k, p in model.named_parameters()} == ref.PARAM_SIZES


def test_init():
    model = VQCHeadJ(scale_init=0.1)
    for r in ref.REPS:
        assert torch.allclose(getattr(model, f"w{r}"), torch.full((12,), 0.1))
        assert torch.allclose(getattr(model, f"c{r}"), torch.full((8,), 0.1))
    # u, v, wf draw what VQCHeadI draws (the weights are filled, not drawn)
    torch.manual_seed(0)
    model = VQCHeadJ(angle_std=0.05, wf_offset=np.pi / 2)
    torch.manual_seed(0)
    for r in ref.REPS:
        assert torch.equal(getattr(model, f"u{r}"), torch.randn(9) * 0.05)
        assert torch.equal(getattr(model, f"v{r}"), torch.randn(9) * 0.05)
    assert torch.equal(model.wf, torch.randn(3) * 0.05 + np.pi / 2)


@pytest.mark.parametrize("seed", range(5))
def test_near_identity_init_starts_from_a_near_uniform_policy(seed):
    x = torch.as_tensor(rng.uniform(-1, 1, (64, FEATURES)), dtype=torch.float32)
    torch.manual_seed(seed)
    # the registered init ("vqc-nav-id"): scale 0.05, half of VQCHeadI's 0.1, since the navigator enters 9 times
    probs = VQCHeadJ(scale_init=0.05, angle_std=0.05, wf_offset=np.pi / 2)(x).softmax(-1)
    assert (-(probs * probs.log()).sum(-1)).mean() > 2.0  # ln 8 = 2.079


@pytest.mark.parametrize("code", ["binary", "gray"])
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


@pytest.mark.parametrize("name", ["v1", "v2", "v3", "wf", "w3", "c2", "u2"])
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
    w, c: each entry scales a feature inside an angle -> central finite differences, h = 1e-5."""
    params = random_params(1.0)
    names = list(ref.PARAM_SIZES)
    features = rng.uniform(-1, 1, size=(3, FEATURES))
    w = rng.normal(size=2 ** ref.N_MEASURED)

    def loss_qiskit(theta):
        p = kit.unflatten(theta, ref.PARAM_SIZES)
        return sum(float(w @ ref.register_probabilities(ref.statevector(p, f))) for f in features)

    theta = kit.flatten(params, names)
    is_scale = np.concatenate([np.full(ref.PARAM_SIZES[k], k[0] in "wc" and k != "wf") for k in names])
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
    binary = VQCHeadJ(scale_init=0.7)
    torch.manual_seed(3)
    gray = VQCHeadJ(scale_init=0.7, action_code="gray")
    for p, q in zip(binary.parameters(), gray.parameters()):
        assert torch.equal(p, q)
    x = torch.rand(16, FEATURES) * 2 - 1
    assert torch.equal(gray(x), binary(x)[:, ref.ACTION_CODES["gray"]])
    with pytest.raises(ValueError, match="action_code"):
        VQCHeadJ(action_code="hex")


# --- noise ------------------------------------------------------------------------------------------

def noise_model(p1, p2, p01=None, p10=None):
    readout = None if p01 is None else [(ref.q(i), p01[i], p10[i]) for i in range(ref.N_MEASURED)]
    return kit.noise_model(["ry", "rz"], ["cx"], p1, p2, readout)


@pytest.mark.parametrize("p1,p2", [(0.0, 0.0), (0.01, 0.0), (0.0, 0.03), (0.02, 0.05), (0.2, 0.4)])
def test_density_matrix_matches_aer(p1, p2):
    params = random_params(1.0)
    features = rng.uniform(-1, 1, size=(2, FEATURES))
    model = make_model(params, torch.float64)
    with torch.no_grad():
        rho = model.density_matrix(torch.tensor(features), nz.NoiseParams(p1=p1, p2=p2))
    nm = noise_model(p1, p2)
    for b in range(2):
        qc = ref.build_circuit(params, features[b], labelled=True)
        assert np.abs(rho[b].numpy() - kit.aer_density_matrix(qc, nm)).max() < 1e-10


@pytest.mark.parametrize("code", ["binary", "gray"])
def test_noise_off_equals_clean_path(code):
    model = make_model(random_params(1.0), torch.float64, action_code=code)
    x = torch.tensor(rng.uniform(-1, 1, size=(3, FEATURES)))
    with torch.no_grad():
        assert torch.allclose(model(x), model(x, noise=nz.NoiseParams()), atol=1e-10)


def test_readout_error_wiring_is_exact():
    """All three qubits are measured; with p01 = 1, p10 = 1 every bit of the register flips."""
    model = make_model(random_params(1.0), torch.float64)
    x = torch.tensor(rng.uniform(-1, 1, size=(2, FEATURES)))
    with torch.no_grad():
        clean = model.measured_probabilities(x)
        flipped = model.measured_probabilities(x, nz.NoiseParams(p01=1.0, p10=1.0))
    assert torch.allclose(flipped, clean.flip(-1), atol=1e-10)


def test_noisy_gradient_flows_to_parameters():
    model = make_model(random_params(1.0), torch.float64)
    x = torch.tensor(rng.uniform(-1, 1, size=(2, FEATURES)))
    model(x, noise=nz.NoiseParams(p1=0.02, p2=0.05, p01=0.03, p10=0.05)).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(p.grad.abs().max() > 0 for p in model.parameters())


def test_noisy_probabilities_with_readout_match_aer_sampling():
    """Depolarizing + readout end to end: probabilities within 5 standard errors of 400k seeded shots."""
    params = random_params(1.0)
    features = rng.uniform(-1, 1, size=(1, FEATURES))
    p1, p2, m = 0.02, 0.05, ref.N_MEASURED
    p01, p10 = rng.uniform(0.0, 0.1, size=m), rng.uniform(0.0, 0.1, size=m)
    model = make_model(params, torch.float64)
    noise = nz.NoiseParams(p1=p1, p2=p2, p01=torch.tensor(p01), p10=torch.tensor(p10))
    with torch.no_grad():
        ours = model.measured_probabilities(torch.tensor(features), noise)[0].numpy()
    qc = ref.build_circuit(params, features[0], labelled=True)
    freq = kit.aer_register_frequencies(qc, noise_model(p1, p2, p01, p10), [ref.q(i) for i in range(m)], seed=7)
    kit.assert_frequencies_match(freq, ours)


# --- guards: the equivalence tests above are vacuous if the output ignores inputs or parameters ---------

FEATURE_SLICES = {"M": slice(0, 4), "N": slice(4, 8), "G": slice(8, 12)}
EXPECTED_DEAD = {}


def test_outputs_are_not_degenerate():
    assert kit.output_std(lambda: VQCHeadJ().double(), rng, FEATURES).min() > 1e-3


def test_every_feature_branch_matters():
    assert kit.blind_feature_slices(lambda: VQCHeadJ().double(), rng, FEATURE_SLICES) == []


def test_dead_parameters_are_the_known_ones():
    assert kit.dead_parameter_entries(lambda: VQCHeadJ().double(), rng, FEATURES) == EXPECTED_DEAD


@pytest.mark.parametrize("branch", ["M", "N", "G"])
def test_routing_of_each_branch(branch):
    """Feature j of a branch enters linearly, angle += weight * x[j], at gate k = (j - i) mod 4 of qubit i. Hence
    x[j] dL/dx[j] = sum over its gates of weight * dL/dweight per sample. N: every qubit of every block (9 gates,
    weight w_r[4i + k]); M: only the B qubit i = (2 - r) mod 3 (weight c_r[k]); G: only the C qubit i = (3 - r) mod 3
    (weight c_r[4 + k]). Every term must be non-zero: the branch really reaches those qubits."""
    params = random_params(1.0)
    model = make_model(params, torch.float64)
    x = torch.tensor(rng.uniform(-1, 1, size=(1, FEATURES)), requires_grad=True)
    w = torch.tensor(rng.normal(size=8))
    (model.measured_probabilities(x) * w).sum().backward()
    offset = {"M": 0, "N": 4, "G": 8}[branch]
    for j in range(4):
        terms = []
        for r in ref.REPS:
            if branch == "N":
                gates = [(f"w{r}", 4 * i + (j - i) % 4) for i in range(3)]
            elif branch == "M":
                gates = [(f"c{r}", (j - (2 - r) % 3) % 4)]
            else:
                gates = [(f"c{r}", 4 + (j - (3 - r) % 3) % 4)]
            for name, idx in gates:
                p = getattr(model, name)
                terms.append(float(p.detach()[idx] * p.grad[idx]))
        assert len(terms) == (9 if branch == "N" else 3)
        assert abs(float(x.detach()[0, offset + j] * x.grad[0, offset + j]) - sum(terms)) < 1e-9
        assert all(abs(t) > 1e-8 for t in terms)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
