"""
VQCHeadI ("vqc-optim", vqc_head_i.py) against the Qiskit reference (vqc_ref_i.py), written from the description.

  1. clean path vs Qiskit: statevector and logits on N_TRIALS random parameter sets and inputs, both precisions
  2. grad and no-grad paths agree; the block cache follows in-place updates
  3. gradient of the register probabilities w.r.t. every parameter vs Qiskit: exact parameter-shift for the free
     angles (u, v, wf), central finite differences for the feature weights s (angle = s * x)
  4. guards: outputs vary, every input branch matters (in every block), dead parameters are the known ones
No noisy path: density_matrix is not implemented for this ansatz.

Run:  .venv/bin/python quantum_heads/test_vqc_head_i.py
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
import vqc_ref_i as ref
from vqc_head_i import VQCHeadI

N_TRIALS = 50
TOL = kit.TOL
FEATURES = 12
rng = np.random.default_rng(0)


@pytest.fixture(params=[torch.float32, torch.float64], ids=["f32", "f64"])
def dtype(request):
    return request.param


def random_params(scale: float) -> dict:
    return kit.random_params(rng, ref.PARAM_SIZES, scale)


def make_model(params: dict, dtype) -> VQCHeadI:
    return kit.make_model(VQCHeadI, params, dtype)


def test_parameter_count():
    model = VQCHeadI()
    assert sum(p.numel() for p in model.parameters()) == 93
    assert {k: p.numel() for k, p in model.named_parameters()} == ref.PARAM_SIZES


def test_feature_weights_start_at_one():
    model = VQCHeadI()
    for r in ref.REPS:
        assert torch.equal(getattr(model, f"s{r}"), torch.ones(12))
    model = VQCHeadI(scale_init=np.pi / 2)
    for r in ref.REPS:
        assert torch.allclose(getattr(model, f"s{r}"), torch.full((12,), np.pi / 2))


def test_default_init_is_unchanged():
    # angle_std = 0.5 and wf_offset = 0 draw exactly what the original randn * 0.5 init drew
    torch.manual_seed(0)
    model = VQCHeadI()
    torch.manual_seed(0)
    expected = {}
    for r in ref.REPS:
        expected[f"u{r}"], expected[f"v{r}"] = torch.randn(9) * 0.5, torch.randn(9) * 0.5
    expected["wf"] = torch.randn(3) * 0.5
    for name, value in expected.items():
        assert torch.equal(getattr(model, name), value), name


@pytest.mark.parametrize("seed", range(5))
def test_near_identity_init_starts_from_a_near_uniform_policy(seed):
    # ln 8 = 2.079; the default init gives ~1.6-1.7 on the same features
    x = torch.as_tensor(rng.uniform(-1, 1, (64, FEATURES)), dtype=torch.float32)
    torch.manual_seed(seed)
    probs = VQCHeadI(scale_init=0.1, angle_std=0.05, wf_offset=np.pi / 2)(x).softmax(-1)
    assert (-(probs * probs.log()).sum(-1)).mean() > 2.0


def test_clean_path_matches_qiskit(dtype):
    for _ in range(N_TRIALS):
        params = random_params(rng.choice([0.5, 1.5]))
        features = rng.uniform(-1, 1, size=(2, FEATURES))
        model = make_model(params, dtype)
        x = torch.tensor(features, dtype=dtype)
        with torch.no_grad():
            state, probs, logits = model.state(x), model.measured_probabilities(x), model(x)
        for b in range(2):
            ref_state = ref.statevector(params, features[b])
            ref_probs = ref.register_probabilities(ref_state)
            assert np.abs(state[b].numpy() - ref_state).max() < TOL[dtype]
            assert np.abs(probs[b].numpy() - ref_probs).max() < TOL[dtype]
            # logits are log P: weight the error by P, where P is not negligible
            err = np.abs(logits[b].numpy() - ref.logits(params, features[b])) * ref_probs
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


@pytest.mark.parametrize("name", ["v1", "v2", "v3", "wf", "s3", "u2"])
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
    s: each entry scales a feature (angle = s * x) -> central finite differences, h = 1e-5."""
    params = random_params(1.0)
    names = list(ref.PARAM_SIZES)
    features = rng.uniform(-1, 1, size=(3, FEATURES))
    w = rng.normal(size=2 ** ref.N_MEASURED)

    def loss_qiskit(theta):
        p = kit.unflatten(theta, ref.PARAM_SIZES)
        return sum(float(w @ ref.register_probabilities(ref.statevector(p, f))) for f in features)

    theta = kit.flatten(params, names)
    is_scale = np.concatenate([np.full(ref.PARAM_SIZES[k], k.startswith("s")) for k in names])
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


# --- guards: the equivalence tests above are vacuous if the output ignores inputs or parameters ---------

FEATURE_SLICES = {"M": slice(0, 4), "N": slice(4, 8), "G": slice(8, 12)}
EXPECTED_DEAD = {}


def test_outputs_are_not_degenerate():
    assert kit.output_std(lambda: VQCHeadI().double(), rng, FEATURES).min() > 1e-3


def test_every_feature_branch_matters():
    assert kit.blind_feature_slices(lambda: VQCHeadI().double(), rng, FEATURE_SLICES) == []


def test_dead_parameters_are_the_known_ones():
    assert kit.dead_parameter_entries(lambda: VQCHeadI().double(), rng, FEATURES) == EXPECTED_DEAD


@pytest.mark.parametrize("branch", range(3))
def test_every_qubit_reads_every_branch(branch):
    """Cyclic routing: feature k of a branch enters only as s_r[4i + k] * x[k], once per block, on qubit
    i = (branch - r + 1) mod 3. Hence x[k] dL/dx[k] = sum_r s_r[4i+k] dL/ds_r[4i+k] per sample, and every block
    contributes: each qubit sees each branch."""
    params = random_params(1.0)
    model = make_model(params, torch.float64)
    x = torch.tensor(rng.uniform(-1, 1, size=(1, FEATURES)), requires_grad=True)
    w = torch.tensor(rng.normal(size=8))
    (model.measured_probabilities(x) * w).sum().backward()
    for k in range(4):
        col = 4 * branch + k
        terms = []
        for r in ref.REPS:
            i = (branch - r + 1) % 3
            s = getattr(model, f"s{r}")
            terms.append(float(s.detach()[4 * i + k] * s.grad[4 * i + k]))
        assert abs(float(x.detach()[0, col] * x.grad[0, col]) - sum(terms)) < 1e-9
        assert all(abs(t) > 1e-8 for t in terms)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


def test_gray_action_code_relabels_the_outcomes():
    """action_code="gray": logit[a] = log P(a ^ (a >> 1)); same parameters and RNG use as the binary head."""
    torch.manual_seed(3)
    binary = VQCHeadI(scale_init=0.7)
    torch.manual_seed(3)
    gray = VQCHeadI(scale_init=0.7, action_code="gray")
    for p, q in zip(binary.parameters(), gray.parameters()):
        assert torch.equal(p, q)
    x = torch.rand(16, FEATURES) * 2 - 1
    code = [a ^ (a >> 1) for a in range(8)]
    assert torch.equal(gray(x), binary(x)[:, code])
    assert torch.allclose(gray(x).exp().sum(1), torch.ones(16, dtype=x.dtype), atol=1e-5)  # still a distribution
    assert all(bin(code[a] ^ code[(a + 1) % 8]).count("1") == 1 for a in range(8))  # compass neighbours: one bit
    with pytest.raises(ValueError, match="action_code"):
        VQCHeadI(action_code="hex")
