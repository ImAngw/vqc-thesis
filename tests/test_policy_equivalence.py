"""Tests of quantum_heads/analysis/policy_equivalence.py (A3 and the K=5 temperature control): distances, the 2x2
pair classes, the periods of J / Fourier angles, turning points, the temperature bisection. No checkpoints needed."""

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "quantum_heads" / "analysis"))
import policy_equivalence as pe  # noqa: E402
from classic_heads.fourier import FourierHead  # noqa: E402
from train import HEADS  # noqa: E402


def test_tv_kl_entropy_known_values():
    p = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    q = torch.tensor([[0.0, 1.0], [0.5, 0.5]])
    assert pe.tv(p, q) == pytest.approx(0.5)  # rows: 1 and 0
    assert pe.tv(p, p) == 0.0
    u, d = torch.tensor([[0.5, 0.5]]), torch.tensor([[0.9, 0.1]])
    assert pe.kl(d, u) == pytest.approx(0.9 * math.log(1.8) + 0.1 * math.log(0.2))
    assert pe.entropy(torch.full((3, 8), 1 / 8)) == pytest.approx(math.log(8))


def test_pair_classes_cover_the_2x2_table():
    c = pe.pair_classes(list(range(10)))
    assert [len(c[k]) for k in ("cross_same_pool", "within_J", "within_F", "cross_diff_pool")] == [10, 45, 45, 90]
    assert all(a[1] == b[1] for a, b in c["cross_same_pool"])
    assert all(a[1] != b[1] and a[0] != b[0] for a, b in c["cross_diff_pool"])
    assert all(a[0] == b[0] == "nav-id" and a[1] != b[1] for a, b in c["within_J"])


def test_pair_distance_uses_the_states_of_both_runs():
    a, b = ("nav-id", 0), ("four", 0)
    same, diff = torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]])
    # the runs agree on a's states and disagree on b's: distance = mean over both = 0.5
    probs = {(a, a): same, (b, a): same, (a, b): same, (b, b): diff}
    assert pe.pair_distances(probs, a, b)["tv"] == pytest.approx(0.5)


def test_summarize_head_effect():
    classes = {"within_J": [(1, 2)], "within_F": [(3, 4)], "cross_same_pool": [(1, 3)], "cross_diff_pool": [(1, 4)]}
    dist = {(1, 2): {"tv": 0.2}, (3, 4): {"tv": 0.4}, (1, 3): {"tv": 0.1}, (1, 4): {"tv": 0.5}}
    out = pe.summarize_classes(dist, classes)
    assert out["head_effect"]["seed_spread"] == pytest.approx(0.3)
    assert out["head_effect"]["tv"] == pytest.approx(0.2)


def test_periods_fourier_projection():
    head = FourierHead(n_freq=1, in_features=2)
    with torch.no_grad():
        head.freq.copy_(torch.tensor([[2.0, 0.0]]))
    x = torch.stack([torch.linspace(0, math.pi, 1001), torch.randn(1001)], dim=1)
    rho = pe.angle_range_periods(pe.head_angles(head, x))
    # angle 2 x0, x0 over [0, pi]: q1-q99 spans 0.98 * 2 pi -> rho 0.98
    assert rho.item() == pytest.approx(0.98, abs=0.01)


def test_periods_j_uses_the_head_data_angles():
    torch.manual_seed(0)
    head = HEADS["vqc-nav-id"]()
    feats = torch.randn(500, 12)
    angles = pe.head_angles(head, feats)
    assert angles.shape == (500, 36)
    assert torch.allclose(angles[:, 12:24], head.data_angles(feats, 2).flatten(1))
    # scale 0.05 at init: tiny ranges, far from the periodic regime
    assert pe.angle_range_periods(angles).max().item() < pe.PERIODIC_RHO


def test_turning_points():
    t = torch.linspace(0, 1, 64)
    curves = torch.stack([t, torch.cos(2 * math.pi * t), torch.cos(4 * math.pi * t), torch.zeros(64)])
    assert pe.turning_points(curves).tolist() == [0, 1, 3, 0]


def test_find_temperature_bisection():
    # entropy grows with T: H(T) = log 8 * T / (1 + T)
    h = lambda t: math.log(8) * t / (1 + t)  # noqa: E731
    t, ht = pe.find_temperature(h, target=1.0)
    assert ht == pytest.approx(1.0, abs=0.02)
    assert t == pytest.approx(1.0 / (math.log(8) - 1.0), rel=0.05)
    assert pe.find_temperature(h, target=0.01)[0] == 0.25  # below the bracket: clipped


def test_tempered_scales_logits():
    class Fixed(torch.nn.Module):
        def forward(self, obs):
            return torch.tensor([[2.0, 0.0]]), torch.zeros(1)
    logits, _ = pe.Tempered(Fixed(), 2.0)({})
    assert torch.allclose(logits, torch.tensor([[1.0, 0.0]]))


def test_nonmonotone_amplitude_and_significant_turning_points():
    t = torch.linspace(0, 1, 65)
    ramp = 0.5 * t                                        # monotone
    swing = 0.5 * torch.sin(math.pi * t)                  # up 0.5 then down 0.5
    wiggle = 0.3 + 0.001 * torch.sin(8 * math.pi * t)     # many tiny reversals
    curves = torch.stack([ramp, swing, wiggle])
    amp = pe.nonmonotone_amplitude(curves)
    assert amp[0].item() == pytest.approx(0.0, abs=1e-6)
    assert amp[1].item() == pytest.approx(1.0, abs=1e-3)  # |path| 1.0, net change 0
    assert amp[2].item() < 0.02
    assert pe.turning_points(curves, tol=1e-6).tolist()[2] >= 6  # the plain count sees the wiggle...
    assert pe.significant_turning_points(curves).tolist() == [0, 1, 0]  # ...the significant count does not
