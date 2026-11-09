"""Shapes and parameter counts of the encoder, classic heads and policy (QML_Thesis architecture)."""

import pytest
import torch

from classic_heads import MLPHead, SharedHead
from encoder import Encoder
from rl_train.policy import Policy


def _n_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def _obs(b: int = 4, k: int = 10, window: int = 3) -> dict[str, torch.Tensor]:
    return {"perception": torch.rand(b, 1, window, window), "navigator": torch.rand(b, 17), "memory": torch.rand(b, k, 10)}


def test_encoder_output_is_12_features_for_any_memory_length():
    enc = Encoder()
    assert enc.out_features == 12
    for k in (1, 10):
        o = _obs(k=k)
        assert enc(o["perception"], o["navigator"], o["memory"]).shape == (4, 12)


def test_encoder_memory_is_a_gru_and_features_are_tanh_bounded():
    enc = Encoder()
    assert isinstance(enc.memory, torch.nn.GRU) and enc.memory.input_size == 10 and enc.memory.hidden_size == 4
    o = _obs()
    assert enc(o["perception"], o["navigator"], o["memory"]).abs().max() <= 1.0


def test_encoder_rmsnorm_variants():
    plain, learned, fixed = Encoder(norm=False), Encoder(), Encoder(norm_affine=False)
    # norm=False keeps the pre-RMSNorm state dict, so older checkpoints still load
    assert set(fixed.state_dict()) == set(plain.state_dict())
    assert _n_params(learned) - _n_params(plain) == 12  # one gain per feature, no bias
    assert sum(isinstance(m, torch.nn.RMSNorm) for m in learned.modules()) == 3
    o = _obs()
    out = fixed(o["perception"], o["navigator"], o["memory"])
    assert out.shape == (4, 12) and out.abs().max() <= 1.0


def test_encoder_perception_is_linear_on_the_3x3_window_by_default():
    enc = Encoder()
    linears = [m for m in enc.perception.modules() if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 1 and (linears[0].in_features, linears[0].out_features) == (9, 4)
    assert not any(isinstance(m, torch.nn.Conv2d) for m in enc.perception.modules())
    assert _n_params(enc.perception) == 9 * 4 + 4 + 4  # Linear 9 -> 4 + RMSNorm gains


def test_encoder_cnn_variant_loads_5x5_checkpoints():
    # the pre-v10 encoder: XS CNN on the 5x5 window, same state dict keys and shapes
    old = Encoder(perception="cnn", window=5)
    assert any(isinstance(m, torch.nn.Conv2d) for m in old.perception.modules())
    assert set(old.state_dict()) >= {"perception.0.weight", "perception.2.weight", "perception.5.weight"}
    assert old.perception[5].in_features == 2 * 3 * 3
    Encoder(perception="cnn", window=5).load_state_dict(old.state_dict())
    o = _obs(window=5)
    assert old(o["perception"], o["navigator"], o["memory"]).shape == (4, 12)
    with pytest.raises(ValueError):
        Encoder(perception="mlp")


def test_head_parameter_counts():
    assert _n_params(SharedHead()) == 96
    assert _n_params(MLPHead(hidden=16)) == 344


def test_policy_outputs_and_critic_input():
    pol = Policy(SharedHead())
    logits, value = pol(_obs())
    assert logits.shape == (4, 8) and value.shape == (4,)
    assert pol.critic[0].in_features == 12


def test_fourier_head_matches_its_formula_and_trains():
    from classic_heads import FourierHead

    torch.manual_seed(0)
    head = FourierHead(n_freq=4)
    assert _n_params(head) == 120  # 4 x 12 frequencies + Linear(8 -> 8)
    x = torch.rand(5, 12) * 2 - 1
    logits = head(x)
    assert logits.shape == (5, 8)
    z = x @ head.freq.T
    assert torch.allclose(logits, head.out(torch.cat([torch.sin(z), torch.cos(z)], dim=-1)))
    logits.sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())
    torch.manual_seed(0)
    again = FourierHead(n_freq=4)
    assert all(torch.equal(p, q) for p, q in zip(head.parameters(), again.parameters()))
