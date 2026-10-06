"""Tests for cs336_basics.layers. Torch reference ops are used only as oracles."""

import math

import pytest
import torch
import torch.nn.functional as F

from cs336_basics.layers import Embedding, Linear, RMSNorm, SwiGLU, default_d_ff, silu, softmax


# ---------------------------------------------------------------- Linear
def test_linear_shape_and_dtype():
    layer = Linear(8, 16)
    x = torch.randn(2, 5, 8)
    y = layer(x)
    assert y.shape == (2, 5, 16)
    assert y.dtype == torch.float32


def test_linear_weight_shape_and_no_bias():
    layer = Linear(8, 16)
    assert layer.weight.shape == (16, 8)
    names = [n for n, _ in layer.named_parameters()]
    assert names == ["weight"]


def test_linear_matches_functional():
    layer = Linear(8, 16)
    x = torch.randn(3, 4, 8)
    expected = F.linear(x, layer.weight)  # oracle, no bias
    torch.testing.assert_close(layer(x), expected)


def test_linear_arbitrary_batch_dims():
    layer = Linear(4, 6)
    for shape in [(4,), (3, 4), (2, 3, 4), (2, 3, 5, 4)]:
        x = torch.randn(*shape)
        assert layer(x).shape == shape[:-1] + (6,)


def test_linear_init_is_truncated_normal():
    in_f, out_f = 256, 512
    layer = Linear(in_f, out_f)
    std = math.sqrt(2.0 / (in_f + out_f))
    w = layer.weight.detach()
    assert w.abs().max() <= 3 * std + 1e-6
    # Truncation at 3 std shrinks the std only slightly (about 1.3 percent).
    assert w.std().item() == pytest.approx(std, rel=0.05)
    assert w.mean().abs().item() < 0.1 * std


def test_linear_dtype_argument():
    layer = Linear(4, 4, dtype=torch.float64)
    assert layer.weight.dtype == torch.float64


def test_linear_gradients_flow():
    layer = Linear(4, 3)
    layer(torch.randn(2, 4)).sum().backward()
    assert layer.weight.grad is not None
    assert layer.weight.grad.shape == layer.weight.shape


# ------------------------------------------------------------- Embedding
def test_embedding_shape_and_dtype():
    emb = Embedding(100, 16)
    ids = torch.randint(0, 100, (2, 7))
    out = emb(ids)
    assert out.shape == (2, 7, 16)
    assert out.dtype == torch.float32


def test_embedding_matches_functional():
    emb = Embedding(50, 8)
    ids = torch.randint(0, 50, (3, 5))
    expected = F.embedding(ids, emb.weight)  # oracle
    torch.testing.assert_close(emb(ids), expected)


def test_embedding_returns_the_right_row():
    emb = Embedding(10, 4)
    out = emb(torch.tensor([3]))
    torch.testing.assert_close(out[0], emb.weight[3])


def test_embedding_arbitrary_batch_dims():
    emb = Embedding(20, 6)
    for shape in [(5,), (2, 5), (2, 3, 5)]:
        ids = torch.randint(0, 20, shape)
        assert emb(ids).shape == shape + (6,)


def test_embedding_init_is_truncated_normal():
    emb = Embedding(2000, 64)
    w = emb.weight.detach()
    assert w.abs().max() <= 3.0 + 1e-6
    # Truncation at 3 std shrinks the std to about 0.987.
    assert w.std().item() == pytest.approx(1.0, rel=0.05)
    assert w.mean().abs().item() < 0.05


def test_embedding_gradient_only_touches_used_rows():
    emb = Embedding(10, 4)
    emb(torch.tensor([2, 2, 5])).sum().backward()
    grad = emb.weight.grad
    assert grad[2].abs().sum() > 0
    assert grad[5].abs().sum() > 0
    unused = [i for i in range(10) if i not in (2, 5)]
    assert grad[unused].abs().sum() == 0


# --------------------------------------------------------------- RMSNorm
def reference_rms_norm(x, gain, eps):
    """Plain float32 formula, used as the oracle."""
    x32 = x.float()
    rms = torch.sqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return (x32 / rms) * gain.float()


def test_rmsnorm_shape_and_dtype():
    norm = RMSNorm(16)
    x = torch.randn(2, 5, 16)
    y = norm(x)
    assert y.shape == x.shape
    assert y.dtype == torch.float32


def test_rmsnorm_matches_reference():
    norm = RMSNorm(32, eps=1e-5)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(32))  # non-trivial gain
    x = torch.randn(4, 7, 32) * 10
    torch.testing.assert_close(norm(x), reference_rms_norm(x, norm.weight, 1e-5))


def test_rmsnorm_matches_torch_rms_norm_if_available():
    if not hasattr(F, "rms_norm"):
        pytest.skip("F.rms_norm needs a newer torch")
    norm = RMSNorm(32, eps=1e-5)
    x = torch.randn(3, 32)
    expected = F.rms_norm(x, (32,), weight=norm.weight, eps=1e-5)
    torch.testing.assert_close(norm(x), expected)


def test_rmsnorm_output_has_unit_rms_with_unit_gain():
    norm = RMSNorm(64, eps=1e-8)
    y = norm(torch.randn(10, 64) * 5)
    rms = y.pow(2).mean(-1).sqrt()
    torch.testing.assert_close(rms, torch.ones(10), atol=1e-4, rtol=1e-4)


def test_rmsnorm_is_scale_invariant():
    norm = RMSNorm(16, eps=1e-8)
    x = torch.randn(3, 16)
    torch.testing.assert_close(norm(x), norm(x * 100), atol=1e-4, rtol=1e-4)


def test_rmsnorm_preserves_low_precision_dtype():
    norm = RMSNorm(16)
    x = torch.randn(2, 16).to(torch.bfloat16)
    y = norm(x)
    assert y.dtype == torch.bfloat16
    expected = reference_rms_norm(x, norm.weight, 1e-5).to(torch.bfloat16)
    torch.testing.assert_close(y, expected)


def test_rmsnorm_fp16_large_values_do_not_overflow():
    # 300^2 = 90000 > fp16 max (65504). Upcasting avoids inf.
    norm = RMSNorm(8)
    x = torch.full((1, 8), 300.0, dtype=torch.float16)
    y = norm(x)
    assert torch.isfinite(y).all()


def test_rmsnorm_gain_is_learnable_and_initialized_to_one():
    norm = RMSNorm(8)
    assert norm.weight.shape == (8,)
    assert torch.all(norm.weight == 1)
    norm(torch.randn(2, 8)).sum().backward()
    assert norm.weight.grad is not None


# --------------------------------------------------------------- softmax
def test_softmax_matches_torch():
    x = torch.randn(4, 5, 10)
    for dim in (0, 1, 2, -1):
        torch.testing.assert_close(softmax(x, dim=dim), torch.softmax(x, dim=dim))


def test_softmax_shape_dtype_and_sums_to_one():
    x = torch.randn(3, 8)
    y = softmax(x, dim=-1)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
    torch.testing.assert_close(y.sum(-1), torch.ones(3))
    assert (y >= 0).all()


def test_softmax_large_values_are_stable():
    x = torch.tensor([[1000.0, 1001.0, 1002.0], [-1000.0, -1001.0, -1002.0]])
    y = softmax(x, dim=-1)
    assert torch.isfinite(y).all()
    torch.testing.assert_close(y, torch.softmax(x, dim=-1))


def test_softmax_is_shift_invariant():
    x = torch.randn(2, 6)
    torch.testing.assert_close(softmax(x), softmax(x + 123.0))


def test_softmax_handles_neg_inf_mask_entries():
    # Masked attention scores use -inf. Those entries must get probability 0.
    x = torch.tensor([[1.0, 2.0, float("-inf")]])
    y = softmax(x, dim=-1)
    assert y[0, 2] == 0
    torch.testing.assert_close(y, torch.softmax(x, dim=-1))


def test_softmax_gradient_matches_torch():
    x = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    x2 = x.detach().clone().requires_grad_(True)
    w = torch.randn(3, 5, dtype=torch.float64)
    (softmax(x) * w).sum().backward()
    (torch.softmax(x2, dim=-1) * w).sum().backward()
    torch.testing.assert_close(x.grad, x2.grad)


# ------------------------------------------------------------------ SiLU
def test_silu_matches_torch():
    x = torch.randn(5, 7) * 5
    torch.testing.assert_close(silu(x), F.silu(x))


def test_silu_known_values():
    assert silu(torch.tensor(0.0)).item() == 0.0
    # Large positive: close to identity. Large negative: close to 0.
    assert silu(torch.tensor(20.0)).item() == pytest.approx(20.0, abs=1e-4)
    assert abs(silu(torch.tensor(-20.0)).item()) < 1e-6


# --------------------------------------------------------------- SwiGLU
def test_default_d_ff_is_multiple_of_64_and_near_8_3():
    for d_model in (64, 128, 256, 512, 768, 1024):
        d_ff = default_d_ff(d_model)
        assert d_ff % 64 == 0
        assert abs(d_ff - 8 * d_model / 3) <= 32
    assert default_d_ff(768) == 2048
    assert default_d_ff(512) == 1344
    assert default_d_ff(64) == 192


def test_swiglu_shape_and_dtype():
    ffn = SwiGLU(32, 96)
    x = torch.randn(2, 5, 32)
    y = ffn(x)
    assert y.shape == (2, 5, 32)
    assert y.dtype == torch.float32


def test_swiglu_uses_default_d_ff_when_none():
    ffn = SwiGLU(64)
    assert ffn.d_ff == default_d_ff(64)


def test_swiglu_weight_shapes():
    ffn = SwiGLU(32, 96)
    assert ffn.w1.weight.shape == (96, 32)
    assert ffn.w3.weight.shape == (96, 32)
    assert ffn.w2.weight.shape == (32, 96)


def test_swiglu_matches_reference():
    ffn = SwiGLU(32, 96)
    x = torch.randn(3, 4, 32)
    gate = F.silu(F.linear(x, ffn.w1.weight))
    value = F.linear(x, ffn.w3.weight)
    expected = F.linear(gate * value, ffn.w2.weight)
    torch.testing.assert_close(ffn(x), expected)


def test_swiglu_zero_input_gives_zero_output():
    ffn = SwiGLU(16, 64)
    y = ffn(torch.zeros(2, 16))
    assert torch.all(y == 0)


def test_swiglu_gradients_reach_all_three_matrices():
    ffn = SwiGLU(16, 64)
    ffn(torch.randn(2, 16)).sum().backward()
    for name in ("w1", "w2", "w3"):
        assert getattr(ffn, name).weight.grad is not None
