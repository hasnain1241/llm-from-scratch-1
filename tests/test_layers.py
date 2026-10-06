"""Tests for cs336_basics.layers. Torch reference ops are used only as oracles."""

import math

import pytest
import torch
import torch.nn.functional as F

from cs336_basics.layers import Embedding, Linear, RMSNorm


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
