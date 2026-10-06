"""Tests for cs336_basics.layers. Torch reference ops are used only as oracles."""

import math

import pytest
import torch
import torch.nn.functional as F

from cs336_basics.layers import Embedding, Linear


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
