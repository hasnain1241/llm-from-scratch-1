"""Tests for cs336_basics.layers. Torch reference ops are used only as oracles."""

import math

import pytest
import torch
import torch.nn.functional as F

from cs336_basics.layers import Linear


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
