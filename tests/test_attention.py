"""Tests for cs336_basics.attention. Torch reference ops are used only as oracles."""

import torch
import torch.nn.functional as F

import pytest

from cs336_basics.attention import (
    CausalMultiHeadSelfAttention,
    scaled_dot_product_attention,
)
from cs336_basics.layers import RotaryPositionalEmbedding


def test_sdpa_shape_and_dtype():
    Q = torch.randn(2, 5, 8)
    K = torch.randn(2, 7, 8)
    V = torch.randn(2, 7, 4)  # d_v differs from d_k
    out = scaled_dot_product_attention(Q, K, V)
    assert out.shape == (2, 5, 4)
    assert out.dtype == torch.float32


def test_sdpa_matches_torch_no_mask():
    Q = torch.randn(3, 6, 16)
    K = torch.randn(3, 6, 16)
    V = torch.randn(3, 6, 16)
    expected = F.scaled_dot_product_attention(Q, K, V)
    torch.testing.assert_close(
        scaled_dot_product_attention(Q, K, V), expected, atol=1e-5, rtol=1e-5
    )


def test_sdpa_matches_torch_causal_mask():
    Q = torch.randn(2, 4, 6, 8)  # (batch, heads, seq, d_k)
    K = torch.randn(2, 4, 6, 8)
    V = torch.randn(2, 4, 6, 8)
    mask = torch.tril(torch.ones(6, 6, dtype=torch.bool))
    # F.sdpa uses the same convention: bool True = attend.
    expected = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)
    torch.testing.assert_close(
        scaled_dot_product_attention(Q, K, V, mask), expected, atol=1e-5, rtol=1e-5
    )


def test_sdpa_matches_torch_random_mask_with_per_batch_mask():
    Q = torch.randn(3, 5, 8)
    K = torch.randn(3, 7, 8)
    V = torch.randn(3, 7, 8)
    mask = torch.rand(3, 5, 7) > 0.5
    mask[..., 0] = True  # guarantee every row has something to attend to
    expected = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)
    torch.testing.assert_close(
        scaled_dot_product_attention(Q, K, V, mask), expected, atol=1e-5, rtol=1e-5
    )


def test_sdpa_arbitrary_leading_dims():
    for lead in [(), (3,), (2, 3), (2, 3, 4)]:
        Q = torch.randn(*lead, 5, 8)
        K = torch.randn(*lead, 5, 8)
        V = torch.randn(*lead, 5, 8)
        assert scaled_dot_product_attention(Q, K, V).shape == (*lead, 5, 8)


def test_sdpa_masked_positions_have_no_influence():
    Q = torch.randn(1, 4, 8)
    K = torch.randn(1, 4, 8)
    V = torch.randn(1, 4, 8)
    mask = torch.tril(torch.ones(4, 4, dtype=torch.bool))
    out1 = scaled_dot_product_attention(Q, K, V, mask)

    # Change the last key and value. Query 0 may only see key 0, so it must not move.
    K2, V2 = K.clone(), V.clone()
    K2[:, -1] += 100
    V2[:, -1] += 100
    out2 = scaled_dot_product_attention(Q, K2, V2, mask)
    torch.testing.assert_close(out1[:, :3], out2[:, :3])


def test_sdpa_single_visible_key_returns_its_value():
    Q = torch.randn(1, 3, 8)
    K = torch.randn(1, 3, 8)
    V = torch.randn(1, 3, 8)
    mask = torch.eye(3, dtype=torch.bool)  # each query sees only its own key
    out = scaled_dot_product_attention(Q, K, V, mask)
    torch.testing.assert_close(out, V)


def test_sdpa_output_is_finite_with_large_scores():
    Q = torch.randn(1, 4, 8) * 100
    K = torch.randn(1, 4, 8) * 100
    V = torch.randn(1, 4, 8)
    assert torch.isfinite(scaled_dot_product_attention(Q, K, V)).all()


# ------------------------------------------------- CausalMultiHeadSelfAttention
def reference_mha(attn, x, rope=None, positions=None):
    """Oracle built from F.linear and F.scaled_dot_product_attention."""
    b, s, d = x.shape
    h, dk = attn.num_heads, attn.d_k

    def split(t):  # (b, s, d) -> (b, h, s, dk)
        return t.reshape(b, s, h, dk).transpose(1, 2)

    q = split(F.linear(x, attn.q_proj.weight))
    k = split(F.linear(x, attn.k_proj.weight))
    v = split(F.linear(x, attn.v_proj.weight))
    if rope is not None:
        pos = torch.arange(s) if positions is None else positions
        q, k = rope(q, pos), rope(k, pos)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out = out.transpose(1, 2).reshape(b, s, d)
    return F.linear(out, attn.output_proj.weight)


def test_mha_shape_and_dtype():
    attn = CausalMultiHeadSelfAttention(32, 4)
    x = torch.randn(2, 10, 32)
    y = attn(x)
    assert y.shape == (2, 10, 32)
    assert y.dtype == torch.float32


def test_mha_matches_reference_without_rope():
    attn = CausalMultiHeadSelfAttention(32, 4)
    x = torch.randn(2, 9, 32)
    torch.testing.assert_close(attn(x), reference_mha(attn, x), atol=1e-5, rtol=1e-5)


def test_mha_matches_reference_with_rope():
    rope = RotaryPositionalEmbedding(10000.0, 8, 64)  # d_k = 32 / 4 = 8
    attn = CausalMultiHeadSelfAttention(32, 4, rope=rope)
    x = torch.randn(2, 9, 32)
    expected = reference_mha(attn, x, rope=rope)
    torch.testing.assert_close(attn(x), expected, atol=1e-5, rtol=1e-5)


def test_mha_is_causal():
    rope = RotaryPositionalEmbedding(10000.0, 8, 64)
    attn = CausalMultiHeadSelfAttention(32, 4, rope=rope)
    x = torch.randn(1, 8, 32)
    y1 = attn(x)
    x2 = x.clone()
    x2[:, 5:] = torch.randn(1, 3, 32)  # change only the future
    y2 = attn(x2)
    torch.testing.assert_close(y1[:, :5], y2[:, :5])
    assert not torch.allclose(y1[:, 5:], y2[:, 5:])


def test_mha_rope_changes_output():
    rope = RotaryPositionalEmbedding(10000.0, 8, 64)
    with_rope = CausalMultiHeadSelfAttention(32, 4, rope=rope)
    without = CausalMultiHeadSelfAttention(32, 4)
    without.load_state_dict(with_rope.state_dict())
    x = torch.randn(1, 6, 32)
    assert not torch.allclose(with_rope(x), without(x), atol=1e-4)


def test_mha_rope_only_on_q_and_k_gives_shift_invariance():
    # q.k depends only on position differences, and v is not rotated, so
    # shifting every position by a constant must leave the output unchanged.
    # If RoPE were wrongly applied to v, this would fail.
    rope = RotaryPositionalEmbedding(10000.0, 8, 64)
    attn = CausalMultiHeadSelfAttention(32, 4, rope=rope)
    x = torch.randn(2, 6, 32)
    pos = torch.arange(6)
    y0 = attn(x, token_positions=pos)
    y1 = attn(x, token_positions=pos + 17)
    torch.testing.assert_close(y0, y1, atol=1e-4, rtol=1e-4)


def test_mha_accepts_batched_token_positions():
    rope = RotaryPositionalEmbedding(10000.0, 8, 64)
    attn = CausalMultiHeadSelfAttention(32, 4, rope=rope)
    x = torch.randn(2, 6, 32)
    pos = torch.arange(6).expand(2, 6)
    torch.testing.assert_close(attn(x, token_positions=pos), attn(x), atol=1e-5, rtol=1e-5)


def test_mha_parameter_names_and_shapes():
    attn = CausalMultiHeadSelfAttention(32, 4)
    shapes = {n: tuple(p.shape) for n, p in attn.named_parameters()}
    assert shapes == {
        "q_proj.weight": (32, 32),
        "k_proj.weight": (32, 32),
        "v_proj.weight": (32, 32),
        "output_proj.weight": (32, 32),
    }


def test_mha_rejects_indivisible_heads():
    with pytest.raises(ValueError):
        CausalMultiHeadSelfAttention(30, 4)


def test_mha_gradients_flow():
    attn = CausalMultiHeadSelfAttention(16, 2)
    attn(torch.randn(2, 5, 16)).sum().backward()
    for p in attn.parameters():
        assert p.grad is not None
