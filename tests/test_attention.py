"""Tests for cs336_basics.attention. Torch reference ops are used only as oracles."""

import torch
import torch.nn.functional as F

from cs336_basics.attention import scaled_dot_product_attention


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
