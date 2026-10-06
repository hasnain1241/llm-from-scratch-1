"""Attention: scaled dot-product attention and causal multi-head self-attention."""

import math

import torch
from einops import einsum, rearrange
from torch import nn

from cs336_basics.layers import Linear, softmax


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Scaled dot-product attention.

    Concept:
        Each query looks at every key, turns the similarity scores into weights
        with softmax, and returns the weighted average of the values.

    Math:
        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

        Dividing by sqrt(d_k) keeps the score variance near 1. Without it, large
        d_k makes scores large, softmax saturates, and gradients vanish.

    Shapes (any number of leading batch dims, written "..."):
        Q:    (..., queries, d_k)
        K:    (..., keys, d_k)
        V:    (..., keys, d_v)
        mask: (..., queries, keys) bool, True = may attend, False = blocked.
              Must broadcast against the score shape. Optional.
        out:  (..., queries, d_v)

    Every query row must have at least one True in the mask, otherwise the
    softmax row is all -inf and the output is NaN.
    """
    d_k = Q.shape[-1]

    # scores[..., i, j] = q_i . k_j / sqrt(d_k)       (..., queries, keys)
    scores = einsum(Q, K, "... q d, ... k d -> ... q k") / math.sqrt(d_k)

    if mask is not None:
        # Blocked positions get -inf, so softmax gives them weight exactly 0.
        scores = scores.masked_fill(~mask, float("-inf"))

    # Softmax in at least float32: under bf16/fp16 autocast, exp and the sum are
    # too imprecise. Cast the weights back so the matmul with V stays low precision.
    compute_dtype = torch.promote_types(scores.dtype, torch.float32)
    weights = softmax(scores.to(compute_dtype), dim=-1).to(V.dtype)  # rows sum to 1
    return einsum(weights, V, "... q k, ... k d -> ... q d")


class CausalMultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention, with optional RoPE on q and k.

    Concept:
        Several attention "heads" run in parallel, each on its own d_k = d_model /
        num_heads slice of the features, so different heads can focus on
        different relations. Causal masking stops a token from attending to
        later tokens, which is what makes next-token training valid.

    Math:
        q = x Wq, k = x Wk, v = x Wv            (then split into heads)
        q, k = RoPE(q), RoPE(k)                 (v is NOT rotated)
        head_h = softmax(q_h k_h^T / sqrt(d_k) + causal mask) v_h
        out = concat(head_1 ... head_H) Wo

    Shapes:
        x:   (batch, seq, d_model)
        q,k,v after split: (batch, heads, seq, d_k)
        mask: (seq, seq) lower triangular, True = may attend
        out: (batch, seq, d_model)

    Design choice: separate q, k, v projections (three Linear layers) instead of
    one fused (3 * d_model) matrix. The fused version is a little faster on GPU
    because it launches one matmul, but separate layers keep each step explicit
    and map one-to-one onto the formulas above. The math is identical.

    Why RoPE touches only q and k: position should change *which tokens match*
    (the q . k score), not the content that gets copied out (v).
    """

    def __init__(self, d_model, num_heads, rope=None, device=None, dtype=None):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.rope = rope  # RotaryPositionalEmbedding or None (NoPE)

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x, token_positions=None):
        seq_len = x.shape[-2]

        # Project, then split d_model into (heads, d_k) and move heads next to batch.
        q = rearrange(self.q_proj(x), "... s (h d) -> ... h s d", h=self.num_heads)
        k = rearrange(self.k_proj(x), "... s (h d) -> ... h s d", h=self.num_heads)
        v = rearrange(self.v_proj(x), "... s (h d) -> ... h s d", h=self.num_heads)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            if token_positions.ndim > 1:
                # (batch, seq) -> (batch, 1, seq) so it broadcasts over heads.
                token_positions = token_positions.unsqueeze(-2)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # Lower triangular: query i may attend to keys j <= i.
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        )

        attended = scaled_dot_product_attention(q, k, v, causal_mask)

        # Merge the heads back: (..., heads, seq, d_k) -> (..., seq, d_model).
        merged = rearrange(attended, "... h s d -> ... s (h d)")
        return self.output_proj(merged)
