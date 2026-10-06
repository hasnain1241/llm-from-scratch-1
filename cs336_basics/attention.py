"""Attention: scaled dot-product attention and causal multi-head self-attention."""

import math

import torch
from einops import einsum

from cs336_basics.layers import softmax


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

    weights = softmax(scores, dim=-1)  # each row sums to 1 over the keys
    return einsum(weights, V, "... q k, ... k d -> ... q d")
