"""Core layers built from primitives: Linear, Embedding, RMSNorm, softmax, SwiGLU, RoPE.

Only nn.Module, nn.Parameter and plain tensor ops are used here. No nn.Linear,
nn.Embedding or nn.LayerNorm.
"""

import math

import torch
from einops import einsum
from torch import nn


class Linear(nn.Module):
    """Linear map without bias: y = x W^T.

    Concept:
        A learned matrix multiply. Every projection in the Transformer (q, k, v,
        output, FFN, LM head) is one of these.

    Math:
        y = W x, applied to the last dimension of x.

    Shapes:
        weight: (out_features, in_features)   stored as (out, in) so that
                                              y = x @ weight.T
        x:      (..., in_features)
        y:      (..., out_features)

    Init:
        Truncated normal, mean 0, std = sqrt(2 / (in + out)), cut at +-3 std.
        This keeps activation variance roughly stable across layers.
    """

    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        std = math.sqrt(2.0 / (in_features + out_features))
        # a and b are absolute cut points, so +-3 std means +-3 * std here.
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x):
        # "..." keeps any leading batch dims untouched.
        return einsum(x, self.weight, "... in_dim, out_dim in_dim -> ... out_dim")
