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


class Embedding(nn.Module):
    """Lookup table that maps integer token ids to vectors.

    Concept:
        Each token id owns one learned row of the table. Looking up a row is
        the same as multiplying a one-hot vector by the table, without
        materializing the one-hot.

    Shapes:
        weight:    (vocab_size, d_model)
        token_ids: (...,)  integer tensor, e.g. (batch, seq)
        output:    (..., d_model)

    Init:
        Truncated normal, mean 0, std 1, cut at +-3.
    """

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids):
        # Advanced indexing: each id selects a row, giving (..., d_model).
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """Root mean square layer normalization.

    Concept:
        Rescales each token's vector to unit root-mean-square, then applies a
        learned per-feature gain. Unlike LayerNorm it does not subtract the mean
        and has no bias, which makes it cheaper and works as well in practice.

    Math:
        RMS(x) = sqrt(mean(x_i^2) + eps)
        y_i    = x_i / RMS(x) * g_i

    Shapes:
        x:    (..., d_model)
        gain: (d_model,)
        y:    (..., d_model), same dtype as x

    The norm is computed in float32 and cast back, because squaring and
    averaging in bf16 or fp16 loses precision (or overflows in fp16).
    """

    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # Mean over the feature dim only; keepdim so it broadcasts back.
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(mean_square + self.eps)

        # Apply the gain in float32 too, then cast back to the input dtype.
        out = x_normed * self.weight.to(torch.float32)
        return out.to(in_dtype)


def softmax(x, dim=-1):
    """Numerically stable softmax along one dimension.

    Math:
        softmax(x)_i = exp(x_i) / sum_j exp(x_j)

    Stability:
        exp overflows for large x (exp(1000) = inf). Softmax is unchanged if the
        same constant is subtracted from every entry, so we subtract the max.
        The largest exponent is then exp(0) = 1, and nothing overflows.

    Shapes:
        x:      (...) any shape
        output: same shape as x, sums to 1 along `dim`
    """
    # keepdim so the max broadcasts back against x.
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def silu(x):
    """SiLU (a.k.a. swish): x * sigmoid(x) = x / (1 + exp(-x)).

    Smooth, non-monotonic cousin of ReLU. Shape is unchanged.
    """
    return x * torch.sigmoid(x)


def default_d_ff(d_model, multiple_of=64):
    """About 8/3 * d_model, rounded to the nearest multiple of `multiple_of`.

    Why 8/3: a standard FFN has 2 matrices of size d_model x 4*d_model. SwiGLU
    has 3 matrices, so using 8/3 * d_model keeps the parameter count about equal
    (3 * 8/3 = 8 = 2 * 4). Multiples of 64 keep matmuls hardware friendly.
    """
    d_ff = int(round(8 * d_model / 3 / multiple_of)) * multiple_of
    return max(d_ff, multiple_of)


class SwiGLU(nn.Module):
    """Position-wise feed-forward network with a gated SiLU (SwiGLU).

    Concept:
        One linear branch is passed through SiLU and used as a gate that
        multiplies a second linear branch, element by element. A third matrix
        projects back to d_model.

    Math:
        FFN(x) = W2 ( SiLU(W1 x) * (W3 x) )

    Shapes:
        x:  (..., d_model)
        W1: (d_ff, d_model)   gate branch
        W3: (d_ff, d_model)   value branch
        W2: (d_model, d_ff)   output projection
        y:  (..., d_model)

    If d_ff is None it defaults to default_d_ff(d_model).
    """

    def __init__(self, d_model, d_ff=None, device=None, dtype=None):
        super().__init__()
        if d_ff is None:
            d_ff = default_d_ff(d_model)
        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x):
        gate = silu(self.w1(x))  # (..., d_ff)
        value = self.w3(x)  # (..., d_ff)
        return self.w2(gate * value)  # (..., d_model)
