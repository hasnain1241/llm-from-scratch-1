"""Transformer language model: TransformerBlock, TransformerLM, and text generation.

Architecture (decoder-only, pre-norm, as in modern LLMs like Llama):

    token ids (batch, seq)
      -> Embedding                          (batch, seq, d_model)
      -> N x TransformerBlock               (batch, seq, d_model)
      -> final RMSNorm                      (batch, seq, d_model)
      -> LM head (Linear)                   (batch, seq, vocab)   logits

    TransformerBlock (pre-norm):
        x = x + Attention(RMSNorm(x))
        x = x + FFN(RMSNorm(x))

The residual path stays "clean": norms sit inside each branch, so gradients flow
straight through the additions. That makes deep stacks easier to train than
post-norm.

Ablation switches (all default to the standard model):
    use_rmsnorm=False  no normalization at all
    post_norm=True     x = RMSNorm(x + Attn(x)), x = RMSNorm(x + FFN(x))
    use_rope=False     no positional encoding (NoPE), relying on causal masking
    ffn_type="silu"    plain SiLU FFN instead of SwiGLU
"""

import torch
from torch import nn

from cs336_basics.attention import CausalMultiHeadSelfAttention
from cs336_basics.layers import (
    Embedding,
    Linear,
    RMSNorm,
    RotaryPositionalEmbedding,
    SiLUFFN,
    SwiGLU,
    default_d_ff,
    softmax,
)


class TransformerBlock(nn.Module):
    """One Transformer layer: attention sublayer, then feed-forward sublayer.

    Shapes:
        x:   (batch, seq, d_model)
        out: (batch, seq, d_model)

    Pre-norm (default):
        x = x + Attn(RMSNorm(x));  x = x + FFN(RMSNorm(x))
    Post-norm (ablation):
        x = RMSNorm(x + Attn(x));  x = RMSNorm(x + FFN(x))
    With use_rmsnorm=False the norms are identity functions.
    """

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        rope=None,
        use_rmsnorm=True,
        post_norm=False,
        ffn_type="swiglu",
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.post_norm = post_norm

        self.attn = CausalMultiHeadSelfAttention(
            d_model, num_heads, rope=rope, device=device, dtype=dtype
        )
        if ffn_type == "swiglu":
            self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        elif ffn_type == "silu":
            self.ffn = SiLUFFN(d_model, d_ff, device=device, dtype=dtype)
        else:
            raise ValueError(f"unknown ffn_type {ffn_type!r}, use 'swiglu' or 'silu'")

        if use_rmsnorm:
            self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
            self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        else:
            self.ln1 = nn.Identity()
            self.ln2 = nn.Identity()

    def forward(self, x, token_positions=None):
        if self.post_norm:
            x = self.ln1(x + self.attn(x, token_positions))
            x = self.ln2(x + self.ffn(x))
        else:
            x = x + self.attn(self.ln1(x), token_positions)
            x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    """Decoder-only Transformer language model.

    Forward:
        token_ids: (batch, seq) int64, seq <= context_length
        returns:   (batch, seq, vocab_size) float logits; logits[b, t] scores the
                   token that follows position t, using only tokens 0..t.

    d_ff defaults to about 8/3 * d_model (SwiGLU) or 4 * d_model (SiLU FFN).
    One RoPE module (cos/sin tables) is shared by all layers.

    With post_norm or use_rmsnorm=False there is no final norm before the LM head.
    """

    def __init__(
        self,
        vocab_size,
        context_length,
        num_layers,
        d_model,
        num_heads,
        d_ff=None,
        rope_theta=10000.0,
        use_rmsnorm=True,
        post_norm=False,
        use_rope=True,
        ffn_type="swiglu",
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads

        if d_ff is None:
            d_ff = default_d_ff(d_model) if ffn_type == "swiglu" else 4 * d_model
        self.d_ff = d_ff

        self.tok_emb = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        rope = None
        if use_rope:
            rope = RotaryPositionalEmbedding(
                rope_theta, d_model // num_heads, context_length, device=device
            )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    rope=rope,
                    use_rmsnorm=use_rmsnorm,
                    post_norm=post_norm,
                    ffn_type=ffn_type,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        if use_rmsnorm and not post_norm:
            self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        else:
            self.ln_final = nn.Identity()
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids):
        seq_len = token_ids.shape[-1]
        if seq_len > self.context_length:
            raise ValueError(
                f"sequence length {seq_len} exceeds context_length {self.context_length}"
            )

        x = self.tok_emb(token_ids)  # (batch, seq, d_model)
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        return self.lm_head(x)  # (batch, seq, vocab)

    def num_parameters(self, non_embedding=False):
        """Total parameter count. non_embedding=True excludes the token embedding table."""
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total -= self.tok_emb.weight.numel()
        return total

    def flops_per_token(self, seq_len=None):
        """Approximate forward-pass FLOPs per token.

        Matmul FLOPs: 2 * (number of weights in 2D layers, excluding the token
        embedding, which is a lookup). Attention scores and the weighted sum of
        values add 4 * num_layers * seq_len * d_model (2 for QK^T, 2 for AV).
        Training costs about 3x the forward pass (forward plus 2x for backward).
        """
        if seq_len is None:
            seq_len = self.context_length
        matmul_weights = sum(
            p.numel()
            for name, p in self.named_parameters()
            if p.ndim == 2 and name != "tok_emb.weight"
        )
        attention = 4 * self.num_layers * seq_len * self.d_model
        return 2 * matmul_weights + attention


# ---------------------------------------------------------------- generation
def top_p_filter(probs, top_p):
    """Nucleus filtering: keep the smallest set of top tokens with mass >= top_p.

    Sort probabilities from high to low. A token is kept if the probability mass
    of all tokens ranked before it is still below top_p, so the top token is
    always kept. Everything else is zeroed and the rest renormalized.

    Shapes:
        probs: (..., vocab) probabilities summing to 1 over the last dim
        returns: same shape, summing to 1
    """
    if not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")

    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    mass_before = torch.cumsum(sorted_probs, dim=-1) - sorted_probs
    keep = mass_before < top_p
    sorted_probs = sorted_probs * keep

    filtered = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)  # undo the sort
    return filtered / filtered.sum(dim=-1, keepdim=True)


def sample_next_token(logits, temperature=1.0, top_p=1.0, generator=None):
    """Pick the next token id from a 1D logits vector of shape (vocab,).

    temperature <= 0 means greedy (argmax). Otherwise sample from
    softmax(logits / temperature), optionally restricted by top-p. Low
    temperature sharpens the distribution toward the argmax, high temperature
    flattens it.
    """
    if temperature <= 0:
        return int(torch.argmax(logits))
    probs = softmax(logits.to(torch.float32) / temperature, dim=-1)
    if top_p < 1.0:
        probs = top_p_filter(probs, top_p)
    return int(torch.multinomial(probs, num_samples=1, generator=generator))


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=100,
    temperature=1.0,
    top_p=1.0,
    eos_token="<|endoftext|>",
    generator=None,
):
    """Autoregressively extend `prompt` and return the decoded text.

    Each step runs the model on the (cropped) sequence so far, takes the logits
    at the last position, samples one token, and appends it. Stops after
    max_new_tokens or when the end-of-text token is sampled (it is not included
    in the output). An empty prompt starts from the end-of-text token.
    """
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device

    eos_id = tokenizer.token_to_id.get(eos_token.encode("utf-8"))
    ids = tokenizer.encode(prompt)
    first_output = 0
    if not ids:
        if eos_id is None:
            raise ValueError("empty prompt and no end-of-text token to start from")
        ids = [eos_id]
        first_output = 1  # do not echo the artificial start token

    for _ in range(max_new_tokens):
        context = ids[-model.context_length :]  # the model only sees its context window
        x = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(x)[0, -1]  # (vocab,) logits for the next token
        next_id = sample_next_token(logits, temperature, top_p, generator)
        if eos_id is not None and next_id == eos_id:
            break
        ids.append(next_id)

    model.train(was_training)
    return tokenizer.decode(ids[first_output:])
