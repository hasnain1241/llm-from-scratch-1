"""Tests for cs336_basics.model: block, language model, ablations, generation."""

import pytest
import torch
import torch.nn as nn

from cs336_basics.layers import RMSNorm, RotaryPositionalEmbedding, default_d_ff
from cs336_basics.model import (
    TransformerBlock,
    TransformerLM,
    generate,
    sample_next_token,
    top_p_filter,
)
from cs336_basics.optim import AdamW
from cs336_basics.utils import cross_entropy

CFG = dict(
    vocab_size=50,
    context_length=16,
    num_layers=2,
    d_model=32,
    num_heads=4,
    d_ff=64,
    rope_theta=10000.0,
)


def make_lm(**overrides):
    torch.manual_seed(0)
    return TransformerLM(**{**CFG, **overrides})


# ------------------------------------------------------------------- block
def test_block_shape():
    block = TransformerBlock(32, 4, 64)
    x = torch.randn(2, 10, 32)
    assert block(x).shape == (2, 10, 32)


def test_block_prenorm_structure():
    rope = RotaryPositionalEmbedding(10000.0, 8, 16)
    block = TransformerBlock(32, 4, 64, rope=rope)
    x = torch.randn(2, 6, 32)
    h = x + block.attn(block.ln1(x))
    expected = h + block.ffn(block.ln2(h))
    torch.testing.assert_close(block(x), expected)


def test_block_postnorm_structure():
    block = TransformerBlock(32, 4, 64, post_norm=True)
    x = torch.randn(2, 6, 32)
    h = block.ln1(x + block.attn(x))
    expected = block.ln2(h + block.ffn(h))
    torch.testing.assert_close(block(x), expected)


def test_block_without_rmsnorm_has_no_norm_parameters():
    block = TransformerBlock(32, 4, 64, use_rmsnorm=False)
    assert isinstance(block.ln1, nn.Identity)
    assert not any("ln" in name for name, _ in block.named_parameters())


def test_block_rejects_unknown_ffn_type():
    with pytest.raises(ValueError):
        TransformerBlock(32, 4, 64, ffn_type="relu")


def test_block_is_causal():
    rope = RotaryPositionalEmbedding(10000.0, 8, 16)
    block = TransformerBlock(32, 4, 64, rope=rope)
    x = torch.randn(1, 8, 32)
    x2 = x.clone()
    x2[:, 6:] = torch.randn(1, 2, 32)
    torch.testing.assert_close(block(x)[:, :6], block(x2)[:, :6])


# ---------------------------------------------------------------------- LM
def test_lm_forward_shape_and_dtype():
    model = make_lm()
    tokens = torch.randint(0, 50, (3, 12))
    logits = model(tokens)
    assert logits.shape == (3, 12, 50)
    assert logits.dtype == torch.float32


def test_lm_accepts_shorter_than_context_and_rejects_longer():
    model = make_lm()
    assert model(torch.randint(0, 50, (1, 3))).shape == (1, 3, 50)
    with pytest.raises(ValueError):
        model(torch.randint(0, 50, (1, 17)))


def test_lm_causality_future_token_does_not_change_earlier_logits():
    model = make_lm().eval()
    tokens = torch.randint(0, 50, (2, 12))
    changed = tokens.clone()
    changed[:, 8] = (changed[:, 8] + 1) % 50  # change position 8
    with torch.no_grad():
        a = model(tokens)
        b = model(changed)
    torch.testing.assert_close(a[:, :8], b[:, :8])
    assert not torch.allclose(a[:, 8:], b[:, 8:])


def test_lm_logits_at_a_position_ignore_everything_after_it():
    model = make_lm().eval()
    tokens = torch.randint(0, 50, (1, 10))
    with torch.no_grad():
        full = model(tokens)
        prefix = model(tokens[:, :5])
    torch.testing.assert_close(full[:, :5], prefix, atol=1e-5, rtol=1e-5)


def expected_param_count(vocab, layers, d, d_ff, ffn="swiglu", norms=True):
    attn = 4 * d * d
    ffn_params = 3 * d * d_ff if ffn == "swiglu" else 2 * d * d_ff
    block_norms = 2 * d if norms else 0
    final_norm = d if norms else 0
    return vocab * d + layers * (attn + ffn_params + block_norms) + final_norm + vocab * d


def test_param_count_matches_formula_swiglu():
    model = make_lm()
    assert model.num_parameters() == expected_param_count(50, 2, 32, 64)
    assert model.num_parameters(non_embedding=True) == expected_param_count(50, 2, 32, 64) - 50 * 32


def test_param_count_matches_formula_silu_ffn():
    model = make_lm(ffn_type="silu", d_ff=128)
    assert model.num_parameters() == expected_param_count(50, 2, 32, 128, ffn="silu")


def test_default_d_ff():
    assert make_lm(d_ff=None).d_ff == default_d_ff(32)
    assert make_lm(d_ff=None, ffn_type="silu").d_ff == 4 * 32


def test_flops_per_token_formula():
    model = make_lm()
    matmul_weights = (
        2 * (4 * 32 * 32 + 3 * 32 * 64)  # blocks
        + 32 * 50  # lm head
    )
    expected = 2 * matmul_weights + 4 * 2 * 16 * 32
    assert model.flops_per_token(16) == expected


def test_lm_gradients_reach_every_parameter():
    model = make_lm()
    tokens = torch.randint(0, 50, (2, 8))
    cross_entropy(model(tokens), tokens).backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name


def test_lm_has_no_forbidden_torch_layers():
    model = make_lm()
    forbidden = (nn.Linear, nn.Embedding, nn.LayerNorm, nn.MultiheadAttention)
    assert not any(isinstance(m, forbidden) for m in model.modules())


def test_lm_can_overfit_one_batch():
    model = make_lm(num_layers=2)
    gen = torch.Generator().manual_seed(0)
    x = torch.randint(0, 50, (4, 16), generator=gen)
    y = torch.randint(0, 50, (4, 16), generator=gen)
    opt = AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    first = None
    for _ in range(60):
        opt.zero_grad()
        loss = cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < 0.7 * first


# --------------------------------------------------------------- ablations
@pytest.mark.parametrize(
    "flags",
    [
        {"use_rmsnorm": False},
        {"post_norm": True},
        {"use_rope": False},
        {"ffn_type": "silu"},
        {"use_rmsnorm": False, "use_rope": False, "ffn_type": "silu"},
    ],
)
def test_ablation_variants_run(flags):
    model = make_lm(**flags)
    tokens = torch.randint(0, 50, (2, 10))
    logits = model(tokens)
    assert logits.shape == (2, 10, 50)
    assert torch.isfinite(logits).all()


def test_nope_model_has_no_rope_and_still_causal():
    model = make_lm(use_rope=False).eval()
    assert all(block.attn.rope is None for block in model.blocks)
    tokens = torch.randint(0, 50, (1, 10))
    changed = tokens.clone()
    changed[:, 7] = (changed[:, 7] + 1) % 50
    with torch.no_grad():
        torch.testing.assert_close(model(tokens)[:, :7], model(changed)[:, :7])


def test_norm_free_and_postnorm_have_no_final_norm():
    assert isinstance(make_lm(use_rmsnorm=False).ln_final, nn.Identity)
    assert isinstance(make_lm(post_norm=True).ln_final, nn.Identity)
    assert isinstance(make_lm().ln_final, RMSNorm)


def test_rope_is_shared_across_layers():
    model = make_lm()
    ropes = {id(block.attn.rope) for block in model.blocks}
    assert len(ropes) == 1


# ---------------------------------------------------------- top-p / sampling
PROBS = torch.tensor([0.5, 0.3, 0.15, 0.05])


def test_top_p_keeps_smallest_set_reaching_mass():
    out = top_p_filter(PROBS, 0.75)  # 0.5 < 0.75, 0.8 >= 0.75 -> keep two
    torch.testing.assert_close(out, torch.tensor([0.5 / 0.8, 0.3 / 0.8, 0.0, 0.0]))

    out = top_p_filter(PROBS, 0.85)  # needs the third token
    torch.testing.assert_close(out, torch.tensor([0.5, 0.3, 0.15, 0.0]) / 0.95)


def test_top_p_one_keeps_everything_and_tiny_keeps_only_top():
    torch.testing.assert_close(top_p_filter(PROBS, 1.0), PROBS)
    torch.testing.assert_close(top_p_filter(PROBS, 0.01), torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_top_p_handles_unsorted_input():
    probs = torch.tensor([0.15, 0.5, 0.05, 0.3])
    out = top_p_filter(probs, 0.75)
    torch.testing.assert_close(out, torch.tensor([0.0, 0.5 / 0.8, 0.0, 0.3 / 0.8]))


def test_top_p_output_sums_to_one_and_validates_range():
    out = top_p_filter(torch.softmax(torch.randn(3, 20), dim=-1), 0.9)
    torch.testing.assert_close(out.sum(-1), torch.ones(3))
    with pytest.raises(ValueError):
        top_p_filter(PROBS, 0.0)
    with pytest.raises(ValueError):
        top_p_filter(PROBS, 1.5)


def test_sampling_only_draws_from_the_nucleus():
    logits = torch.log(PROBS)
    gen = torch.Generator().manual_seed(0)
    draws = {sample_next_token(logits, 1.0, 0.75, gen) for _ in range(300)}
    assert draws == {0, 1}


def test_sampling_without_top_p_can_draw_every_token():
    logits = torch.log(PROBS)
    gen = torch.Generator().manual_seed(0)
    draws = {sample_next_token(logits, 1.0, 1.0, gen) for _ in range(2000)}
    assert draws == {0, 1, 2, 3}


def test_temperature_near_zero_is_greedy():
    logits = torch.tensor([1.0, 3.0, 2.0, 0.5])
    gen = torch.Generator().manual_seed(0)
    assert {sample_next_token(logits, 1e-4, 1.0, gen) for _ in range(50)} == {1}
    assert sample_next_token(logits, 0.0) == 1


def test_high_temperature_spreads_probability():
    logits = torch.tensor([1.0, 3.0, 2.0, 0.5])
    gen = torch.Generator().manual_seed(0)
    draws = {sample_next_token(logits, 50.0, 1.0, gen) for _ in range(500)}
    assert len(draws) >= 3


# ----------------------------------------------------------------- generate
class CountingModel(nn.Module):
    """Stub LM: always predicts (last token + 1) mod vocab with near certainty."""

    def __init__(self, vocab=10, context_length=8):
        super().__init__()
        self.vocab = vocab
        self.context_length = context_length
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        assert x.shape[-1] <= self.context_length, "generate must crop to the context window"
        logits = torch.zeros(*x.shape, self.vocab)
        nxt = (x + 1) % self.vocab
        logits.scatter_(-1, nxt.unsqueeze(-1), 50.0)
        return logits


class StubTokenizer:
    token_to_id = {b"<|endoftext|>": 5}

    def encode(self, text):
        return [int(t) for t in text.split()]

    def decode(self, ids):
        return " ".join(str(i) for i in ids)


def test_generate_stops_at_end_of_text():
    out = generate(CountingModel(), StubTokenizer(), "2", max_new_tokens=20, temperature=0.0)
    assert out == "2 3 4"  # next would be 5, the end-of-text id, so it stops


def test_generate_respects_max_new_tokens():
    out = generate(CountingModel(), StubTokenizer(), "0", max_new_tokens=3, temperature=0.0)
    assert out == "0 1 2 3"


def test_generate_low_temperature_matches_greedy():
    greedy = generate(CountingModel(), StubTokenizer(), "6", max_new_tokens=3, temperature=0.0)
    cold = generate(CountingModel(), StubTokenizer(), "6", max_new_tokens=3, temperature=1e-4)
    assert greedy == cold == "6 7 8 9"


def test_generate_empty_prompt_starts_from_end_of_text_without_echoing_it():
    out = generate(CountingModel(), StubTokenizer(), "", max_new_tokens=2, temperature=0.0)
    assert out == "6 7"  # started from id 5, which is not echoed


def test_generate_crops_to_context_window_and_restores_mode():
    model = CountingModel(vocab=20, context_length=2)
    model.train()
    out = generate(model, StubTokenizer(), "6", max_new_tokens=6, temperature=0.0)
    assert out == "6 7 8 9 10 11 12"  # the stub asserts the input never exceeds 2 tokens
    assert model.training  # training mode is restored after generation


def test_generate_with_real_model_runs():
    model = make_lm()

    class ByteTok:
        token_to_id = {b"<|endoftext|>": 49}

        def encode(self, text):
            return [b % 49 for b in text.encode()]

        def decode(self, ids):
            return ",".join(map(str, ids))

    out = generate(model, ByteTok(), "hi", max_new_tokens=5, temperature=1.0, top_p=0.9)
    assert isinstance(out, str) and len(out) > 0
