"""Tests for cs336_basics.data."""

import numpy as np
import pytest
import torch

from cs336_basics.data import (
    build_sample_corpus,
    get_batch,
    load_dataset,
    prepare_sample_data,
    tokenize_to_npy,
)


def test_get_batch_shapes_dtype_and_device():
    data = np.arange(1000, dtype=np.uint16)
    x, y = get_batch(data, batch_size=8, context_length=16, device="cpu")
    assert x.shape == (8, 16)
    assert y.shape == (8, 16)
    assert x.dtype == torch.int64
    assert y.dtype == torch.int64
    assert x.device.type == "cpu"


def test_get_batch_targets_are_inputs_shifted_by_one():
    data = np.arange(1000, dtype=np.uint16)  # token value == position
    x, y = get_batch(data, batch_size=16, context_length=12)
    assert torch.equal(y, x + 1)
    # Each row is a contiguous window.
    assert torch.equal(x[:, 1:], x[:, :-1] + 1)


def test_get_batch_values_stay_in_range():
    data = (np.arange(500) % 50).astype(np.uint16)
    x, y = get_batch(data, batch_size=32, context_length=20)
    assert x.min() >= 0 and x.max() < 50
    assert y.min() >= 0 and y.max() < 50


def test_get_batch_can_reach_the_last_valid_window():
    # Exactly one valid start (0) when num_tokens = context_length + 1.
    data = np.arange(9, dtype=np.uint16)
    x, y = get_batch(data, batch_size=4, context_length=8)
    assert torch.equal(x[0], torch.arange(8))
    assert torch.equal(y[0], torch.arange(1, 9))


def test_get_batch_never_reads_past_the_end():
    data = np.arange(30, dtype=np.uint16)
    rng = np.random.default_rng(0)
    for _ in range(200):
        x, y = get_batch(data, batch_size=4, context_length=10, rng=rng)
        assert y.max() <= 29


def test_get_batch_is_reproducible_with_seeded_rng():
    data = np.arange(1000, dtype=np.uint16)
    x1, _ = get_batch(data, 4, 8, rng=np.random.default_rng(7))
    x2, _ = get_batch(data, 4, 8, rng=np.random.default_rng(7))
    assert torch.equal(x1, x2)


def test_get_batch_rejects_short_dataset():
    with pytest.raises(ValueError):
        get_batch(np.arange(5, dtype=np.uint16), batch_size=2, context_length=5)


def test_get_batch_works_with_memmap(tmp_path):
    path = tmp_path / "tokens.npy"
    np.save(path, np.arange(2000, dtype=np.uint16))
    data = load_dataset(path)
    assert isinstance(data, np.memmap)
    x, y = get_batch(data, batch_size=4, context_length=16)
    assert torch.equal(y, x + 1)


def test_sample_corpus_is_deterministic_and_has_special_tokens():
    a = build_sample_corpus(20, seed=3)
    b = build_sample_corpus(20, seed=3)
    assert a == b
    assert a.count("<|endoftext|>") == 20
    assert build_sample_corpus(20, seed=4) != a


def test_prepare_sample_data_writes_expected_files(tmp_path):
    from cs336_basics.tokenizer import Tokenizer

    info = prepare_sample_data(str(tmp_path), vocab_size=290, num_docs=30)
    for name in ("vocab.json", "merges.txt", "train.npy", "val.npy"):
        assert (tmp_path / name).exists()

    train = np.load(tmp_path / "train.npy")
    val = np.load(tmp_path / "val.npy")
    assert train.dtype == np.uint16
    assert val.dtype == np.uint16
    assert len(train) == info["train_tokens"] > len(val) > 0
    assert int(train.max()) < info["vocab_size"]

    tok = Tokenizer.from_files(
        str(tmp_path / "vocab.json"), str(tmp_path / "merges.txt"), ["<|endoftext|>"]
    )
    text = (tmp_path / "corpus.txt").read_text(encoding="utf-8")
    # Train + val together are the whole corpus, split at a token boundary.
    all_tokens = np.concatenate([train, val]).tolist()
    assert tok.decode(all_tokens) == text


def test_tokenize_to_npy_respects_max_chars(tmp_path):
    from cs336_basics.tokenizer import Tokenizer, train_bpe

    text_path = tmp_path / "t.txt"
    text_path.write_text("hello world\n" * 200, encoding="utf-8")
    vocab, merges = train_bpe(str(text_path), 270, [], num_processes=1)
    tok = Tokenizer(vocab, merges)
    full = tokenize_to_npy(tok, text_path, tmp_path / "full.npy")
    part = tokenize_to_npy(tok, text_path, tmp_path / "part.npy", max_chars=100)
    assert 0 < part < full
