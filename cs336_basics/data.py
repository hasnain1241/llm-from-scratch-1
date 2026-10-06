"""Data utilities: batching from a token array, dataset preparation, sample corpus."""

import array
import os
import random

import numpy as np
import torch


# ------------------------------------------------------------------ batching
def load_dataset(path):
    """Open a 1D token array saved with np.save, memory-mapped (not read into RAM).

    Memory mapping lets you sample batches from a file much bigger than memory:
    only the pages that get sliced are loaded from disk.
    """
    return np.load(path, mmap_mode="r")


def get_batch(dataset, batch_size, context_length, device="cpu", rng=None):
    """Sample random training windows from a 1D token array.

    For a window starting at s:
        x = tokens[s     : s + context_length]
        y = tokens[s + 1 : s + context_length + 1]
    so y[t] is the token that follows x[t]. This is the next-token target.

    Shapes:
        dataset: (num_tokens,) integer array (np.ndarray or np.memmap)
        returns: x, y both (batch_size, context_length) int64 tensors on `device`

    The largest valid start is num_tokens - context_length - 1, so that the
    shifted window y still fits inside the array.
    """
    num_tokens = len(dataset)
    if num_tokens <= context_length:
        raise ValueError(
            f"dataset has {num_tokens} tokens, need more than context_length={context_length}"
        )
    if rng is None:
        rng = np.random.default_rng()

    # integers(low, high) excludes high, so max start = num_tokens - context_length - 1.
    starts = rng.integers(0, num_tokens - context_length, size=batch_size)

    # uint16 -> int64: torch embeddings and indexing need int64 ids.
    x = np.stack([dataset[s : s + context_length] for s in starts]).astype(np.int64)
    y = np.stack([dataset[s + 1 : s + 1 + context_length] for s in starts]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


# ------------------------------------------------------- dataset preparation
def tokenize_to_npy(tokenizer, text_path, out_path, max_chars=None):
    """Tokenize a text file line by line and save a uint16 .npy array.

    Uses tokenizer.encode_iterable on the open file, so the text is never fully
    loaded. Token ids are accumulated in a compact array('H') (2 bytes each).
    uint16 limits the vocabulary to 65536 ids. Returns the number of tokens.

    max_chars stops after roughly that many characters (useful to cap the cost
    of tokenizing a huge file in pure Python).
    """
    buffer = array.array("H")

    def limited_lines(f):
        seen = 0
        for line in f:
            yield line
            seen += len(line)
            if max_chars is not None and seen >= max_chars:
                return

    with open(text_path, "r", encoding="utf-8", newline="") as f:
        try:
            for token_id in tokenizer.encode_iterable(limited_lines(f)):
                buffer.append(token_id)
        except OverflowError as err:
            raise ValueError("token id does not fit in uint16; use a vocab of at most 65536") from err

    tokens = np.frombuffer(buffer, dtype=np.uint16)
    np.save(out_path, tokens)
    return len(tokens)


def prepare_dataset(
    input_path,
    out_dir,
    vocab_size,
    special_tokens=("<|endoftext|>",),
    val_input=None,
    val_fraction=0.1,
    tokenizer_train_path=None,
    num_processes=None,
    max_chars=None,
):
    """Train a BPE tokenizer (if needed) and write train.npy and val.npy.

    Files written to out_dir: vocab.json, merges.txt, train.npy, val.npy.
    If vocab.json and merges.txt already exist they are reused.

    Validation data comes from `val_input` if given, otherwise from the last
    `val_fraction` of the training tokens.
    """
    from cs336_basics.tokenizer import Tokenizer, save_bpe, train_bpe

    os.makedirs(out_dir, exist_ok=True)
    vocab_path = os.path.join(out_dir, "vocab.json")
    merges_path = os.path.join(out_dir, "merges.txt")
    special_tokens = list(special_tokens)

    if not (os.path.exists(vocab_path) and os.path.exists(merges_path)):
        print(f"Training BPE tokenizer (vocab_size={vocab_size}) ...")
        vocab, merges = train_bpe(
            tokenizer_train_path or input_path,
            vocab_size,
            special_tokens,
            num_processes=num_processes,
        )
        save_bpe(vocab, merges, vocab_path, merges_path)
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)

    train_path = os.path.join(out_dir, "train.npy")
    val_path = os.path.join(out_dir, "val.npy")
    print("Tokenizing training text ...")
    num_train = tokenize_to_npy(tokenizer, input_path, train_path, max_chars=max_chars)

    if val_input is not None:
        print("Tokenizing validation text ...")
        num_val = tokenize_to_npy(tokenizer, val_input, val_path, max_chars=max_chars)
    else:
        tokens = np.load(train_path)
        split = int(len(tokens) * (1 - val_fraction))
        np.save(train_path, tokens[:split])
        np.save(val_path, tokens[split:])
        num_train, num_val = split, len(tokens) - split

    print(f"Done: {num_train} train tokens, {num_val} val tokens, vocab size {len(tokenizer.vocab)}")
    return {"train_tokens": num_train, "val_tokens": num_val, "vocab_size": len(tokenizer.vocab)}


# -------------------------------------------------------------- sample corpus
_NAMES = ["Tom", "Lily", "Ben", "Mia", "Sam", "Anna", "Max", "Zoe"]
_ADJECTIVES = ["red", "small", "shiny", "old", "soft", "big", "happy", "blue"]
_THINGS = ["ball", "kite", "hat", "book", "box", "toy", "boat", "cake"]
_PLACES = ["park", "garden", "river", "school", "hill", "beach"]
_ANIMALS = ["dog", "cat", "bird", "fish", "rabbit", "frog"]
_ENDINGS = [
    "{name} was very happy.",
    "{name} smiled and went home.",
    "They played until the sun went down.",
    "It was a good day.",
]


def build_sample_corpus(num_docs=400, seed=0):
    """Build a tiny synthetic story corpus, so smoke tests work offline.

    Each document is a few template sentences. Documents are joined with the
    <|endoftext|> special token. The text is repetitive on purpose: a small
    model can learn it in a few dozen steps, which makes loss curves easy to read.
    """
    rng = random.Random(seed)
    docs = []
    for _ in range(num_docs):
        name = rng.choice(_NAMES)
        adj = rng.choice(_ADJECTIVES)
        thing = rng.choice(_THINGS)
        place = rng.choice(_PLACES)
        animal = rng.choice(_ANIMALS)
        sentences = [
            f"{name} had a {adj} {thing}.",
            f"{name} took it to the {place}.",
            f"At the {place}, {name} saw a {animal}.",
            f"The {animal} liked the {adj} {thing}.",
            rng.choice(_ENDINGS).format(name=name),
        ]
        docs.append(" ".join(sentences))
    return "<|endoftext|>".join(docs) + "<|endoftext|>"


def prepare_sample_data(out_dir, vocab_size=300, num_docs=400, num_processes=1):
    """Write the sample corpus to out_dir/corpus.txt and tokenize it."""
    os.makedirs(out_dir, exist_ok=True)
    corpus_path = os.path.join(out_dir, "corpus.txt")
    with open(corpus_path, "w", encoding="utf-8", newline="") as f:
        f.write(build_sample_corpus(num_docs))
    return prepare_dataset(
        corpus_path,
        out_dir,
        vocab_size=vocab_size,
        num_processes=num_processes,
    )
