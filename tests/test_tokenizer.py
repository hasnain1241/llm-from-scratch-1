"""Tests for cs336_basics.tokenizer."""

import pytest

from cs336_basics.data import build_sample_corpus
from cs336_basics.tokenizer import (
    Tokenizer,
    save_bpe,
    train_bpe,
    train_bpe_reference,
)

EOT = "<|endoftext|>"

# The classic worked example. See docs/NOTES.md for the step-by-step derivation.
WORKED_CORPUS = (
    "low low low low low lower lower widest widest widest "
    "newest newest newest newest newest newest" + EOT
)
WORKED_MERGES = [
    (b"s", b"t"),
    (b"e", b"st"),
    (b"o", b"w"),
    (b"l", b"ow"),
    (b"w", b"est"),
    (b"n", b"e"),
    (b"ne", b"west"),
    (b" ", b"newest"),
    (b" ", b"low"),
    (b"w", b"i"),
]
WORKED_VOCAB_SIZE = 256 + 1 + len(WORKED_MERGES)  # bytes + 1 special + 10 merges


@pytest.fixture
def worked_path(tmp_path):
    path = tmp_path / "worked.txt"
    path.write_text(WORKED_CORPUS, encoding="utf-8")
    return str(path)


@pytest.fixture
def sample_path(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text(build_sample_corpus(150, seed=1), encoding="utf-8", newline="")
    return str(path)


# ------------------------------------------------------- BPE training
@pytest.mark.parametrize("trainer", [train_bpe_reference, train_bpe])
def test_worked_example_merges(trainer, worked_path):
    vocab, merges = trainer(worked_path, WORKED_VOCAB_SIZE, [EOT])
    assert merges == WORKED_MERGES


@pytest.mark.parametrize("trainer", [train_bpe_reference, train_bpe])
def test_worked_example_vocab_layout(trainer, worked_path):
    vocab, merges = trainer(worked_path, WORKED_VOCAB_SIZE, [EOT])
    assert len(vocab) == WORKED_VOCAB_SIZE
    assert [vocab[i] for i in range(256)] == [bytes([i]) for i in range(256)]
    assert vocab[256] == EOT.encode()
    for offset, (first, second) in enumerate(WORKED_MERGES):
        assert vocab[257 + offset] == first + second


def test_train_bpe_stops_when_no_pairs_remain(worked_path):
    vocab, merges = train_bpe(worked_path, 5000, [EOT])
    assert len(vocab) < 5000
    assert len(vocab) == 256 + 1 + len(merges)


def test_train_bpe_rejects_too_small_vocab(worked_path):
    with pytest.raises(ValueError):
        train_bpe(worked_path, 100, [EOT])


@pytest.mark.parametrize("trainer", [train_bpe_reference, train_bpe])
def test_merges_never_cross_special_tokens(trainer, tmp_path):
    path = tmp_path / "docs.txt"
    path.write_text((WORKED_CORPUS) * 2, encoding="utf-8")
    vocab, merges = trainer(str(path), 400, [EOT])
    for first, second in merges:
        merged = first + second
        assert not any(ch in merged for ch in (b"<", b"|", b">")), merged
    assert sum(1 for t in vocab.values() if t == EOT.encode()) == 1


def test_fast_trainer_matches_reference_on_sample_corpus(sample_path):
    ref_vocab, ref_merges = train_bpe_reference(sample_path, 400, [EOT])
    fast_vocab, fast_merges = train_bpe(sample_path, 400, [EOT], num_processes=1)
    assert fast_merges == ref_merges
    assert fast_vocab == ref_vocab


def test_parallel_pretokenization_matches_reference(sample_path):
    ref_vocab, ref_merges = train_bpe_reference(sample_path, 400, [EOT])
    par_vocab, par_merges = train_bpe(sample_path, 400, [EOT], num_processes=3)
    assert par_merges == ref_merges
    assert par_vocab == ref_vocab


def test_trainers_agree_without_special_tokens_and_with_unicode(tmp_path):
    path = tmp_path / "uni.txt"
    path.write_text("héllo wörld 日本語 🙃 héllo wörld\nab ab ab abab\n" * 20, encoding="utf-8")
    ref = train_bpe_reference(str(path), 300, [])
    fast = train_bpe(str(path), 300, [], num_processes=2)
    assert ref == fast


# ----------------------------------------------------------- Tokenizer
@pytest.fixture
def worked_tokenizer(worked_path):
    vocab, merges = train_bpe(worked_path, WORKED_VOCAB_SIZE, [EOT])
    return Tokenizer(vocab, merges, [EOT])


@pytest.fixture
def byte_tokenizer():
    """No merges: every byte is its own token."""
    vocab = {i: bytes([i]) for i in range(256)}
    return Tokenizer(vocab, [], [EOT])


@pytest.mark.parametrize(
    "text",
    [
        "",
        "hello world",
        "Hello, World! 123 \n\t  trailing   ",
        "héllo wörld",
        "日本語のテキスト",
        "emoji 🙃 and 👨‍👩‍👧 family",
        "don't we're they'll I'm",
        EOT,
        "a" + EOT + "b",
        EOT + EOT,
    ],
)
def test_roundtrip(text, byte_tokenizer, worked_tokenizer):
    for tok in (byte_tokenizer, worked_tokenizer):
        assert tok.decode(tok.encode(text)) == text


def test_empty_string_encodes_to_empty_list(byte_tokenizer):
    assert byte_tokenizer.encode("") == []
    assert byte_tokenizer.decode([]) == ""


def test_byte_tokenizer_encodes_utf8_bytes(byte_tokenizer):
    assert byte_tokenizer.encode("hello") == list(b"hello")
    assert byte_tokenizer.encode("é") == list("é".encode("utf-8"))


def test_merges_are_applied(worked_tokenizer):
    ids = worked_tokenizer.encode(" newest")
    assert ids == [worked_tokenizer.token_to_id[b" newest"]]
    assert worked_tokenizer.encode("low") == [worked_tokenizer.token_to_id[b"low"]]


def test_merges_apply_in_rank_order(worked_tokenizer):
    # "lower" -> "low" + "e" + "r" ("e","r" was never merged).
    ids = worked_tokenizer.encode("lower")
    pieces = [worked_tokenizer.vocab[i] for i in ids]
    assert pieces == [b"low", b"e", b"r"]


def test_special_token_is_a_single_id(worked_tokenizer):
    ids = worked_tokenizer.encode("low" + EOT + "low")
    eot_id = worked_tokenizer.token_to_id[EOT.encode()]
    assert ids.count(eot_id) == 1
    assert ids[1] == eot_id


def test_overlapping_special_tokens_use_longest_match():
    vocab = {i: bytes([i]) for i in range(256)}
    double = EOT + EOT
    tok = Tokenizer(vocab, [], [EOT, double])  # shorter one listed first on purpose
    ids = tok.encode("a" + double + "b")
    assert len(ids) == 3
    assert tok.vocab[ids[1]] == double.encode()
    # Three in a row: one double plus one single.
    ids3 = tok.encode(EOT * 3)
    assert [tok.vocab[i] for i in ids3] == [double.encode(), EOT.encode()]


def test_special_token_missing_from_vocab_is_appended():
    vocab = {i: bytes([i]) for i in range(256)}
    tok = Tokenizer(vocab, [], ["<|new|>"])
    ids = tok.encode("x<|new|>")
    assert ids[-1] == 256
    assert tok.decode(ids) == "x<|new|>"


def test_tokenizer_without_special_tokens_treats_them_as_text(byte_tokenizer):
    plain = Tokenizer({i: bytes([i]) for i in range(256)}, [])
    ids = plain.encode(EOT)
    assert len(ids) == len(EOT.encode())


def test_encode_iterable_matches_encode(worked_tokenizer):
    lines = ["low lower newest\n", "widest low" + EOT + "\n", "newest newest\n"]
    expected = worked_tokenizer.encode("".join(lines))
    assert list(worked_tokenizer.encode_iterable(lines)) == expected


def test_encode_iterable_is_lazy(byte_tokenizer):
    consumed = []

    def lines():
        for i in range(3):
            consumed.append(i)
            yield f"line {i}\n"

    stream = byte_tokenizer.encode_iterable(lines())
    next(stream)
    assert consumed == [0]  # only the first line has been read


def test_encode_iterable_from_file_handle(worked_tokenizer, tmp_path):
    path = tmp_path / "t.txt"
    path.write_text("low lower\nnewest widest\n", encoding="utf-8")
    with open(path, encoding="utf-8") as f:
        ids = list(worked_tokenizer.encode_iterable(f))
    assert worked_tokenizer.decode(ids) == "low lower\nnewest widest\n"


def test_decode_handles_invalid_utf8(byte_tokenizer):
    # 0xff alone is not valid UTF-8; decoding must not raise.
    assert byte_tokenizer.decode([0xFF]) == "�"


def test_from_files_roundtrip(worked_path, tmp_path):
    vocab, merges = train_bpe(worked_path, WORKED_VOCAB_SIZE, [EOT])
    vocab_file = tmp_path / "vocab.json"
    merges_file = tmp_path / "merges.txt"
    save_bpe(vocab, merges, str(vocab_file), str(merges_file))

    original = Tokenizer(vocab, merges, [EOT])
    loaded = Tokenizer.from_files(str(vocab_file), str(merges_file), [EOT])
    assert loaded.vocab == original.vocab
    assert loaded.merges == original.merges
    text = "low lower newest widest" + EOT + "zzz 🙃"
    assert loaded.encode(text) == original.encode(text)


def test_trained_tokenizer_compresses_training_text(sample_path):
    vocab, merges = train_bpe(sample_path, 400, [EOT], num_processes=1)
    tok = Tokenizer(vocab, merges, [EOT])
    text = build_sample_corpus(10, seed=1)
    ids = tok.encode(text)
    assert len(ids) < len(text.encode("utf-8"))
    assert tok.decode(ids) == text


# ------------------------------------------------------ optional: tiktoken
@pytest.mark.optional
def test_matches_tiktoken_gpt2():
    """Compare with GPT-2 encodings. Skipped without tiktoken or network access.

    GPT-2 merges are recovered from tiktoken's ranks: for each multi-byte token,
    pick the split into two lower-ranked tokens whose larger rank is smallest.
    If this test disagrees, suspect that recovery heuristic before the Tokenizer.
    """
    tiktoken = pytest.importorskip("tiktoken")
    try:
        enc = tiktoken.get_encoding("gpt2")
    except Exception as err:  # no network to fetch the vocab files
        pytest.skip(f"cannot load gpt2 encoding: {err}")

    ranks = enc._mergeable_ranks  # dict[bytes, int]
    vocab = {rank: token for token, rank in ranks.items()}
    merges = []
    for token, rank in sorted(ranks.items(), key=lambda kv: kv[1]):
        if len(token) == 1:
            continue
        best = None
        for i in range(1, len(token)):
            left, right = token[:i], token[i:]
            if left in ranks and right in ranks and max(ranks[left], ranks[right]) < rank:
                score = max(ranks[left], ranks[right])
                if best is None or score < best[0]:
                    best = (score, (left, right))
        merges.append(best[1])

    tok = Tokenizer(vocab, merges, ["<|endoftext|>"])
    for text in ["Hello, world!", "  leading spaces", "tab\tand\nnewline", "héllo 🙃", "don't stop"]:
        assert tok.encode(text) == enc.encode(text)
