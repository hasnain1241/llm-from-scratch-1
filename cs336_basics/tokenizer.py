"""Byte-level BPE: tokenizer training (reference and optimized) and a Tokenizer class.

Concept:
    Byte-pair encoding starts from the 256 possible bytes and repeatedly merges
    the most frequent adjacent pair of tokens into a new token. After N merges
    we have a vocabulary of 256 + N tokens. Working on bytes means any text,
    including emoji and unseen scripts, can be encoded with no "unknown" token.

Pipeline:
    1. Split the text on special tokens (like <|endoftext|>) so merges never
       cross them.
    2. Pre-tokenize each piece with the GPT-2 regex (words, numbers,
       punctuation, whitespace). Merges happen only inside a pre-token.
    3. Count pre-tokens, represent each as a sequence of single bytes.
    4. Repeat: count adjacent pairs (weighted by pre-token counts), merge the
       best pair everywhere. Ties are broken by taking the lexicographically
       greater pair of byte strings.

Vocabulary layout:
    ids 0..255                 the single bytes
    next len(special_tokens)   the special tokens, in the order given
    after that                 merged tokens, in merge order

Two trainers are provided and give identical output:
    train_bpe_reference: short and obviously correct, but slow. It recounts all
                         pairs from scratch after every merge.
    train_bpe:           caches pre-token counts, updates pair counts
                         incrementally using an index from pair to the words
                         that contain it, and pre-tokenizes in parallel.
"""

import json
import multiprocessing
import os
from collections import Counter, defaultdict

import regex

# GPT-2 pre-tokenization pattern.
GPT2_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_PRETOKEN_RE = regex.compile(GPT2_PATTERN)


# ----------------------------------------------------------------- helpers
def _split_on_special_tokens(text, special_tokens):
    """Split text on special tokens and drop them. Longest token matches first."""
    if not special_tokens:
        return [text]
    pattern = "|".join(regex.escape(t) for t in sorted(special_tokens, key=len, reverse=True))
    return regex.split(pattern, text)


def count_pretokens(text, special_tokens):
    """Count GPT-2 pre-tokens (as bytes) in text, never spanning special tokens."""
    counts = Counter()
    for piece in _split_on_special_tokens(text, special_tokens):
        for match in _PRETOKEN_RE.finditer(piece):
            counts[match.group().encode("utf-8")] += 1
    return counts


def _init_vocab(special_tokens):
    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    return vocab


def _merge_word(word, pair):
    """Replace every non-overlapping, left-to-right occurrence of `pair` in `word`."""
    first, second = pair
    merged = first + second
    out = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
            out.append(merged)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


def _best_pair(pair_counts):
    # Highest count wins. On a tie, the lexicographically greater pair of bytes wins.
    return max(pair_counts, key=lambda pair: (pair_counts[pair], pair))


def _prepare_special_tokens(special_tokens, vocab_size):
    special_tokens = list(dict.fromkeys(special_tokens or []))  # dedupe, keep order
    vocab = _init_vocab(special_tokens)
    if vocab_size < len(vocab):
        raise ValueError(
            f"vocab_size={vocab_size} is smaller than the initial vocab ({len(vocab)})"
        )
    return special_tokens, vocab


# --------------------------------------------------------- reference trainer
def train_bpe_reference(input_path, vocab_size, special_tokens=None):
    """Clear, slow BPE training. Used as the reference for the optimized trainer.

    Returns:
        vocab:  dict[int, bytes]
        merges: list[tuple[bytes, bytes]] in the order they were learned
    """
    special_tokens, vocab = _prepare_special_tokens(special_tokens, vocab_size)

    # newline="" disables newline translation so bytes match the optimized trainer.
    with open(input_path, "r", encoding="utf-8", newline="") as f:
        text = f.read()

    # word (tuple of single-byte tokens) -> how often it occurs
    words = {
        tuple(bytes([b]) for b in pretoken): count
        for pretoken, count in count_pretokens(text, special_tokens).items()
    }

    merges = []
    while len(vocab) < vocab_size:
        # Recount every adjacent pair from scratch, weighted by word frequency.
        pair_counts = Counter()
        for word, count in words.items():
            for pair in zip(word, word[1:]):
                pair_counts[pair] += count
        if not pair_counts:
            break  # nothing left to merge

        best = _best_pair(pair_counts)
        merges.append(best)
        vocab[len(vocab)] = best[0] + best[1]
        words = {_merge_word(word, best): count for word, count in words.items()}

    return vocab, merges


# ---------------------------------------------------------- optimized trainer
def _find_chunk_boundaries(path, num_chunks, split_token):
    """Byte offsets that cut the file into ~num_chunks pieces at special-token starts.

    Each boundary is pushed forward to the next occurrence of `split_token`, so no
    chunk cuts a document (or a multi-byte character) in half.
    """
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        if num_chunks <= 1 or size == 0 or not split_token:
            return [0, size]

        chunk_size = size // num_chunks
        boundaries = [i * chunk_size for i in range(num_chunks + 1)]
        boundaries[-1] = size

        read_size = 4096
        step = read_size - len(split_token) + 1  # overlap so a token on the edge is found
        for i in range(1, len(boundaries) - 1):
            position = boundaries[i]
            while True:
                f.seek(position)
                block = f.read(read_size)
                if block == b"":  # reached EOF without finding the token
                    boundaries[i] = size
                    break
                found = block.find(split_token)
                if found != -1:
                    boundaries[i] = position + found
                    break
                position += step
    return sorted(set(boundaries))


def _count_pretokens_in_chunk(args):
    """Worker: count pre-tokens in bytes [start, end) of the file."""
    path, start, end, special_tokens = args
    with open(path, "rb") as f:
        f.seek(start)
        data = f.read(end - start)
    return count_pretokens(data.decode("utf-8"), special_tokens)


def _parallel_pretokenize(path, special_tokens, num_processes):
    if num_processes is None:
        num_processes = min(os.cpu_count() or 1, 8)
    split_token = special_tokens[0].encode("utf-8") if special_tokens else b""
    boundaries = _find_chunk_boundaries(path, num_processes, split_token)
    jobs = [
        (path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    if num_processes <= 1 or len(jobs) <= 1:
        results = [_count_pretokens_in_chunk(job) for job in jobs]
    else:
        with multiprocessing.Pool(num_processes) as pool:
            results = pool.map(_count_pretokens_in_chunk, jobs)

    total = Counter()
    for partial in results:
        total.update(partial)
    return total


def train_bpe(input_path, vocab_size, special_tokens=None, num_processes=None):
    """Optimized BPE training. Gives the same output as train_bpe_reference.

    Speedups over the reference:
        1. Pre-token counts are cached (each distinct word is stored once with a
           count) and pre-tokenization runs in parallel over file chunks that
           are split at special-token boundaries.
        2. pair_counts is updated incrementally: after a merge, only the words
           that contained the merged pair are touched. `pair_to_words` maps each
           pair to the set of word indices that (may) contain it.

    pair_to_words may hold stale entries (a word that no longer contains the
    pair). They are harmless: a stale word is detected because merging it does
    not change it.

    Choosing the best pair still scans all current pairs once per merge. That is
    simple and fast enough for vocabularies in the 10k range on small data.

    Returns:
        vocab:  dict[int, bytes]
        merges: list[tuple[bytes, bytes]]
    """
    special_tokens, vocab = _prepare_special_tokens(special_tokens, vocab_size)

    word_counts = _parallel_pretokenize(input_path, special_tokens, num_processes)
    words = [tuple(bytes([b]) for b in pretoken) for pretoken in word_counts]
    freqs = list(word_counts.values())

    pair_counts = Counter()
    pair_to_words = defaultdict(set)
    for index, word in enumerate(words):
        for pair in zip(word, word[1:]):
            pair_counts[pair] += freqs[index]
            pair_to_words[pair].add(index)

    merges = []
    while len(vocab) < vocab_size:
        if not pair_counts:
            break

        best = _best_pair(pair_counts)
        merges.append(best)
        vocab[len(vocab)] = best[0] + best[1]

        # Only words containing `best` change. The new pairs created never equal
        # `best`, so popping its index set before the loop is safe.
        for index in pair_to_words.pop(best, ()):
            old_word = words[index]
            new_word = _merge_word(old_word, best)
            if len(new_word) == len(old_word):
                continue  # stale index entry, word does not contain the pair

            freq = freqs[index]
            for pair in zip(old_word, old_word[1:]):  # remove old contributions
                pair_counts[pair] -= freq
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
            for pair in zip(new_word, new_word[1:]):  # add new contributions
                pair_counts[pair] += freq
                pair_to_words[pair].add(index)
            words[index] = new_word

    return vocab, merges


# ------------------------------------------------------------ serialization
def save_bpe(vocab, merges, vocab_filepath, merges_filepath):
    """Save a vocab and merge list in a simple, lossless text format.

    vocab file:  JSON object {"<id>": "<token bytes as hex>"}
    merges file: one merge per line, "<hex of first> <hex of second>"
    Hex avoids every escaping problem with arbitrary bytes.
    """
    with open(vocab_filepath, "w", encoding="utf-8") as f:
        json.dump({str(i): token.hex() for i, token in vocab.items()}, f)
    with open(merges_filepath, "w", encoding="utf-8") as f:
        for first, second in merges:
            f.write(f"{first.hex()} {second.hex()}\n")


# ---------------------------------------------------------------- Tokenizer
class Tokenizer:
    """Encode text to token ids and decode ids back to text with a trained BPE.

    Encoding a pre-token: start from its single bytes, then repeatedly apply the
    merge with the lowest rank (earliest learned) found in the sequence, until no
    learned merge applies. Results are cached per pre-token.

    Special tokens are matched first, longest first, and map straight to their
    id. If a special token is not in the vocab it is appended with a new id.
    """

    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(dict.fromkeys(special_tokens or []))

        # bytes -> id. If two ids share the same bytes, the lowest id wins.
        self.token_to_id = {}
        for token_id in sorted(self.vocab):
            self.token_to_id.setdefault(self.vocab[token_id], token_id)

        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.token_to_id:
                new_id = max(self.vocab) + 1
                self.vocab[new_id] = token_bytes
                self.token_to_id[token_bytes] = new_id

        self._merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self._special_set = set(self.special_tokens)
        self._special_re = None
        if self.special_tokens:
            # Longest first, so overlapping specials resolve to the longest match.
            ordered = sorted(self.special_tokens, key=len, reverse=True)
            # The capture group makes regex.split keep the special tokens in the output.
            self._special_re = regex.compile("(" + "|".join(regex.escape(t) for t in ordered) + ")")
        self._cache = {}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """Load files written by save_bpe."""
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab = {int(i): bytes.fromhex(h) for i, h in json.load(f).items()}
        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    first, second = line.split(" ")
                    merges.append((bytes.fromhex(first), bytes.fromhex(second)))
        return cls(vocab, merges, special_tokens)

    def _split_special(self, text):
        if self._special_re is None:
            return [text] if text else []
        return [piece for piece in self._special_re.split(text) if piece]

    def _encode_pretoken(self, pretoken):
        """BPE-encode one pre-token (bytes) into a list of ids."""
        cached = self._cache.get(pretoken)
        if cached is not None:
            return cached

        parts = tuple(pretoken[i : i + 1] for i in range(len(pretoken)))
        while len(parts) > 1:
            best_pair, best_rank = None, None
            for pair in zip(parts, parts[1:]):
                rank = self._merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_pair, best_rank = pair, rank
            if best_pair is None:
                break
            parts = _merge_word(parts, best_pair)

        ids = [self.token_to_id[part] for part in parts]
        self._cache[pretoken] = ids
        return ids

    def encode(self, text):
        """text -> list of token ids."""
        ids = []
        for piece in self._split_special(text):
            if piece in self._special_set:
                ids.append(self.token_to_id[piece.encode("utf-8")])
                continue
            for match in _PRETOKEN_RE.finditer(piece):
                ids.extend(self._encode_pretoken(match.group().encode("utf-8")))
        return ids

    def encode_iterable(self, iterable):
        """Lazily encode an iterable of strings (for example an open file).

        Yields token ids one at a time without holding the whole file in memory.
        Each string is encoded on its own, so whitespace runs that cross a line
        boundary (such as a blank line "\\n\\n") are tokenized per line. This is
        the usual convention for file iteration and rarely matters in practice.
        """
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids):
        """list of token ids -> text. Invalid UTF-8 becomes U+FFFD."""
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")
