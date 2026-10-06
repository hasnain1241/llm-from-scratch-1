"""Time the reference and optimized BPE trainers on a synthetic corpus.

Usage (from the repo root):
    python scripts/time_bpe.py
    python scripts/time_bpe.py --num-docs 3000 --vocab-size 600 --processes 4
    python scripts/time_bpe.py --input path/to/text.txt   # use your own file

It also checks that both trainers return identical vocab and merges.
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cs336_basics.data import build_sample_corpus  # noqa: E402
from cs336_basics.tokenizer import train_bpe, train_bpe_reference  # noqa: E402


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=None, help="text file (default: synthetic corpus)")
    parser.add_argument("--num-docs", type=int, default=1500)
    parser.add_argument("--vocab-size", type=int, default=500)
    parser.add_argument("--processes", type=int, default=4)
    args = parser.parse_args()
    special = ["<|endoftext|>"]

    with tempfile.TemporaryDirectory() as tmp:
        path = args.input
        if path is None:
            path = os.path.join(tmp, "corpus.txt")
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(build_sample_corpus(args.num_docs))
        print(f"corpus: {os.path.getsize(path) / 1e6:.2f} MB, vocab_size={args.vocab_size}")

        ref, t_ref = timed(train_bpe_reference, path, args.vocab_size, special)
        fast1, t_fast1 = timed(train_bpe, path, args.vocab_size, special, num_processes=1)
        fastn, t_fastn = timed(train_bpe, path, args.vocab_size, special, num_processes=args.processes)

    print(f"reference:               {t_ref:8.2f} s")
    print(f"optimized, 1 process:    {t_fast1:8.2f} s  ({t_ref / t_fast1:.1f}x faster)")
    print(f"optimized, {args.processes} processes:   {t_fastn:8.2f} s  ({t_ref / t_fastn:.1f}x faster)")
    print("outputs identical:", ref == fast1 == fastn)


if __name__ == "__main__":
    main()
