"""Tokenize a text file into uint16 .npy arrays (and train a BPE tokenizer first).

Writes to --out-dir: vocab.json, merges.txt, train.npy, val.npy.

Examples (run from the repo root):
    # Offline smoke test on a tiny built-in corpus
    python scripts/prepare_data.py --out-dir data/sample

    # Your own text file (documents separated by <|endoftext|>)
    python scripts/prepare_data.py --input my.txt --out-dir data/mine --vocab-size 4000

    # TinyStories (downloads about 2 GB, so cap the tokenized amount)
    python scripts/prepare_data.py --download-tinystories --out-dir data/tinystories \
        --vocab-size 10000 --max-chars 200000000

Tokenizing is pure Python, so use --max-chars to bound the time. The BPE
tokenizer is trained on the (small) validation file for TinyStories, because
training on 2 GB of text in Python takes far too long.
"""

import argparse
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cs336_basics.data import prepare_dataset, prepare_sample_data  # noqa: E402

BASE_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
TINYSTORIES_FILES = {
    "train": "TinyStoriesV2-GPT4-train.txt",
    "valid": "TinyStoriesV2-GPT4-valid.txt",
}


def download(url, dest):
    if os.path.exists(dest):
        print(f"already downloaded: {dest}")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=None, help="training text file")
    parser.add_argument("--val-input", default=None, help="validation text file (default: last 10 percent of train)")
    parser.add_argument("--download-tinystories", action="store_true")
    parser.add_argument("--out-dir", default="data/sample")
    parser.add_argument("--vocab-size", type=int, default=None, help="default: 300 for the sample, 10000 otherwise")
    parser.add_argument("--max-chars", type=int, default=None, help="stop tokenizing after about this many characters")
    parser.add_argument("--num-processes", type=int, default=None, help="workers for BPE pre-tokenization")
    args = parser.parse_args()

    if args.download_tinystories:
        raw_dir = os.path.join(args.out_dir, "raw")
        train_file = os.path.join(raw_dir, TINYSTORIES_FILES["train"])
        valid_file = os.path.join(raw_dir, TINYSTORIES_FILES["valid"])
        download(BASE_URL + TINYSTORIES_FILES["train"], train_file)
        download(BASE_URL + TINYSTORIES_FILES["valid"], valid_file)
        prepare_dataset(
            train_file,
            args.out_dir,
            vocab_size=args.vocab_size or 10000,
            val_input=valid_file,
            tokenizer_train_path=valid_file,
            num_processes=args.num_processes,
            max_chars=args.max_chars,
        )
    elif args.input:
        prepare_dataset(
            args.input,
            args.out_dir,
            vocab_size=args.vocab_size or 10000,
            val_input=args.val_input,
            num_processes=args.num_processes,
            max_chars=args.max_chars,
        )
    else:
        prepare_sample_data(args.out_dir, vocab_size=args.vocab_size or 300)


if __name__ == "__main__":
    main()
