# LLM From Scratch, Part 1

A small decoder-only Transformer language model stack written from scratch in PyTorch, following the structure of Stanford CS336 Assignment 1. It is a self-study project: the code favors clarity and explanation over cleverness. See [docs/NOTES.md](docs/NOTES.md) for study notes and design decisions.

No `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, `nn.MultiheadAttention` or `torch.optim.AdamW` are used in the library code. Torch reference ops appear only in tests, as correctness oracles.

## Setup

Requires Python 3.11+.

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The device is auto-detected (`cuda`, then `mps`, then `cpu`), so the same code runs on CPU locally and on a Kaggle GPU.

## Run the tests

```
pytest
```

## Quick start

Smoke test on CPU (trains a tiny model for 50 steps on a built-in corpus, shows the
loss going down, and prints a sample):

```
python -m cs336_basics.train --smoke
```

Prepare your own data and train (see [scripts/prepare_data.py](scripts/prepare_data.py)):

```
python scripts/prepare_data.py --input my.txt --out-dir data/mine --vocab-size 4000
python -m cs336_basics.train --data-dir data/mine --ckpt-dir checkpoints/mine --resume auto
```

Other guides:

- [scripts/kaggle_notebook.md](scripts/kaggle_notebook.md): train on a Kaggle GPU, resume after a timeout
- [scripts/run_ablations.md](scripts/run_ablations.md): no RMSNorm, post-norm, NoPE, SiLU FFN
- [scripts/time_bpe.py](scripts/time_bpe.py): time the BPE trainers

## Layout

```
cs336_basics/
  layers.py      Linear, Embedding, RMSNorm, softmax, SwiGLU, RoPE
  attention.py   scaled dot-product attention, causal multi-head attention
  model.py       TransformerBlock, TransformerLM, generate
  optim.py       AdamW, cosine lr schedule, gradient clipping
  data.py        get_batch, memmap loading, data preparation
  tokenizer.py   BPE training (reference and fast) and the Tokenizer class
  train.py       training loop and CLI
  utils.py       cross_entropy, checkpoints, device helper
tests/           one test file per area, plus test_utils.py and test_train.py
docs/NOTES.md    study notes, design decisions, worked examples
```
