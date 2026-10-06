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
