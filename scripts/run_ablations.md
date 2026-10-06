# Running the ablations

Each ablation changes one thing in the model. Train every variant with the same
data, seed, step count and hyperparameters, then compare final validation loss.

All commands run from the repo root. For a quick CPU check add `--smoke`
(tiny model, 50 steps). For a real comparison use the Kaggle setup in
[kaggle_notebook.md](kaggle_notebook.md) and the same flags you use for the baseline.

## Setup

```
python scripts/prepare_data.py --out-dir data/sample          # offline sample data
```

Define the shared flags once (change to taste):

```
COMMON="--data-dir data/sample --max-iters 2000 --warmup-iters 100 --eval-interval 200 --seed 0"
```

On Windows PowerShell, set `$COMMON` the same way, or paste the flags inline.

## The comparisons

| name | flag | what it tests |
|------|------|---------------|
| baseline | none | pre-norm, RMSNorm, RoPE, SwiGLU |
| no RMSNorm | `--no-rmsnorm` | how much normalization helps stability. Often needs a lower learning rate. |
| post-norm | `--post-norm` | norm after the residual add instead of before the sublayer. Usually less stable at high lr and depth. |
| NoPE | `--no-rope` | no positional encoding; the causal mask alone gives some order information |
| SiLU FFN | `--ffn-type silu` | plain two-matrix FFN vs the gated SwiGLU, at equal parameter count |

```
python -m cs336_basics.train $COMMON --ckpt-dir checkpoints/baseline
python -m cs336_basics.train $COMMON --ckpt-dir checkpoints/no_rmsnorm --no-rmsnorm
python -m cs336_basics.train $COMMON --ckpt-dir checkpoints/post_norm --post-norm
python -m cs336_basics.train $COMMON --ckpt-dir checkpoints/nope --no-rope
python -m cs336_basics.train $COMMON --ckpt-dir checkpoints/silu_ffn --ffn-type silu
```

Use a different `--ckpt-dir` for each run so checkpoints and metrics do not mix.

## Comparing results

Each run writes `<ckpt-dir>/metrics.json` with `train_losses` and `val_losses`
(lists of `[step, loss]`) and prints the final validation loss. A small script:

```python
import json
for name in ["baseline", "no_rmsnorm", "post_norm", "nope", "silu_ffn"]:
    m = json.load(open(f"checkpoints/{name}/metrics.json"))
    print(f"{name:12s} final val loss {m['val_losses'][-1][1]:.4f}")
```

## Things to try

- Rerun the no-RMSNorm and post-norm variants with the learning rate divided by
  3 or 10 to see whether the gap was only about stability.
- Run each variant with two seeds (`--seed 0`, `--seed 1`). Differences smaller
  than the seed-to-seed spread are not meaningful.
- With `--no-rope`, generate text from a trained model and look at word order.
