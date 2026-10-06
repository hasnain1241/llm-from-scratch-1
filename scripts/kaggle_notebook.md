# Training on Kaggle (T4 or P100 GPU)

Paste each block into its own notebook cell, in order. Before starting, in the
notebook sidebar set **Accelerator** to a GPU (T4 or P100) and turn **Internet**
on (needed for `git clone` and the data download).

## Session limits

- A session lasts up to roughly 12 hours, and idle or disconnected sessions are
  shut down earlier. There is also a weekly GPU quota (about 30 hours at the time
  of writing). Kaggle changes these numbers, so check the current limits in your account.
- Files in `/kaggle/working` are kept when a notebook version is saved and run
  ("Save Version"), but an interactive session that gets reset can lose them.
  Save checkpoints out of the session regularly (see cell 7).
- Training here checkpoints every `--ckpt-interval` steps and resumes with
  `--resume auto`, so a timeout costs you at most one interval.

## Cells

**1. Check the GPU**

```python
!nvidia-smi
```

**2. Get the code.** Use your own repo URL (the example assumes the name `llm-from-scratch-part-1`).

```python
!git clone https://github.com/hasnain1241/llm-from-scratch-part-1.git
%cd llm-from-scratch-part-1
```

If the clone already exists after a restart, run `%cd /kaggle/working/llm-from-scratch-part-1`
and `!git pull` instead.

**3. Install dependencies.** Kaggle already ships PyTorch, so install only the rest
(do not reinstall torch; that wastes time and can break the CUDA build).

```python
!pip install -q einops regex tqdm
```

`uv` also works (`!pip install -q uv && uv pip install --system einops regex tqdm`)
but plain `pip` is enough here.

**4. Prepare data.** This downloads TinyStories, trains a 10k-vocab BPE tokenizer on
the validation file, and tokenizes up to 200M characters of the training file.
Tokenizing is pure Python, so it takes a while; lower `--max-chars` to go faster.

```python
!python scripts/prepare_data.py --download-tinystories \
    --out-dir /kaggle/working/data/tinystories \
    --vocab-size 10000 --max-chars 200000000
```

**5. Quick sanity run** (about a minute, confirms the GPU path works).

```python
!python -m cs336_basics.train --smoke --device cuda --dtype float16
```

**6. Train.**

```python
!python -m cs336_basics.train \
    --data-dir /kaggle/working/data/tinystories \
    --ckpt-dir /kaggle/working/checkpoints/run1 \
    --context-length 256 --num-layers 6 --d-model 384 --num-heads 6 \
    --batch-size 32 --max-iters 10000 --warmup-iters 300 \
    --max-lr 1e-3 --min-lr 1e-4 \
    --eval-interval 500 --ckpt-interval 500 \
    --dtype float16 \
    --resume auto
```

`--resume auto` starts fresh the first time and continues from `last.pt` afterwards,
so the same cell works for the first run and for every restart.

**7. Save checkpoints out of the session** (run now and then, and before the session ends).

```python
!cp /kaggle/working/checkpoints/run1/last.pt /kaggle/working/last_backup.pt
```

To keep it across sessions, create a Kaggle Dataset from the file (Output tab, "New
Dataset"), then attach it to the next session as an input.

## Resuming after a session restart

Repeat cells 2 and 3. If `/kaggle/working/checkpoints/run1/last.pt` and the data
folder survived, run cell 6 unchanged. If they did not, restore from the dataset
you saved, then rerun cell 4 (the data is not in the checkpoint) and train with an
explicit path:

```python
!mkdir -p /kaggle/working/checkpoints/run1
!cp /kaggle/input/YOUR_DATASET/last.pt /kaggle/working/checkpoints/run1/last.pt
```

Then run cell 6 (still with `--resume auto`). Keep `--max-iters` and the other
schedule flags the same as in the interrupted run so the learning-rate curve continues.

## What to tune for a T4 or P100

Both have 16 GB of memory.

- **Precision.** On a T4 use `--dtype float16` (tensor cores, with a GradScaler
  handled for you). `bfloat16` needs Ampere or newer, so it is slow or unsupported
  on T4 and P100. The P100 has no fp16 tensor cores, so it gains little from
  `float16`; try `float32` and compare speed.
- **Batch size.** Raise `--batch-size` until you run out of memory, then step back.
  A CUDA out-of-memory error means halve it. Larger batches use the GPU better.
- **Context length.** Memory for attention grows with the square of
  `--context-length` (this code builds the full score matrix). 256 is a good start.
  Doubling it costs about 4x the attention memory.
- **Model size.** `--d-model 384 --num-layers 6` is a sensible size for a T4. Check the
  printed parameter count, and use the FLOPs worked example in `docs/NOTES.md` to
  estimate step time.
- **Learning rate.** Small models tolerate `--max-lr 1e-3`. If the loss spikes or
  turns NaN, lower it to `3e-4`.
- **Steps.** Total tokens seen = `batch_size * context_length * max_iters`. Aim for
  tens of millions of tokens or more for readable TinyStories samples.
- **Speed.** The progress bar shows iterations per second. If it is much slower than
  expected, confirm `device=cuda` in the first printed line.
