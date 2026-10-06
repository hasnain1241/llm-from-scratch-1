"""Training loop with validation, checkpointing, resume, mixed precision and a CLI.

Run (from the repo root):
    python -m cs336_basics.train --smoke                  # tiny CPU run, about a minute
    python -m cs336_basics.train --data-dir data/tinystories --ckpt-dir checkpoints/run1

Each step:
    1. set the learning rate from the cosine schedule
    2. sample a batch (x, y), y = x shifted by one token
    3. forward pass, cross-entropy loss
    4. backward pass, clip the global gradient norm, AdamW step

Resuming: checkpoints are written to <ckpt-dir>/last.pt (atomically). Pass
`--resume auto` to continue from it if it exists, or `--resume path/to/file.pt`.
This is what makes training safe on Kaggle, where sessions time out. The data
sampler is reseeded on resume, so batches differ from an uninterrupted run, but
weights, optimizer state and step count are restored exactly.
"""

import argparse
import contextlib
import json
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from cs336_basics.data import get_batch, load_dataset, prepare_sample_data
from cs336_basics.model import TransformerLM, generate
from cs336_basics.optim import AdamW, clip_gradients, get_lr_cosine_schedule
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.utils import cross_entropy, get_device, load_checkpoint, save_checkpoint

EOT = "<|endoftext|>"

# Applied with --smoke unless overridden explicitly on the command line.
SMOKE_DEFAULTS = dict(
    data_dir="data/smoke",
    ckpt_dir="checkpoints/smoke",
    context_length=64,
    num_layers=2,
    d_model=64,
    num_heads=4,
    batch_size=16,
    max_lr=3e-3,
    min_lr=3e-4,
    warmup_iters=5,
    max_iters=50,
    eval_interval=25,
    eval_batches=5,
    log_interval=10,
    ckpt_interval=25,
    weight_decay=0.01,
)


def build_parser():
    p = argparse.ArgumentParser(description="Train a small Transformer LM from scratch.")
    p.add_argument("--smoke", action="store_true", help="tiny CPU config on a built-in corpus")

    g = p.add_argument_group("data")
    g.add_argument("--data-dir", default="data/tinystories",
                   help="folder with train.npy, val.npy and (optionally) vocab.json, merges.txt")
    g.add_argument("--vocab-size", type=int, default=None,
                   help="needed only if the data folder has no tokenizer files")

    g = p.add_argument_group("model")
    g.add_argument("--context-length", type=int, default=256)
    g.add_argument("--num-layers", type=int, default=4)
    g.add_argument("--d-model", type=int, default=256)
    g.add_argument("--num-heads", type=int, default=4)
    g.add_argument("--d-ff", type=int, default=None, help="default: about 8/3 d_model (4 d_model for silu)")
    g.add_argument("--rope-theta", type=float, default=10000.0)
    g.add_argument("--no-rmsnorm", action="store_true", help="ablation: remove all normalization")
    g.add_argument("--post-norm", action="store_true", help="ablation: post-norm instead of pre-norm")
    g.add_argument("--no-rope", action="store_true", help="ablation: no positional encoding (NoPE)")
    g.add_argument("--ffn-type", choices=["swiglu", "silu"], default="swiglu",
                   help="ablation: silu uses a plain SiLU FFN")

    g = p.add_argument_group("optimization")
    g.add_argument("--batch-size", type=int, default=32)
    g.add_argument("--max-lr", type=float, default=1e-3)
    g.add_argument("--min-lr", type=float, default=1e-4)
    g.add_argument("--warmup-iters", type=int, default=200)
    g.add_argument("--max-iters", type=int, default=5000, help="total steps; cosine decays over this")
    g.add_argument("--weight-decay", type=float, default=0.1)
    g.add_argument("--beta1", type=float, default=0.9)
    g.add_argument("--beta2", type=float, default=0.95)
    g.add_argument("--grad-clip", type=float, default=1.0)

    g = p.add_argument_group("logging, evaluation, checkpoints")
    g.add_argument("--eval-interval", type=int, default=250)
    g.add_argument("--eval-batches", type=int, default=20)
    g.add_argument("--log-interval", type=int, default=50)
    g.add_argument("--ckpt-interval", type=int, default=500)
    g.add_argument("--ckpt-dir", default="checkpoints/run")
    g.add_argument("--resume", default=None, help="'auto' or a checkpoint path")
    g.add_argument("--no-progress", action="store_true", help="disable the progress bar")
    g.add_argument("--wandb", action="store_true", help="log to Weights and Biases (off by default)")
    g.add_argument("--wandb-project", default="llm-from-scratch-part-1")

    g = p.add_argument_group("system")
    g.add_argument("--device", default="auto", help="auto, cpu, cuda, mps")
    g.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32",
                   help="autocast precision. float16 uses a GradScaler (T4/P100), bfloat16 needs Ampere+")
    g.add_argument("--seed", type=int, default=0)
    return p


def parse_args(argv=None):
    parser = build_parser()
    preliminary, _ = parser.parse_known_args(argv)
    if preliminary.smoke:
        parser.set_defaults(**SMOKE_DEFAULTS)  # explicit flags still win
    return parser.parse_args(argv)


def amp_context(device, amp_dtype):
    """Autocast context for mixed precision, or a no-op for float32."""
    if amp_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


@torch.no_grad()
def estimate_loss(model, dataset, args, device, rng, amp_dtype):
    """Mean loss over a few random batches, with the model in eval mode."""
    model.eval()
    losses = []
    for _ in range(args.eval_batches):
        x, y = get_batch(dataset, args.batch_size, args.context_length, device, rng)
        with amp_context(device, amp_dtype):
            losses.append(cross_entropy(model(x), y).item())
    model.train()
    return float(np.mean(losses))


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = get_device(args.device)

    # ---- data
    train_path = os.path.join(args.data_dir, "train.npy")
    val_path = os.path.join(args.data_dir, "val.npy")
    if args.smoke and not os.path.exists(train_path):
        print("Preparing the built-in sample corpus ...")
        prepare_sample_data(args.data_dir, vocab_size=300)
    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        raise SystemExit(
            f"Missing {train_path} or {val_path}. Create them with scripts/prepare_data.py."
        )
    train_data = load_dataset(train_path)
    val_data = load_dataset(val_path)

    tokenizer = None
    vocab_file = os.path.join(args.data_dir, "vocab.json")
    merges_file = os.path.join(args.data_dir, "merges.txt")
    if os.path.exists(vocab_file) and os.path.exists(merges_file):
        tokenizer = Tokenizer.from_files(vocab_file, merges_file, [EOT])
        vocab_size = len(tokenizer.vocab)
    elif args.vocab_size is not None:
        vocab_size = args.vocab_size
    else:
        raise SystemExit("No tokenizer files in the data dir, so pass --vocab-size.")

    # ---- model and optimizer
    model_kwargs = dict(
        vocab_size=vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        use_rmsnorm=not args.no_rmsnorm,
        post_norm=args.post_norm,
        use_rope=not args.no_rope,
        ffn_type=args.ffn_type,
    )
    model = TransformerLM(**model_kwargs).to(device)
    print(
        f"device={device}  params={model.num_parameters():,} "
        f"(non-embedding {model.num_parameters(non_embedding=True):,})"
    )

    # Weight decay on matrices only; norm gains (1D) are not decayed.
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    groups = [{"params": decay, "weight_decay": args.weight_decay}]
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    optimizer = AdamW(groups, lr=args.max_lr, betas=(args.beta1, args.beta2))

    # ---- mixed precision
    amp_dtype = {"float32": None, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    if amp_dtype is torch.float16 and device.type != "cuda":
        raise SystemExit("--dtype float16 needs a CUDA device. Use bfloat16 or float32 here.")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype is torch.float16)

    # ---- resume
    os.makedirs(args.ckpt_dir, exist_ok=True)
    last_path = os.path.join(args.ckpt_dir, "last.pt")
    metrics_path = os.path.join(args.ckpt_dir, "metrics.json")
    resume_path = None
    if args.resume == "auto":
        resume_path = last_path if os.path.exists(last_path) else None
    elif args.resume:
        resume_path = args.resume

    start_iter = 0
    metrics = {"train_losses": [], "val_losses": []}
    if resume_path is not None:
        start_iter = load_checkpoint(resume_path, model, optimizer, map_location=device)
        print(f"Resumed from {resume_path} at step {start_iter}")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
    elif args.resume == "auto":
        print("No checkpoint found, starting from scratch")

    with open(os.path.join(args.ckpt_dir, "config.json"), "w") as f:
        json.dump({"model_kwargs": model_kwargs, "args": vars(args)}, f, indent=2)

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(project=args.wandb_project, config=vars(args))

    # ---- training loop
    model.train()
    progress = tqdm(total=args.max_iters, initial=start_iter, disable=args.no_progress, desc="train")
    start_time = time.time()

    for it in range(start_iter, args.max_iters):
        lr = get_lr_cosine_schedule(it, args.max_lr, args.min_lr, args.warmup_iters, args.max_iters)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_data, args.batch_size, args.context_length, device, rng)

        optimizer.zero_grad(set_to_none=True)
        with amp_context(device, amp_dtype):
            loss = cross_entropy(model(x), y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)  # gradients back to true scale before clipping
        grad_norm = clip_gradients(model.parameters(), args.grad_clip)
        scaler.step(optimizer)  # skipped automatically if fp16 gradients overflowed
        scaler.update()

        step = it + 1
        progress.update(1)

        if step % args.log_interval == 0 or step == start_iter + 1 or step == args.max_iters:
            loss_value = loss.item()  # .item() syncs with the GPU, so only do it when logging
            metrics["train_losses"].append([step, loss_value])
            progress.set_postfix(loss=f"{loss_value:.3f}", lr=f"{lr:.1e}")
            if wandb_run is not None:
                wandb_run.log({"train/loss": loss_value, "lr": lr, "grad_norm": float(grad_norm)}, step=step)

        if step % args.eval_interval == 0 or step == args.max_iters:
            val_loss = estimate_loss(model, val_data, args, device, rng, amp_dtype)
            metrics["val_losses"].append([step, val_loss])
            progress.write(f"step {step}: val loss {val_loss:.4f}")
            if wandb_run is not None:
                wandb_run.log({"val/loss": val_loss}, step=step)

        if step % args.ckpt_interval == 0 or step == args.max_iters:
            save_checkpoint(model, optimizer, step, last_path, extra={"model_kwargs": model_kwargs})
            with open(metrics_path, "w") as f:
                json.dump(metrics, f)

    progress.close()
    elapsed = time.time() - start_time
    print(f"Finished at step {max(start_iter, args.max_iters)} in {elapsed:.1f}s")

    train_losses = metrics["train_losses"]
    if len(train_losses) >= 2:
        first, last = train_losses[0][1], train_losses[-1][1]
        print(f"train loss {first:.3f} -> {last:.3f} ({'decreased' if last < first else 'did NOT decrease'})")
    if metrics["val_losses"]:
        print(f"final val loss {metrics['val_losses'][-1][1]:.4f}")

    if tokenizer is not None and args.max_iters > start_iter:
        sample = generate(model, tokenizer, "", max_new_tokens=60, temperature=0.8, top_p=0.9)
        print("sample:", sample)

    if wandb_run is not None:
        wandb_run.finish()

    return {
        "start_iteration": start_iter,
        "final_iteration": max(start_iter, args.max_iters),
        "train_losses": metrics["train_losses"],
        "val_losses": metrics["val_losses"],
        "checkpoint": last_path,
    }


if __name__ == "__main__":
    main()
