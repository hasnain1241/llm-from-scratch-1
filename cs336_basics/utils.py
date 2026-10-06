"""Shared helpers: device selection, cross-entropy loss, checkpoint save/load."""

import os

import torch


def get_device(preference="auto"):
    """Pick a torch device.

    "auto" tries cuda, then mps (Apple silicon), then cpu, so the same code runs
    unchanged on a laptop and on a Kaggle GPU. Any other string (for example
    "cpu" or "cuda:1") is used as given.
    """
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cross_entropy(logits, targets):
    """Mean cross-entropy between logits and integer targets.

    Math:
        loss_i = -log softmax(logits_i)[target_i]
               = log(sum_j exp(logits_ij)) - logits_i,target_i

    Stability:
        Subtract the row max m before exponentiating (log-sum-exp trick):
        log(sum exp(l)) = m + log(sum exp(l - m)). The m cancels against the
        same shift applied to the target logit, so the loss is unchanged.

    Shapes:
        logits:  (..., vocab)   e.g. (batch, seq, vocab)
        targets: (...)          integer class ids, e.g. (batch, seq)
        returns: scalar, averaged over every leading position
    """
    logits = logits.to(torch.float32)  # loss math in float32 even under autocast
    logits = logits.reshape(-1, logits.shape[-1])  # (N, vocab)
    targets = targets.reshape(-1)  # (N,)

    shifted = logits - logits.max(dim=-1, keepdim=True).values
    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=-1))  # (N,)
    target_logit = shifted.gather(-1, targets[:, None]).squeeze(-1)  # (N,)
    return (log_sum_exp - target_logit).mean()


def save_checkpoint(model, optimizer, iteration, out, extra=None):
    """Save model weights, optimizer state and the iteration counter.

    The file is written to a temporary name and then renamed, so a crash or a
    Kaggle session timeout in the middle of saving cannot leave a corrupt
    checkpoint behind. `extra` is an optional dict of plain Python values.
    """
    out = os.fspath(out)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    if extra is not None:
        state["extra"] = extra
    tmp_path = out + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, out)


def load_checkpoint(src, model, optimizer=None, map_location=None):
    """Restore a checkpoint in place and return the saved iteration."""
    state = torch.load(os.fspath(src), map_location=map_location)
    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    return state["iteration"]
