"""Tests for cs336_basics.utils (device, cross entropy, checkpoints)."""

import pytest
import torch
import torch.nn.functional as F

from cs336_basics.model import TransformerLM
from cs336_basics.optim import AdamW
from cs336_basics.utils import cross_entropy, get_device, load_checkpoint, save_checkpoint


# ------------------------------------------------------------------ device
def test_get_device_auto_returns_a_torch_device():
    device = get_device("auto")
    assert isinstance(device, torch.device)
    assert device.type in ("cuda", "mps", "cpu")


def test_get_device_respects_explicit_choice():
    assert get_device("cpu") == torch.device("cpu")


# ----------------------------------------------------------- cross entropy
def test_cross_entropy_matches_torch_2d():
    logits = torch.randn(10, 7)
    targets = torch.randint(0, 7, (10,))
    torch.testing.assert_close(cross_entropy(logits, targets), F.cross_entropy(logits, targets))


def test_cross_entropy_matches_torch_batched_sequences():
    logits = torch.randn(3, 5, 11)
    targets = torch.randint(0, 11, (3, 5))
    expected = F.cross_entropy(logits.reshape(-1, 11), targets.reshape(-1))
    torch.testing.assert_close(cross_entropy(logits, targets), expected)


def test_cross_entropy_is_stable_for_huge_logits():
    logits = torch.tensor([[1000.0, 0.0, -1000.0], [-5000.0, 5000.0, 0.0]])
    targets = torch.tensor([0, 1])
    loss = cross_entropy(logits, targets)
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, F.cross_entropy(logits, targets))


def test_cross_entropy_uniform_logits_gives_log_vocab():
    loss = cross_entropy(torch.zeros(4, 16), torch.zeros(4, dtype=torch.long))
    assert loss.item() == pytest.approx(torch.log(torch.tensor(16.0)).item(), rel=1e-6)


def test_cross_entropy_is_a_scalar_and_gradients_match():
    logits = torch.randn(6, 9, requires_grad=True)
    logits2 = logits.detach().clone().requires_grad_(True)
    targets = torch.randint(0, 9, (6,))
    loss = cross_entropy(logits, targets)
    assert loss.ndim == 0
    loss.backward()
    F.cross_entropy(logits2, targets).backward()
    torch.testing.assert_close(logits.grad, logits2.grad)


def test_cross_entropy_low_precision_input_returns_float32():
    logits = torch.randn(4, 8).to(torch.bfloat16)
    targets = torch.randint(0, 8, (4,))
    assert cross_entropy(logits, targets).dtype == torch.float32


# ------------------------------------------------------------- checkpoints
def _tiny_model(seed):
    torch.manual_seed(seed)
    return TransformerLM(
        vocab_size=30, context_length=8, num_layers=1, d_model=16, num_heads=2, d_ff=32
    )


def _train_steps(model, opt, x, y, steps):
    for _ in range(steps):
        opt.zero_grad()
        cross_entropy(model(x), y).backward()
        opt.step()


def test_checkpoint_roundtrip_restores_model_optimizer_and_iteration(tmp_path):
    x = torch.randint(0, 30, (4, 8))
    y = torch.randint(0, 30, (4, 8))

    model = _tiny_model(0)
    opt = AdamW(model.parameters(), lr=1e-2)
    _train_steps(model, opt, x, y, 3)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(model, opt, iteration=3, out=path)
    assert not (tmp_path / "ckpt.pt.tmp").exists()

    # A fresh model with different init and a fresh optimizer.
    model2 = _tiny_model(1)
    opt2 = AdamW(model2.parameters(), lr=1e-2)
    iteration = load_checkpoint(path, model2, opt2)
    assert iteration == 3

    for (n1, p1), (n2, p2) in zip(model.state_dict().items(), model2.state_dict().items()):
        assert n1 == n2
        torch.testing.assert_close(p1, p2)

    for p1, p2 in zip(model.parameters(), model2.parameters()):
        s1, s2 = opt.state[p1], opt2.state[p2]
        assert s1["step"] == s2["step"]
        torch.testing.assert_close(s1["exp_avg"], s2["exp_avg"])
        torch.testing.assert_close(s1["exp_avg_sq"], s2["exp_avg_sq"])

    # Training can continue identically from the restored state.
    _train_steps(model, opt, x, y, 2)
    _train_steps(model2, opt2, x, y, 2)
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        torch.testing.assert_close(p1, p2)


def test_checkpoint_load_without_optimizer(tmp_path):
    model = _tiny_model(0)
    opt = AdamW(model.parameters())
    path = tmp_path / "c.pt"
    save_checkpoint(model, opt, 7, path)
    other = _tiny_model(5)
    assert load_checkpoint(path, other) == 7
    for p1, p2 in zip(model.parameters(), other.parameters()):
        torch.testing.assert_close(p1, p2)


def test_checkpoint_extra_is_saved(tmp_path):
    model = _tiny_model(0)
    opt = AdamW(model.parameters())
    path = tmp_path / "c.pt"
    save_checkpoint(model, opt, 1, path, extra={"note": "hello"})
    assert torch.load(path)["extra"] == {"note": "hello"}
