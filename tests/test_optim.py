"""Tests for cs336_basics.optim. Official torch optimizers are used only as oracles."""

import math

import pytest
import torch

from cs336_basics.optim import AdamW, clip_gradients, get_lr_cosine_schedule


# ----------------------------------------------------------------- AdamW
def _make_problem():
    gen = torch.Generator().manual_seed(0)
    w0 = torch.randn(5, 3, dtype=torch.float64, generator=gen)
    b0 = torch.randn(5, dtype=torch.float64, generator=gen)
    x = torch.randn(20, 3, dtype=torch.float64, generator=gen)
    y = torch.randn(20, 5, dtype=torch.float64, generator=gen)
    return w0, b0, x, y


def _run(opt_factory, steps=30):
    w0, b0, x, y = _make_problem()
    w = torch.nn.Parameter(w0.clone())
    b = torch.nn.Parameter(b0.clone())
    opt = opt_factory([w, b])
    for _ in range(steps):
        opt.zero_grad()
        loss = ((x @ w.T + b - y) ** 2).mean()
        loss.backward()
        opt.step()
    return w.detach(), b.detach()


@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
def test_adamw_matches_torch(weight_decay):
    kwargs = dict(lr=1e-2, betas=(0.9, 0.99), eps=1e-8, weight_decay=weight_decay)
    mine = _run(lambda p: AdamW(p, **kwargs))
    ref = _run(lambda p: torch.optim.AdamW(p, **kwargs))
    for a, b in zip(mine, ref):
        torch.testing.assert_close(a, b, rtol=1e-7, atol=1e-9)


def test_adamw_reduces_loss():
    w0, b0, x, y = _make_problem()
    w = torch.nn.Parameter(w0.clone())
    opt = AdamW([w], lr=5e-2)
    first = None
    for _ in range(50):
        opt.zero_grad()
        loss = ((x @ w.T - y) ** 2).mean()
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < first


def test_adamw_decay_is_decoupled_from_gradient():
    # With zero gradient the Adam step is zero, but decay still shrinks the weight.
    w = torch.nn.Parameter(torch.ones(3, dtype=torch.float64))
    opt = AdamW([w], lr=0.1, weight_decay=0.5)
    w.grad = torch.zeros_like(w)
    opt.step()
    expected = 1.0 * (1 - 0.1 * 0.5)
    torch.testing.assert_close(w.detach(), torch.full((3,), expected, dtype=torch.float64))


def test_adamw_skips_params_without_grad():
    a = torch.nn.Parameter(torch.ones(2))
    b = torch.nn.Parameter(torch.ones(2))
    opt = AdamW([a, b], lr=0.1)
    a.grad = torch.ones(2)
    opt.step()
    assert torch.all(b == 1)
    assert not torch.all(a == 1)


def test_adamw_supports_closure_and_param_groups():
    a = torch.nn.Parameter(torch.ones(2))
    b = torch.nn.Parameter(torch.ones(2))
    opt = AdamW([{"params": [a], "weight_decay": 0.0}, {"params": [b], "weight_decay": 0.5}], lr=0.1)

    def closure():
        opt.zero_grad()
        loss = (a.sum() + b.sum()) * 0.0
        loss.backward()
        return loss

    loss = opt.step(closure)
    assert loss is not None
    torch.testing.assert_close(a.detach(), torch.ones(2))  # no decay, zero grad
    assert torch.all(b < 1)  # decayed


def test_adamw_rejects_bad_hyperparameters():
    p = [torch.nn.Parameter(torch.ones(1))]
    with pytest.raises(ValueError):
        AdamW(p, lr=-1.0)
    with pytest.raises(ValueError):
        AdamW(p, betas=(1.0, 0.9))


# ------------------------------------------------------------ lr schedule
SCHED = dict(max_lr=1.0, min_lr=0.1, warmup_iters=10, cosine_cycle_iters=110)


def test_schedule_starts_at_zero_and_warms_up_linearly():
    assert get_lr_cosine_schedule(0, **SCHED) == 0.0
    assert get_lr_cosine_schedule(5, **SCHED) == pytest.approx(0.5)


def test_schedule_peaks_at_end_of_warmup():
    assert get_lr_cosine_schedule(10, **SCHED) == pytest.approx(1.0)


def test_schedule_midpoint_of_cosine_is_average():
    mid = get_lr_cosine_schedule(60, **SCHED)  # halfway between 10 and 110
    assert mid == pytest.approx((1.0 + 0.1) / 2)


def test_schedule_reaches_min_at_cycle_end_and_stays():
    assert get_lr_cosine_schedule(110, **SCHED) == pytest.approx(0.1)
    assert get_lr_cosine_schedule(111, **SCHED) == pytest.approx(0.1)
    assert get_lr_cosine_schedule(10_000, **SCHED) == pytest.approx(0.1)


def test_schedule_decays_monotonically_after_warmup():
    lrs = [get_lr_cosine_schedule(i, **SCHED) for i in range(10, 111)]
    assert all(a >= b - 1e-12 for a, b in zip(lrs, lrs[1:]))


def test_schedule_matches_formula():
    it = 37
    progress = (it - 10) / 100
    expected = 0.1 + 0.5 * (1 + math.cos(math.pi * progress)) * 0.9
    assert get_lr_cosine_schedule(it, **SCHED) == pytest.approx(expected)


def test_schedule_zero_warmup_does_not_divide_by_zero():
    assert get_lr_cosine_schedule(0, 1.0, 0.1, 0, 100) == pytest.approx(1.0)


# --------------------------------------------------------------- clipping
def _params_with_grads(scale):
    gen = torch.Generator().manual_seed(1)
    params = [torch.nn.Parameter(torch.randn(4, 3, generator=gen)), torch.nn.Parameter(torch.randn(7, generator=gen))]
    for p in params:
        p.grad = torch.randn(p.shape, generator=gen) * scale
    return params


@pytest.mark.parametrize("scale", [0.01, 10.0])  # below and above the threshold
def test_clip_matches_torch(scale):
    mine = _params_with_grads(scale)
    ref = _params_with_grads(scale)
    clip_gradients(mine, max_l2_norm=1.0)
    torch.nn.utils.clip_grad_norm_(ref, max_norm=1.0)
    for a, b in zip(mine, ref):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=1e-7)


def test_clip_result_has_norm_at_most_max():
    params = _params_with_grads(10.0)
    clip_gradients(params, max_l2_norm=1.0)
    total = torch.sqrt(sum(p.grad.pow(2).sum() for p in params))
    assert total.item() <= 1.0 + 1e-5


def test_clip_returns_norm_before_clipping():
    params = _params_with_grads(10.0)
    expected = torch.sqrt(sum(p.grad.pow(2).sum() for p in params))
    returned = clip_gradients(params, max_l2_norm=1.0)
    torch.testing.assert_close(returned, expected)


def test_clip_leaves_small_gradients_untouched():
    params = _params_with_grads(0.001)
    before = [p.grad.clone() for p in params]
    clip_gradients(params, max_l2_norm=100.0)
    for p, b in zip(params, before):
        torch.testing.assert_close(p.grad, b)


def test_clip_handles_none_grads_and_generators():
    a = torch.nn.Parameter(torch.ones(3))
    b = torch.nn.Parameter(torch.ones(3))
    a.grad = torch.full((3,), 10.0)
    clip_gradients((p for p in [a, b]), max_l2_norm=1.0)  # generator input
    assert b.grad is None
    assert a.grad.norm().item() <= 1.0 + 1e-5
