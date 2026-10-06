"""Optimization: AdamW, cosine learning-rate schedule, gradient clipping."""

import math

import torch


class AdamW(torch.optim.Optimizer):
    """AdamW: Adam with decoupled weight decay (Loshchilov and Hutter).

    Concept:
        Adam keeps a running average of the gradient (first moment m) and of the
        squared gradient (second moment v). Each parameter is moved by m scaled
        by 1 / sqrt(v), so parameters with consistently large gradients take
        smaller steps. Weight decay is applied directly to the weights instead of
        being added to the gradient, which keeps it independent of the
        adaptive scaling.

    Math (step t, gradient g):
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g^2
        step_size = lr * sqrt(1 - beta2^t) / (1 - beta1^t)       bias correction
        theta = theta - lr * weight_decay * theta                decoupled decay
        theta = theta - step_size * m / (sqrt(v) + eps * sqrt(1 - beta2^t))

    This is algebraically the same as the "m_hat / (sqrt(v_hat) + eps)" form, and
    it follows the same operation order as torch.optim.AdamW so that results
    match closely.

    Shapes: m and v have the same shape as their parameter.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid betas: {betas}")
        if eps < 0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:  # first time we see this parameter
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)  # m
                    state["exp_avg_sq"] = torch.zeros_like(p)  # v

                state["step"] += 1
                t = state["step"]
                m = state["exp_avg"]
                v = state["exp_avg_sq"]

                # Update the moment estimates in place.
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1**t
                bias_correction2 = 1 - beta2**t
                step_size = lr / bias_correction1
                denom = (v.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                # Decoupled weight decay first, then the Adam step.
                p.mul_(1 - lr * weight_decay)
                p.addcdiv_(m, denom, value=-step_size)

        return loss


def get_lr_cosine_schedule(it, max_lr, min_lr, warmup_iters, cosine_cycle_iters):
    """Learning rate at iteration `it`: linear warmup, cosine decay, then flat.

    Phases:
        it < warmup_iters:                  lr = it / warmup_iters * max_lr
        warmup_iters <= it <= cycle_iters:  lr = min_lr + 0.5 (1 + cos(pi * progress)) (max_lr - min_lr)
                                            with progress = (it - warmup) / (cycle - warmup)
        it > cycle_iters:                   lr = min_lr

    Warmup avoids huge early updates while Adam's moment estimates are noisy.
    Decay lets the model settle into a minimum at the end of training.
    """
    if it < warmup_iters:
        return it / warmup_iters * max_lr
    if it > cosine_cycle_iters:
        return min_lr

    decay_span = cosine_cycle_iters - warmup_iters
    if decay_span <= 0:  # no room for a cosine phase
        return min_lr
    progress = (it - warmup_iters) / decay_span  # goes 0 -> 1
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))  # goes 1 -> 0
    return min_lr + cosine * (max_lr - min_lr)


def clip_gradients(parameters, max_l2_norm, eps=1e-6):
    """Clip the global L2 norm of all gradients to at most `max_l2_norm`, in place.

    Math:
        total_norm = sqrt(sum over all params of sum(grad^2))
        if total_norm > max: grad <- grad * max / (total_norm + eps)

    One global norm (not one per tensor) keeps the update direction unchanged
    and only shrinks its length. eps avoids division by zero. Returns the norm
    measured before clipping, which is handy to log.
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)

    total_norm = torch.sqrt(sum(g.pow(2).sum() for g in grads))
    clip_coef = max_l2_norm / (total_norm + eps)
    if clip_coef < 1.0:  # only ever shrink, never grow
        for g in grads:
            g.mul_(clip_coef)
    return total_norm
