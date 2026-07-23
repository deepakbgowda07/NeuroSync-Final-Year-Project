"""
ranger.py
==========
Ranger optimizer: RAdam (Rectified Adam, Liu et al. 2019) combined with
Lookahead (Zhang, Lucas, Hinton & Ba, 2019), following Wright's original
"Ranger" formulation. Implemented from scratch here (rather than
depending on an external `ranger` package) since it is not part of
`torch.optim` and this project's requirements.txt only depends on
official PyPI/PyTorch packages.

Usage matches any standard torch.optim.Optimizer:
    optimizer = Ranger(model.parameters(), lr=1e-3)
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

try:
    import torch
    from torch.optim.optimizer import Optimizer
except ImportError:  # pragma: no cover - allow import-time introspection without torch
    torch = None
    Optimizer = object  # type: ignore


class Ranger(Optimizer if torch is not None else object):
    """RAdam + Lookahead optimizer.

    Args:
        params: iterable of parameters to optimize.
        lr: base learning rate for the inner RAdam step.
        betas: RAdam's (beta1, beta2) coefficients.
        eps: RAdam numerical-stability epsilon.
        weight_decay: L2 weight decay coefficient.
        k: Lookahead synchronization period (steps between slow-weight updates).
        alpha: Lookahead slow-weight interpolation factor.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.95, 0.999),
        eps: float = 1e-5,
        weight_decay: float = 0.0,
        k: int = 6,
        alpha: float = 0.5,
    ):
        if torch is None:
            raise ImportError("PyTorch is required to instantiate Ranger.")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        defaults: Dict = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, k=k, alpha=alpha)
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
            k = group["k"]
            alpha = group["alpha"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Ranger does not support sparse gradients.")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["slow_buffer"] = p.detach().clone()

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                # --- RAdam rectification term ---
                rho_inf = 2 / (1 - beta2) - 1
                rho_t = rho_inf - 2 * step * (beta2 ** step) / bias_correction2

                step_size = lr / bias_correction1

                if rho_t > 4:
                    rect = math.sqrt(
                        ((rho_t - 4) * (rho_t - 2) * rho_inf)
                        / ((rho_inf - 4) * (rho_inf - 2) * rho_t)
                    )
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                    p.addcdiv_(exp_avg, denom, value=-step_size * rect)
                else:
                    # Variance not yet tractable — fall back to an SGD-with-momentum step.
                    p.add_(exp_avg, alpha=-step_size)

                # --- Lookahead slow-weight sync every k steps ---
                if step % k == 0:
                    slow = state["slow_buffer"]
                    slow.add_(p - slow, alpha=alpha)
                    p.copy_(slow)

        return loss
