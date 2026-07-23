"""
optimizer_factory.py
=====================
Config-driven optimizer construction. Supports AdamW, Adam, SGD, and
Ranger (RAdam + Lookahead, see training/optimizers/ranger.py).
"""

from __future__ import annotations

from training.optimizers.ranger import Ranger
from utils.logger import get_logger

logger = get_logger(__name__)


def build_optimizer(model, training_cfg):
    import torch.optim as optim

    name = training_cfg.optimizer.lower()
    params = model.parameters()
    lr = training_cfg.learning_rate
    weight_decay = training_cfg.weight_decay
    opt_params = training_cfg.get("optimizer_params", {}) if hasattr(training_cfg, "get") else {}

    if name == "adamw":
        optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif name == "adam":
        optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        momentum = opt_params.get("sgd_momentum", 0.9) if opt_params else 0.9
        optimizer = optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    elif name == "ranger":
        betas = tuple(opt_params.get("ranger_betas", (0.95, 0.999))) if opt_params else (0.95, 0.999)
        eps = opt_params.get("ranger_eps", 1e-5) if opt_params else 1e-5
        k = opt_params.get("ranger_k", 6) if opt_params else 6
        alpha = opt_params.get("ranger_alpha", 0.5) if opt_params else 0.5
        optimizer = Ranger(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, k=k, alpha=alpha)
    else:
        raise ValueError(f"Unknown optimizer: {training_cfg.optimizer}")

    logger.info("Built optimizer: %s (lr=%s, weight_decay=%s)", name, lr, weight_decay)
    return optimizer
