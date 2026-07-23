"""
scheduler.py
============
Config-driven learning-rate scheduler construction.
"""

from __future__ import annotations


def build_scheduler(optimizer, training_cfg):
    import torch.optim.lr_scheduler as lr_scheduler

    name = (training_cfg.scheduler or "none").lower()

    if name == "none":
        return None
    if name == "cosine":
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_cfg.epochs)
    if name == "step":
        return lr_scheduler.StepLR(optimizer, step_size=max(1, training_cfg.epochs // 4), gamma=0.5)
    if name == "plateau":
        return lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    raise ValueError(f"Unknown scheduler: {training_cfg.scheduler}")
