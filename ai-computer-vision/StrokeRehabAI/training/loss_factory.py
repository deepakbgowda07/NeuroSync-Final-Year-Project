"""
loss_factory.py
================
Config-driven loss function construction: CrossEntropy, MSE, Huber, and
class-weighted CrossEntropy (with optional automatic inverse-frequency
weight computation from the training split).
"""

from __future__ import annotations

from typing import List, Optional


def compute_class_weights(labels: List[int], num_classes: int):
    """Inverse-frequency class weights from a list of integer labels,
    normalized so weights sum to num_classes (keeps the loss scale
    comparable to an unweighted cross-entropy)."""
    import torch

    counts = torch.zeros(num_classes)
    for label in labels:
        if 0 <= label < num_classes:
            counts[label] += 1
    counts = torch.clamp(counts, min=1)  # avoid divide-by-zero for absent classes
    weights = 1.0 / counts
    weights = weights * (num_classes / weights.sum())
    return weights


def build_loss(model_cfg, training_cfg=None, class_weights=None):
    """Build the configured loss function.

    `training_cfg.loss` selects the loss family; `training_cfg.loss_params`
    supplies family-specific parameters (e.g. huber_delta). `class_weights`
    may be passed explicitly (a torch.Tensor) to override
    `loss_params.class_weights`.
    """
    import torch
    import torch.nn as nn

    loss_name = (training_cfg.loss if training_cfg and hasattr(training_cfg, "loss") else "cross_entropy").lower()
    loss_params = training_cfg.get("loss_params", {}) if training_cfg and hasattr(training_cfg, "get") else {}

    if loss_name in ("cross_entropy", "weighted_cross_entropy"):
        weights = class_weights
        if weights is None and loss_params:
            configured = loss_params.get("class_weights")
            if isinstance(configured, (list, tuple)):
                weights = torch.tensor(list(configured), dtype=torch.float32)
        return nn.CrossEntropyLoss(weight=weights)

    if loss_name == "mse":
        return nn.MSELoss()

    if loss_name == "huber":
        delta = loss_params.get("huber_delta", 1.0) if loss_params else 1.0
        return nn.HuberLoss(delta=delta)

    # Fallback to the model's output_head for backward compatibility with
    # configs that don't set `training.loss` explicitly.
    if model_cfg is not None:
        if model_cfg.output_head == "classification":
            return nn.CrossEntropyLoss(weight=class_weights)
        if model_cfg.output_head == "regression":
            return nn.MSELoss()

    raise ValueError(f"Unknown loss: {loss_name}")
