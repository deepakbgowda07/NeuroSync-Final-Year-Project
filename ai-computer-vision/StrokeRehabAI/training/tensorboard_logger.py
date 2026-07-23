"""
tensorboard_logger.py
======================
Thin wrapper around torch.utils.tensorboard.SummaryWriter for
consistent metric/scalar/image logging across the training loop.
"""

from __future__ import annotations

from typing import Dict, Optional


class TensorboardLogger:
    def __init__(self, log_dir: str = "logs/tensorboard"):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir=log_dir)

    def log_scalars(self, tag_prefix: str, metrics: Dict[str, float], step: int) -> None:
        for name, value in metrics.items():
            if value is None:
                continue
            self.writer.add_scalar(f"{tag_prefix}/{name}", value, step)

    def log_learning_rate(self, lr: float, step: int) -> None:
        self.writer.add_scalar("train/learning_rate", lr, step)

    def log_confusion_matrix_figure(self, figure, step: int, tag: str = "eval/confusion_matrix") -> None:
        self.writer.add_figure(tag, figure, step)

    def close(self) -> None:
        self.writer.close()
