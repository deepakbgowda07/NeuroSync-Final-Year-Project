"""
checkpoint_manager.py
======================
Checkpoint persistence for the training loop: saves "last.pt" every
epoch (for resume), keeps the top-K "best_*.pt" checkpoints ranked by a
validation metric, and bundles everything a full resume needs — model
weights, optimizer state, scheduler state, epoch number, metrics, and
the training configuration used to produce the run.
"""

from __future__ import annotations

import heapq
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.checkpoint_utils import load_checkpoint, save_checkpoint
from utils.logger import get_logger

logger = get_logger(__name__)


class CheckpointManager:
    """Tracks the "last" checkpoint (always overwritten, for resume) and
    the top-`save_top_k` checkpoints by a validation metric (lower is
    better, e.g. validation loss) — evicting the worst of the tracked
    best checkpoints once the cap is exceeded."""

    def __init__(self, checkpoint_dir: str, save_top_k: int = 3, metric_mode: str = "min"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_top_k = save_top_k
        self.metric_mode = metric_mode  # "min" (e.g. loss) or "max" (e.g. accuracy/F1)
        # min-heap of (comparable_metric, path); comparable_metric is negated for "max" mode
        # so the heap root is always the *worst* of the currently-kept best checkpoints.
        self._heap: List[Tuple[float, str]] = []

    def _comparable(self, metric: float) -> float:
        return metric if self.metric_mode == "min" else -metric

    def build_state(
        self,
        model,
        optimizer,
        scheduler,
        epoch: int,
        metrics: Dict[str, Any],
        config: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Assemble the full checkpoint payload."""
        return {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "metrics": metrics,
            "config": config,
        }

    def save_last(self, state: Dict[str, Any]) -> Path:
        """Always overwrite last.pt — the single source of truth for resuming."""
        path = self.checkpoint_dir / "last.pt"
        save_checkpoint(state, str(path))
        return path

    def maybe_save_best(self, state: Dict[str, Any], epoch: int, val_metric: float) -> Optional[Path]:
        """Save a checkpoint if it ranks in the current top-K by val_metric."""
        comparable = self._comparable(val_metric)
        checkpoint_path = self.checkpoint_dir / f"best_epoch{epoch:04d}_metric{val_metric:.4f}.pt"

        if len(self._heap) < self.save_top_k:
            save_checkpoint(state, str(checkpoint_path))
            heapq.heappush(self._heap, (-comparable, str(checkpoint_path)))
            self._refresh_best_alias()
            return checkpoint_path

        worst_neg_comparable, worst_path = self._heap[0]
        if -comparable > worst_neg_comparable:
            save_checkpoint(state, str(checkpoint_path))
            heapq.heapreplace(self._heap, (-comparable, str(checkpoint_path)))
            Path(worst_path).unlink(missing_ok=True)
            logger.info("Evicted checkpoint: %s", worst_path)
            self._refresh_best_alias()
            return checkpoint_path

        return None

    def _refresh_best_alias(self) -> None:
        """Copy the single best checkpoint to a stable `best.pt` path so
        inference/eval code can always load "the best" without knowing
        the metric-suffixed filename."""
        if not self._heap:
            return
        best_neg_comparable, best_path = max(self._heap, key=lambda item: item[0])
        alias_path = self.checkpoint_dir / "best.pt"
        try:
            import shutil

            shutil.copyfile(best_path, alias_path)
        except OSError as exc:
            logger.warning("Could not refresh best.pt alias: %s", exc)

    def best_checkpoint_path(self) -> str:
        if not self._heap:
            raise RuntimeError("No checkpoints saved yet.")
        return max(self._heap, key=lambda item: item[0])[1]

    def load_for_resume(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Load a checkpoint (default: last.pt) for resuming training."""
        resume_path = path or str(self.checkpoint_dir / "last.pt")
        checkpoint = load_checkpoint(resume_path)
        logger.info("Resuming from checkpoint: %s (epoch=%s)", resume_path, checkpoint.get("epoch"))
        return checkpoint
