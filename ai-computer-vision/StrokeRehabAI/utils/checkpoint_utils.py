"""
checkpoint_utils.py
====================
Helpers for saving/loading PyTorch model + optimizer checkpoints in a
consistent format, shared by training/checkpoint_manager.py and
inference/predictor.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from utils.file_io import ensure_dir
from utils.logger import get_logger

logger = get_logger(__name__)


def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_path: str,
) -> Path:
    """Persist a training checkpoint dict (model/optimizer/epoch/metrics) to disk."""
    import torch

    path = Path(checkpoint_path)
    ensure_dir(path.parent)
    torch.save(state, path)
    logger.info("Saved checkpoint: %s", path)
    return path


def load_checkpoint(checkpoint_path: str, map_location: Optional[str] = None) -> Dict[str, Any]:
    """Load a checkpoint dict from disk onto the given device (or CPU by default).

    Uses `weights_only=False`: this project's checkpoints intentionally
    bundle non-tensor Python objects (metrics dict, training config)
    alongside the model/optimizer state, which PyTorch >=2.6's default
    `weights_only=True` restricted-unpickler rejects. Only load
    checkpoints produced by this project's own training run (see
    training/checkpoint_manager.py) — never an untrusted `.pt` file.
    """
    import torch

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=map_location or "cpu", weights_only=False)
    logger.info("Loaded checkpoint: %s", path)
    return checkpoint


def load_model_weights(model, checkpoint_path: str, strict: bool = True, map_location: Optional[str] = None):
    """Load only the model state_dict from a checkpoint into `model`, in place."""
    checkpoint = load_checkpoint(checkpoint_path, map_location=map_location)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=strict)
    logger.info("Loaded model weights from %s (strict=%s)", checkpoint_path, strict)
    return model
