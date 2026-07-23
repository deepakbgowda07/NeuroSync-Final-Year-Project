"""
seed.py
=======
Deterministic seeding helpers for reproducible experiments.
"""

from __future__ import annotations

import os
import random
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


def set_global_seed(seed: int = 42, deterministic_cudnn: bool = False) -> None:
    """Seed python, numpy, and torch RNGs for reproducibility.

    Args:
        seed: the seed value to apply everywhere.
        deterministic_cudnn: if True, forces deterministic cuDNN
            algorithms (slower, but bit-for-bit reproducible). Leave
            False for normal training speed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        logger.debug("NumPy not installed; skipping numpy seeding.")

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        logger.debug("PyTorch not installed; skipping torch seeding.")

    logger.info("Global seed set to %d (deterministic_cudnn=%s)", seed, deterministic_cudnn)
