"""
angle_features.py
==================
Extracts clinically meaningful joint-angle features per frame/sequence,
built on utils.joint_angles. These features are more interpretable
than raw coordinates and are used for both explainable feedback (HUD
overlay) and as an alternative/auxiliary model input representation.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from utils.joint_angles import JOINT_ANGLE_DEFINITIONS, compute_all_joint_angles
from utils.logger import get_logger

logger = get_logger(__name__)


class AngleFeatureExtractor:
    """Computes per-frame and per-sequence joint-angle feature summaries."""

    def __init__(self):
        self.angle_names = list(JOINT_ANGLE_DEFINITIONS.keys())

    def extract_frame(self, landmarks_xyz: np.ndarray) -> Dict[str, float]:
        return compute_all_joint_angles(landmarks_xyz)

    def extract_sequence(self, sequence_xyz: np.ndarray) -> np.ndarray:
        """Returns a (T, num_angles) array of joint angles across a sequence."""
        per_frame = [self.extract_frame(frame) for frame in sequence_xyz]
        return np.array([[angles[name] for name in self.angle_names] for angles in per_frame])

    def summary_statistics(self, sequence_xyz: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Return min/max/mean/range per angle across a sequence — useful
        for range-of-motion (ROM) based scoring in evaluation/metrics."""
        angle_array = self.extract_sequence(sequence_xyz)
        stats = {}
        for i, name in enumerate(self.angle_names):
            col = angle_array[:, i]
            col = col[~np.isnan(col)]
            if len(col) == 0:
                stats[name] = {"min": float("nan"), "max": float("nan"), "mean": float("nan"), "range": float("nan")}
                continue
            stats[name] = {
                "min": float(col.min()),
                "max": float(col.max()),
                "mean": float(col.mean()),
                "range": float(col.max() - col.min()),
            }
        return stats
