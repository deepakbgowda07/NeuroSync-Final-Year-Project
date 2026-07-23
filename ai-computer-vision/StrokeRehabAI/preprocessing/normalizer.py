"""
normalizer.py
=============
Pose normalization: removes translation and scale variance so the
model sees exercise *form*, not the patient's distance from the
camera or position in frame.

Standard approach: translate so the hip midpoint is the origin, then
scale by torso length (hip-to-shoulder distance).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from utils.joint_angles import MEDIAPIPE_LANDMARK_INDEX
from utils.logger import get_logger

logger = get_logger(__name__)


class PoseNormalizer:
    """Translation + scale normalization for a (33, 3) landmark frame."""

    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon

    def normalize_frame(self, landmarks_xyz: np.ndarray) -> np.ndarray:
        """Center on the hip midpoint and scale by torso length."""
        left_hip = landmarks_xyz[MEDIAPIPE_LANDMARK_INDEX["left_hip"]]
        right_hip = landmarks_xyz[MEDIAPIPE_LANDMARK_INDEX["right_hip"]]
        left_shoulder = landmarks_xyz[MEDIAPIPE_LANDMARK_INDEX["left_shoulder"]]
        right_shoulder = landmarks_xyz[MEDIAPIPE_LANDMARK_INDEX["right_shoulder"]]

        hip_center = (left_hip + right_hip) / 2.0
        shoulder_center = (left_shoulder + right_shoulder) / 2.0

        torso_length = float(np.linalg.norm(shoulder_center - hip_center))
        scale = max(torso_length, self.epsilon)

        centered = landmarks_xyz - hip_center
        return centered / scale

    def normalize_sequence(self, sequence_xyz: np.ndarray) -> np.ndarray:
        """Apply per-frame normalization across a (T, 33, 3) sequence."""
        return np.stack([self.normalize_frame(frame) for frame in sequence_xyz])
