"""
landmark_extractor.py
======================
Converts raw PoseResult objects into flat feature vectors suitable
for model input, and provides named-landmark convenience accessors
for the visualization and joint-angle modules.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from mediapipe_pipeline.pose_estimator import PoseResult
from utils.logger import get_logger

logger = get_logger(__name__)

# Human-readable names for MediaPipe's 33 pose landmarks, in index order.
LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]


class LandmarkExtractor:
    """Turns a PoseResult into named landmarks and flat feature vectors."""

    def __init__(self, use_world_landmarks: bool = False, visibility_threshold: float = 0.5):
        self.use_world_landmarks = use_world_landmarks
        self.visibility_threshold = visibility_threshold

    def to_named_dict(self, result: PoseResult) -> Dict[str, np.ndarray]:
        """Return {landmark_name: (x, y, z)} for a detected pose."""
        if not result.detected:
            return {}

        coords = result.world_landmarks_xyz if self.use_world_landmarks else result.landmarks_xyz
        return {name: coords[i] for i, name in enumerate(LANDMARK_NAMES)}

    def to_feature_vector(self, result: PoseResult) -> Optional[np.ndarray]:
        """Flatten (33, 3) landmarks into a (99,) feature vector for model input.

        Returns None if no pose was detected in this frame (caller should
        decide how to handle gaps, e.g. hold previous frame or skip).
        """
        if not result.detected:
            return None

        coords = result.world_landmarks_xyz if self.use_world_landmarks else result.landmarks_xyz
        return coords.flatten()

    def low_visibility_landmarks(self, result: PoseResult) -> list:
        """Return names of landmarks below the configured visibility threshold —
        useful for flagging occlusion issues during a rehab session."""
        if not result.detected or result.landmarks_visibility is None:
            return []
        return [
            LANDMARK_NAMES[i]
            for i, vis in enumerate(result.landmarks_visibility)
            if vis < self.visibility_threshold
        ]

    def overall_confidence(self, result: PoseResult, key_landmarks: Optional[list] = None) -> float:
        """A single [0, 1] confidence score for the current frame's pose
        estimate, used by the real-time pipeline's HUD and by
        inference/smoothing.py's confidence smoother.

        Defaults to the mean visibility across the upper-body landmarks
        most relevant to rehab exercises (shoulders/elbows/wrists/hips);
        pass `key_landmarks` to score a different/smaller joint set.
        """
        if not result.detected or result.landmarks_visibility is None:
            return 0.0

        if key_landmarks is None:
            key_landmarks = [
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist", "left_hip", "right_hip",
            ]

        indices = [LANDMARK_NAMES.index(name) for name in key_landmarks if name in LANDMARK_NAMES]
        if not indices:
            return 0.0
        return float(np.mean(result.landmarks_visibility[indices]))
