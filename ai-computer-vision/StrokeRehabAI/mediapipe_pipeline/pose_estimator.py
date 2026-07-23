"""
pose_estimator.py
==================
Thin, typed wrapper around MediaPipe Pose (`mediapipe.solutions.pose`)
providing a stable interface for the rest of the pipeline, so the
underlying pose model can be swapped (e.g. for MediaPipe Tasks API,
or a custom pose model) without touching downstream code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PoseResult:
    """Structured result for a single frame's pose estimation."""

    detected: bool
    landmarks_xyz: Optional[np.ndarray] = None       # (33, 3) normalized coords
    landmarks_visibility: Optional[np.ndarray] = None  # (33,) visibility scores
    world_landmarks_xyz: Optional[np.ndarray] = None  # (33, 3) metric-scale coords
    raw_result: object = None                          # original MediaPipe result, for debugging


class PoseEstimator:
    """Runs MediaPipe Pose on individual BGR frames.

    Usage:
        estimator = PoseEstimator()
        with estimator:
            result = estimator.process(frame)
    """

    NUM_LANDMARKS = 33

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self._pose = None

    def open(self) -> None:
        import mediapipe as mp

        self._pose = mp.solutions.pose.Pose(
            static_image_mode=self.static_image_mode,
            model_complexity=self.model_complexity,
            smooth_landmarks=self.smooth_landmarks,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        logger.info(
            "MediaPipe Pose initialized (complexity=%d, smoothing=%s)",
            self.model_complexity,
            self.smooth_landmarks,
        )

    def process(self, frame_bgr: np.ndarray) -> PoseResult:
        """Run pose estimation on a single BGR frame."""
        if self._pose is None:
            raise RuntimeError("PoseEstimator.process() called before open().")

        import cv2

        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        result = self._pose.process(rgb_frame)

        if not result.pose_landmarks:
            return PoseResult(detected=False, raw_result=result)

        landmarks = result.pose_landmarks.landmark
        xyz = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float64)
        visibility = np.array([lm.visibility for lm in landmarks], dtype=np.float64)

        world_xyz = None
        if getattr(result, "pose_world_landmarks", None):
            world_lms = result.pose_world_landmarks.landmark
            world_xyz = np.array([[lm.x, lm.y, lm.z] for lm in world_lms], dtype=np.float64)

        return PoseResult(
            detected=True,
            landmarks_xyz=xyz,
            landmarks_visibility=visibility,
            world_landmarks_xyz=world_xyz,
            raw_result=result,
        )

    def close(self) -> None:
        if self._pose is not None:
            self._pose.close()
            logger.info("MediaPipe Pose closed.")
            self._pose = None

    def __enter__(self) -> "PoseEstimator":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
