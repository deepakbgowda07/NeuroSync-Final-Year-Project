"""
view_detector.py
==================
Automatically determines which of three camera views the patient is
currently in — front-facing, left-side, or right-side — purely from
MediaPipe pose landmark geometry. The patient never selects a view
manually; several exercises require a specific view (see
`configs/exercises.yaml -> required_view`) and the system adapts to
whatever the patient is actually standing in.

Approach: MediaPipe reports a relative-depth `z` coordinate per
landmark (negative = closer to camera). In a true front view, the left
and right shoulders are at very similar depth and are widely separated
in x; in a side view, one shoulder is markedly closer to the camera
than the other (large z-difference) and the shoulder-to-shoulder x-span
collapses (they visually overlap). Left-vs-right side is then
determined by which shoulder is nearer the camera.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Deque, Optional

import numpy as np
from collections import deque

from utils.joint_angles import MEDIAPIPE_LANDMARK_INDEX
from utils.logger import get_logger

logger = get_logger(__name__)


class CameraView(str, Enum):
    FRONT = "front"
    LEFT_SIDE = "left_side"
    RIGHT_SIDE = "right_side"
    UNKNOWN = "unknown"


@dataclass
class ViewDetectionResult:
    view: CameraView
    confidence: float
    shoulder_x_span: float
    shoulder_z_diff: float


class ViewDetector:
    """Classifies the current camera view from a single frame's landmarks,
    with temporal smoothing (majority vote over a rolling window) so a
    single noisy frame can't flip the detected view mid-exercise.
    """

    def __init__(
        self,
        side_view_x_span_threshold: float = 0.12,
        side_view_z_diff_threshold: float = 0.08,
        smoothing_window: int = 15,
    ):
        self.side_view_x_span_threshold = side_view_x_span_threshold
        self.side_view_z_diff_threshold = side_view_z_diff_threshold
        self._history: Deque[CameraView] = deque(maxlen=smoothing_window)

    def reset(self) -> None:
        self._history.clear()

    def classify_frame(self, landmarks_xyz: np.ndarray) -> ViewDetectionResult:
        """Classify a single frame's (33, 3) landmarks. Use `detect()`
        instead for the temporally-smoothed, session-facing result."""
        left_shoulder = landmarks_xyz[MEDIAPIPE_LANDMARK_INDEX["left_shoulder"]]
        right_shoulder = landmarks_xyz[MEDIAPIPE_LANDMARK_INDEX["right_shoulder"]]

        x_span = abs(left_shoulder[0] - right_shoulder[0])
        z_diff = left_shoulder[2] - right_shoulder[2]  # negative z = closer to camera

        is_side_view = x_span < self.side_view_x_span_threshold or abs(z_diff) > self.side_view_z_diff_threshold

        if not is_side_view:
            view = CameraView.FRONT
            confidence = float(np.clip(1.0 - (x_span - self.side_view_x_span_threshold), 0.0, 1.0)) if x_span >= self.side_view_x_span_threshold else 1.0
        else:
            # More negative z = nearer the camera. If the left shoulder is
            # nearer, the patient's left side faces the camera -> "left_side" view.
            view = CameraView.LEFT_SIDE if z_diff < 0 else CameraView.RIGHT_SIDE
            confidence = float(np.clip(abs(z_diff) / max(self.side_view_z_diff_threshold, 1e-6), 0.0, 1.0))

        return ViewDetectionResult(view=view, confidence=confidence, shoulder_x_span=float(x_span), shoulder_z_diff=float(z_diff))

    def detect(self, landmarks_xyz: np.ndarray) -> ViewDetectionResult:
        """Classify the current frame and return a temporally-smoothed
        result: the majority view over the recent rolling window, with
        confidence equal to that view's share of the window (this
        prevents a single noisy frame from switching the detected view
        mid-repetition)."""
        frame_result = self.classify_frame(landmarks_xyz)
        self._history.append(frame_result.view)

        counts = {}
        for v in self._history:
            counts[v] = counts.get(v, 0) + 1
        majority_view = max(counts, key=counts.get)
        majority_confidence = counts[majority_view] / len(self._history)

        return ViewDetectionResult(
            view=majority_view,
            confidence=majority_confidence,
            shoulder_x_span=frame_result.shoulder_x_span,
            shoulder_z_diff=frame_result.shoulder_z_diff,
        )

    @staticmethod
    def matches_required_view(detected: ViewDetectionResult, required_view: str) -> bool:
        """`required_view` from configs/exercises.yaml is coarse
        ("front" | "side"); this maps the detector's three-way output
        onto that coarser requirement."""
        if required_view == "front":
            return detected.view == CameraView.FRONT
        if required_view == "side":
            return detected.view in (CameraView.LEFT_SIDE, CameraView.RIGHT_SIDE)
        return True  # no specific requirement
