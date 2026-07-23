"""
calibration.py
================
Pre-exercise personalization/calibration: a short (target < 15 second)
session where the patient stands in a neutral pose while the system
estimates body-specific measurements (shoulder width, arm length,
torso length, baseline joint angles). These are used to normalize
subsequent exercise-quality measurements to the individual patient
rather than a population-average body — see `movement_analyzer.py`.

Calibration accumulates a buffer of frames, and only *accepts* the
result once the buffered joint angles are stable (std-dev below a
configured threshold) — an unstable buffer (patient still moving,
poor pose detection) is rejected rather than silently calibrating
against noisy data.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from utils.geometry import euclidean_distance
from utils.joint_angles import MEDIAPIPE_LANDMARK_INDEX, compute_all_joint_angles
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CalibrationProfile:
    neutral_landmarks_xyz: np.ndarray
    shoulder_width: float
    left_arm_length: float
    right_arm_length: float
    torso_length: float
    baseline_joint_angles: Dict[str, float]
    num_frames_used: int
    calibrated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "shoulder_width": self.shoulder_width,
            "left_arm_length": self.left_arm_length,
            "right_arm_length": self.right_arm_length,
            "torso_length": self.torso_length,
            "baseline_joint_angles": self.baseline_joint_angles,
            "num_frames_used": self.num_frames_used,
            "calibrated_at": self.calibrated_at,
        }


class CalibrationSession:
    """Runs a short calibration session over a stream of incoming frames.

    Usage (called once per frame from the real-time pipeline while the
    UI shows a "hold still" countdown):

        session = CalibrationSession(duration_seconds=12, ...)
        session.start()
        while not session.is_complete:
            ok = session.add_frame(landmarks_xyz)
        profile = session.finalize()   # raises if the buffer wasn't stable enough
    """

    def __init__(
        self,
        duration_seconds: float = 12.0,
        min_frames_required: int = 30,
        stability_std_threshold_deg: float = 4.0,
        required_landmarks: Optional[List[str]] = None,
    ):
        self.duration_seconds = duration_seconds
        self.min_frames_required = min_frames_required
        self.stability_std_threshold_deg = stability_std_threshold_deg
        self.required_landmarks = required_landmarks or [
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
        ]

        self._start_time: Optional[float] = None
        self._landmark_buffer: List[np.ndarray] = []
        self._angle_buffer: List[Dict[str, float]] = []

    def start(self) -> None:
        self._start_time = time.perf_counter()
        self._landmark_buffer.clear()
        self._angle_buffer.clear()
        logger.info("Calibration session started (target duration=%.1fs).", self.duration_seconds)

    def add_frame(self, landmarks_xyz: Optional[np.ndarray]) -> bool:
        """Feed one frame's landmarks (skip frames with no detection).
        Returns True if the frame was accepted into the buffer."""
        if self._start_time is None:
            raise RuntimeError("CalibrationSession.add_frame() called before start().")
        if landmarks_xyz is None:
            return False

        self._landmark_buffer.append(landmarks_xyz)
        self._angle_buffer.append(compute_all_joint_angles(landmarks_xyz))
        return True

    @property
    def elapsed_seconds(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.perf_counter() - self._start_time

    @property
    def is_complete(self) -> bool:
        return self.elapsed_seconds >= self.duration_seconds

    @property
    def progress_fraction(self) -> float:
        return float(np.clip(self.elapsed_seconds / self.duration_seconds, 0.0, 1.0))

    def is_stable(self) -> bool:
        """Whether the buffered angle readings are stable enough to trust
        (patient standing reasonably still, pose reliably detected)."""
        if len(self._angle_buffer) < self.min_frames_required:
            return False

        for angle_name in ("left_elbow_angle", "right_elbow_angle", "left_shoulder_angle", "right_shoulder_angle"):
            values = [frame.get(angle_name, np.nan) for frame in self._angle_buffer]
            values = [v for v in values if not np.isnan(v)]
            if len(values) < 2:
                continue
            if float(np.std(values)) > self.stability_std_threshold_deg:
                return False
        return True

    def finalize(self) -> CalibrationProfile:
        """Compute the CalibrationProfile from the buffered frames.

        Raises RuntimeError if too few frames were captured or the
        buffer wasn't stable — callers should prompt the patient to
        redo calibration rather than silently proceeding with bad data.
        """
        if len(self._landmark_buffer) < self.min_frames_required:
            raise RuntimeError(
                f"Calibration failed: only {len(self._landmark_buffer)} usable frames captured "
                f"(need at least {self.min_frames_required}). Ensure the full upper body is visible."
            )
        if not self.is_stable():
            raise RuntimeError(
                "Calibration failed: joint angles were not stable enough during the session "
                "(patient may have been moving). Please stand still and retry."
            )

        stacked = np.stack(self._landmark_buffer)
        neutral_landmarks = np.median(stacked, axis=0)

        idx = MEDIAPIPE_LANDMARK_INDEX
        shoulder_width = euclidean_distance(neutral_landmarks[idx["left_shoulder"]], neutral_landmarks[idx["right_shoulder"]])
        left_arm_length = euclidean_distance(neutral_landmarks[idx["left_shoulder"]], neutral_landmarks[idx["left_elbow"]]) + \
            euclidean_distance(neutral_landmarks[idx["left_elbow"]], neutral_landmarks[idx["left_wrist"]])
        right_arm_length = euclidean_distance(neutral_landmarks[idx["right_shoulder"]], neutral_landmarks[idx["right_elbow"]]) + \
            euclidean_distance(neutral_landmarks[idx["right_elbow"]], neutral_landmarks[idx["right_wrist"]])
        hip_mid = (neutral_landmarks[idx["left_hip"]] + neutral_landmarks[idx["right_hip"]]) / 2.0
        shoulder_mid = (neutral_landmarks[idx["left_shoulder"]] + neutral_landmarks[idx["right_shoulder"]]) / 2.0
        torso_length = euclidean_distance(shoulder_mid, hip_mid)

        baseline_angles: Dict[str, float] = {}
        angle_names = self._angle_buffer[0].keys()
        for name in angle_names:
            values = [frame.get(name, np.nan) for frame in self._angle_buffer]
            values = [v for v in values if not np.isnan(v)]
            baseline_angles[name] = float(np.median(values)) if values else float("nan")

        profile = CalibrationProfile(
            neutral_landmarks_xyz=neutral_landmarks,
            shoulder_width=float(shoulder_width),
            left_arm_length=float(left_arm_length),
            right_arm_length=float(right_arm_length),
            torso_length=float(torso_length),
            baseline_joint_angles=baseline_angles,
            num_frames_used=len(self._landmark_buffer),
        )
        logger.info(
            "Calibration complete: %d frames, shoulder_width=%.3f, torso_length=%.3f.",
            profile.num_frames_used, profile.shoulder_width, profile.torso_length,
        )
        return profile
