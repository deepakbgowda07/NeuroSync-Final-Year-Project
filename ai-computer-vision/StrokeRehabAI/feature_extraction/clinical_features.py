"""
clinical_features.py
======================
Higher-level, clinically-named movement features that go beyond raw
joint angles — the kind of descriptors a physiotherapist would actually
reference (trunk lean, shoulder elevation, movement smoothness,
compensation indicators) rather than joint-angle numbers alone.

These build on utils.joint_angles / feature_extraction.kinematic_features
and are consumed by both offline dataset feature generation
(feature_extraction/feature_pipeline.py) and evaluation reporting.

TODO (next development phase): validate the compensation-indicator
thresholds below against real clinician-annotated sessions once
labeled data is available — current thresholds are principled defaults
from general kinesiology literature, not dataset-fitted values.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from utils.joint_angles import MEDIAPIPE_LANDMARK_INDEX
from utils.logger import get_logger

logger = get_logger(__name__)

_L_SHOULDER = MEDIAPIPE_LANDMARK_INDEX["left_shoulder"]
_R_SHOULDER = MEDIAPIPE_LANDMARK_INDEX["right_shoulder"]
_L_HIP = MEDIAPIPE_LANDMARK_INDEX["left_hip"]
_R_HIP = MEDIAPIPE_LANDMARK_INDEX["right_hip"]
_L_WRIST = MEDIAPIPE_LANDMARK_INDEX["left_wrist"]
_R_WRIST = MEDIAPIPE_LANDMARK_INDEX["right_wrist"]
_L_SHOULDER_MP = MEDIAPIPE_LANDMARK_INDEX["left_shoulder"]


class ClinicalFeatureExtractor:
    """Computes named clinical movement descriptors from a landmark sequence."""

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.dt = 1.0 / fps if fps > 0 else 1.0 / 30.0

    # ------------------------------------------------------------------
    # Posture
    # ------------------------------------------------------------------

    def trunk_lean_deg(self, sequence_xyz: np.ndarray) -> np.ndarray:
        """Trunk lean angle from vertical, per frame, in degrees.

        Computed as the angle between the shoulder-midpoint-to-hip-midpoint
        vector and the vertical (y) axis — 0 degrees is perfectly upright.
        """
        shoulder_mid = (sequence_xyz[:, _L_SHOULDER, :] + sequence_xyz[:, _R_SHOULDER, :]) / 2.0
        hip_mid = (sequence_xyz[:, _L_HIP, :] + sequence_xyz[:, _R_HIP, :]) / 2.0
        trunk_vector = shoulder_mid - hip_mid

        vertical = np.zeros_like(trunk_vector)
        vertical[:, 1] = -1.0  # image y-axis points down; "up" is -y

        dot = np.sum(trunk_vector * vertical, axis=-1)
        norms = np.linalg.norm(trunk_vector, axis=-1) * np.linalg.norm(vertical, axis=-1)
        norms = np.where(norms < 1e-8, 1e-8, norms)
        cos_angle = np.clip(dot / norms, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    def shoulder_elevation(self, sequence_xyz: np.ndarray, side: str = "left") -> np.ndarray:
        """Vertical displacement of the shoulder relative to the hip
        midpoint, normalized by torso length — flags shoulder hiking /
        elevation, a common compensatory pattern. Positive values mean
        the shoulder is raised relative to its neutral position."""
        shoulder_idx = _L_SHOULDER if side == "left" else _R_SHOULDER
        hip_mid = (sequence_xyz[:, _L_HIP, :] + sequence_xyz[:, _R_HIP, :]) / 2.0
        shoulder_mid = (sequence_xyz[:, _L_SHOULDER, :] + sequence_xyz[:, _R_SHOULDER, :]) / 2.0
        torso_length = np.linalg.norm(shoulder_mid - hip_mid, axis=-1)
        torso_length = np.where(torso_length < 1e-8, 1e-8, torso_length)

        neutral_y = shoulder_mid[:, 1]
        shoulder_y = sequence_xyz[:, shoulder_idx, 1]
        # Image y increases downward, so a smaller y = higher (elevated) shoulder.
        elevation = (neutral_y - shoulder_y) / torso_length
        return elevation

    def arm_extension(self, sequence_xyz: np.ndarray, side: str = "left") -> np.ndarray:
        """Normalized shoulder-to-wrist distance (fraction of torso
        length) per frame — a simple proxy for how extended the arm is,
        independent of camera distance."""
        shoulder_idx = _L_SHOULDER if side == "left" else _R_SHOULDER
        wrist_idx = _L_WRIST if side == "left" else _R_WRIST

        hip_mid = (sequence_xyz[:, _L_HIP, :] + sequence_xyz[:, _R_HIP, :]) / 2.0
        shoulder_mid = (sequence_xyz[:, _L_SHOULDER, :] + sequence_xyz[:, _R_SHOULDER, :]) / 2.0
        torso_length = np.linalg.norm(shoulder_mid - hip_mid, axis=-1)
        torso_length = np.where(torso_length < 1e-8, 1e-8, torso_length)

        arm_length = np.linalg.norm(sequence_xyz[:, wrist_idx, :] - sequence_xyz[:, shoulder_idx, :], axis=-1)
        return arm_length / torso_length

    # ------------------------------------------------------------------
    # Movement quality
    # ------------------------------------------------------------------

    def movement_smoothness(self, sequence_xyz: np.ndarray, joint_idx: int) -> float:
        """Normalized dimensionless jerk — a standard motor-control
        smoothness metric (lower magnitude = smoother movement). Computed
        for a single joint's trajectory across the sequence.

        Reference: Hogan & Sternad (2009), "Sensitivity of smoothness
        measures to movement duration, amplitude, and arrests."
        """
        position = sequence_xyz[:, joint_idx, :]
        velocity = np.diff(position, axis=0) / self.dt
        if len(velocity) < 3:
            return 0.0
        acceleration = np.diff(velocity, axis=0) / self.dt
        jerk = np.diff(acceleration, axis=0) / self.dt

        duration = len(position) * self.dt
        peak_speed = np.linalg.norm(velocity, axis=-1).max()
        if peak_speed < 1e-8:
            return 0.0

        jerk_squared_integral = np.sum(np.linalg.norm(jerk, axis=-1) ** 2) * self.dt
        dimensionless_jerk = (duration ** 5 / peak_speed ** 2) * jerk_squared_integral
        return float(-np.log(abs(dimensionless_jerk) + 1e-8))  # log-dimensionless-jerk (SPARC-style, higher = smoother)

    def exercise_duration_seconds(self, sequence_xyz: np.ndarray) -> float:
        """Total sequence duration in seconds, given the configured fps."""
        return len(sequence_xyz) * self.dt

    def compensation_indicators(self, sequence_xyz: np.ndarray, side: str = "left") -> Dict[str, float]:
        """Flags common compensatory movement patterns seen in
        hemiparetic stroke rehab (trunk lean, shoulder hiking) with
        simple threshold-based indicators.

        Returns a dict of {indicator_name: fraction_of_frames_flagged}.
        TODO: thresholds are principled defaults pending clinician
        validation — see module docstring.
        """
        trunk_lean = self.trunk_lean_deg(sequence_xyz)
        shoulder_elevation = self.shoulder_elevation(sequence_xyz, side=side)

        return {
            "excessive_trunk_lean_ratio": float(np.mean(trunk_lean > 15.0)),
            "shoulder_hiking_ratio": float(np.mean(shoulder_elevation > 0.15)),
            "mean_trunk_lean_deg": float(np.mean(trunk_lean)),
            "max_trunk_lean_deg": float(np.max(trunk_lean)),
        }
