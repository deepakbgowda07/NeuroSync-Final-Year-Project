"""
kinematic_features.py
======================
Velocity, acceleration, and bilateral symmetry features derived from
landmark sequences — supplements static joint angles with movement
dynamics, which matter clinically for stroke rehab (e.g. compensatory
movement patterns, movement smoothness/jerk).

TODO (next development phase): add a jerk-based smoothness metric
(third derivative of position) once frame-rate stability is confirmed
on the target webcam hardware.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


class KinematicFeatureExtractor:
    """Derives velocity/acceleration/symmetry features from a landmark sequence."""

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.dt = 1.0 / fps if fps > 0 else 1.0 / 30.0

    def velocity(self, sequence_xyz: np.ndarray) -> np.ndarray:
        """First derivative of landmark position w.r.t. time -> (T-1, 33, 3)."""
        return np.diff(sequence_xyz, axis=0) / self.dt

    def acceleration(self, sequence_xyz: np.ndarray) -> np.ndarray:
        """Second derivative of landmark position w.r.t. time -> (T-2, 33, 3)."""
        vel = self.velocity(sequence_xyz)
        return np.diff(vel, axis=0) / self.dt

    def mean_speed_per_joint(self, sequence_xyz: np.ndarray) -> np.ndarray:
        """Mean scalar speed per joint across the sequence -> (33,)."""
        vel = self.velocity(sequence_xyz)
        speed = np.linalg.norm(vel, axis=-1)  # (T-1, 33)
        return speed.mean(axis=0)

    def bilateral_symmetry_index(
        self, sequence_xyz: np.ndarray, left_idx: int, right_idx: int
    ) -> float:
        """Compares movement magnitude between a left/right joint pair.
        Returns a value in [0, 1] where 1.0 means perfectly symmetric
        movement amplitude — relevant for detecting compensatory use of
        the unaffected side in hemiparetic patients."""
        left_motion = np.linalg.norm(self.velocity(sequence_xyz)[:, left_idx, :], axis=-1).sum()
        right_motion = np.linalg.norm(self.velocity(sequence_xyz)[:, right_idx, :], axis=-1).sum()
        total = left_motion + right_motion
        if total < 1e-8:
            return 1.0
        return 1.0 - abs(left_motion - right_motion) / total

    def joint_distance(self, sequence_xyz: np.ndarray, joint_a: int, joint_b: int) -> np.ndarray:
        """Euclidean distance between two joints at every frame -> (T,).
        Useful e.g. for hand-to-hip distance during reaching exercises."""
        diff = sequence_xyz[:, joint_a, :] - sequence_xyz[:, joint_b, :]
        return np.linalg.norm(diff, axis=-1)

    def pairwise_joint_distance_matrix(self, frame_xyz: np.ndarray) -> np.ndarray:
        """Full (33, 33) pairwise Euclidean distance matrix for one frame —
        a compact whole-body-configuration descriptor sometimes used as an
        auxiliary model input alongside raw coordinates."""
        diffs = frame_xyz[:, None, :] - frame_xyz[None, :, :]
        return np.linalg.norm(diffs, axis=-1)

    def angular_velocity(self, angle_sequence_deg: np.ndarray) -> np.ndarray:
        """First derivative of a (T,) or (T, num_angles) joint-angle
        sequence w.r.t. time, in degrees/second."""
        return np.diff(angle_sequence_deg, axis=0) / self.dt

    def angular_acceleration(self, angle_sequence_deg: np.ndarray) -> np.ndarray:
        """Second derivative of a joint-angle sequence, in degrees/second^2."""
        return np.diff(self.angular_velocity(angle_sequence_deg), axis=0) / self.dt
