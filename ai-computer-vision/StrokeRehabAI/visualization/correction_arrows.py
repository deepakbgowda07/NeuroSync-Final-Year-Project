"""
correction_arrows.py
======================
Draws directional arrows from the patient's current joint position
toward the target/reference position, giving an intuitive "move this
way" visual cue.

TODO (next development phase): drive arrow targets from real-time
model feedback (inference/feedback_engine.py) once trained weights
exist, rather than a static example offset.
"""

from __future__ import annotations

import numpy as np


class CorrectionArrowRenderer:
    def __init__(self, visualization_cfg):
        self.color = tuple(visualization_cfg.error_color)

    def draw(self, frame: np.ndarray, current_point_px, target_point_px) -> np.ndarray:
        import cv2

        cv2.arrowedLine(frame, current_point_px, target_point_px, self.color, 2, tipLength=0.3)
        return frame

    def draw_batch(self, frame: np.ndarray, corrections: list) -> np.ndarray:
        """corrections: list of (current_point_px, target_point_px) tuples."""
        for current, target in corrections:
            frame = self.draw(frame, current, target)
        return frame

    def from_ideal_pose(
        self, frame: np.ndarray, landmarks_xyz: np.ndarray, ideal_landmarks_xyz: np.ndarray,
        joint_names: list, min_pixel_distance: int = 12,
    ) -> np.ndarray:
        """Draw an arrow from each named joint's current pixel position
        to its position in `ideal_landmarks_xyz` (see
        visualization/ideal_pose.py) — skips joints that are already
        close enough that an arrow would be visual noise.
        """
        from utils.joint_angles import MEDIAPIPE_LANDMARK_INDEX

        h, w = frame.shape[:2]
        for name in joint_names:
            idx = MEDIAPIPE_LANDMARK_INDEX.get(name)
            if idx is None:
                continue
            current_px = (int(landmarks_xyz[idx][0] * w), int(landmarks_xyz[idx][1] * h))
            target_px = (int(ideal_landmarks_xyz[idx][0] * w), int(ideal_landmarks_xyz[idx][1] * h))
            distance = ((current_px[0] - target_px[0]) ** 2 + (current_px[1] - target_px[1]) ** 2) ** 0.5
            if distance >= min_pixel_distance:
                frame = self.draw(frame, current_px, target_px)
        return frame
