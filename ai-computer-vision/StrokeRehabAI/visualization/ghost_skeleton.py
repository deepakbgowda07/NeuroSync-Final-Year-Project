"""
ghost_skeleton.py
===================
Renders a semi-transparent "ghost" reference skeleton (e.g. a target
pose or a smoothed trailing average) alongside the live skeleton, to
help patients visually match correct form.

TODO (next development phase): source ghost-skeleton target poses from
a clinician-approved reference exercise recording rather than the
patient's own smoothed trailing average.
"""

from __future__ import annotations

import numpy as np

from utils.joint_angles import MEDIAPIPE_LANDMARK_INDEX
from visualization.skeleton_renderer import POSE_CONNECTIONS


class GhostSkeletonRenderer:
    """Draws a translucent reference/ghost skeleton overlay."""

    def __init__(self, visualization_cfg):
        self.opacity = visualization_cfg.ghost_skeleton_opacity
        self.color = (200, 200, 200)

    def draw(self, frame: np.ndarray, ghost_landmarks_xyz: np.ndarray) -> np.ndarray:
        import cv2

        overlay = frame.copy()
        h, w = frame.shape[:2]
        pixel_coords = {
            name: (int(ghost_landmarks_xyz[idx][0] * w), int(ghost_landmarks_xyz[idx][1] * h))
            for name, idx in MEDIAPIPE_LANDMARK_INDEX.items()
        }

        for joint_a, joint_b in POSE_CONNECTIONS:
            if joint_a in pixel_coords and joint_b in pixel_coords:
                cv2.line(overlay, pixel_coords[joint_a], pixel_coords[joint_b], self.color, 2, cv2.LINE_AA)

        return cv2.addWeighted(overlay, self.opacity, frame, 1 - self.opacity, 0)
