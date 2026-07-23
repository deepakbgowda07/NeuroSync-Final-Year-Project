"""
skeleton_renderer.py
=====================
Draws the live detected skeleton (joints + bone connections) onto a
video frame. Renders green when the tracked joint/limb is performing
correctly, and red at joints flagged by error detection — e.g. a
shoulder joint turns red during a detected "shoulder hiking" error,
while the rest of the skeleton stays green.
"""

from __future__ import annotations

from typing import Dict, Optional, Set

import numpy as np

from inference.error_detector import DetectedError, ErrorType
from utils.joint_angles import MEDIAPIPE_LANDMARK_INDEX, compute_all_joint_angles

# Standard MediaPipe Pose bone connections (subset relevant to limb tracking).
POSE_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

# Which joints turn red when a given error type is active — used to make
# the visual feedback point at the specific body part involved, not just
# recolor the whole skeleton.
_ERROR_TO_JOINTS = {
    ErrorType.SHOULDER_HIKING: {"left_shoulder", "right_shoulder"},
    ErrorType.TRUNK_COMPENSATION: {"left_hip", "right_hip", "left_shoulder", "right_shoulder"},
    ErrorType.BODY_LEAN: {"left_hip", "right_hip"},
    ErrorType.POOR_ALIGNMENT: {"left_shoulder", "right_shoulder", "left_hip", "right_hip"},
    ErrorType.INSUFFICIENT_ROM: {"left_elbow", "right_elbow", "left_wrist", "right_wrist"},
    ErrorType.EXCESSIVE_ROM: {"left_elbow", "right_elbow", "left_wrist", "right_wrist"},
    ErrorType.ASYMMETRICAL_MOTION: {"left_elbow", "right_elbow", "left_wrist", "right_wrist"},
    ErrorType.INCORRECT_JOINT_ANGLE: {"left_elbow", "right_elbow"},
}


class SkeletonRenderer:
    """Renders the pose skeleton onto a BGR frame, coloring joints
    green (correct) or red (flagged by an active error)."""

    def __init__(self, visualization_cfg):
        self.correct_color = tuple(visualization_cfg.get("correct_color", visualization_cfg.skeleton_color))
        self.error_color = tuple(visualization_cfg.get("error_color", (0, 0, 255)))
        self.default_color = tuple(visualization_cfg.skeleton_color)
        self.joint_radius = visualization_cfg.joint_radius_px
        self.thickness = visualization_cfg.connection_thickness_px
        self.show_joint_angles = visualization_cfg.show_joint_angles

    def draw(self, frame: np.ndarray, landmarks_xyz: np.ndarray, errors: Optional[list] = None) -> np.ndarray:
        import cv2

        h, w = frame.shape[:2]
        pixel_coords = {
            name: (int(landmarks_xyz[idx][0] * w), int(landmarks_xyz[idx][1] * h))
            for name, idx in MEDIAPIPE_LANDMARK_INDEX.items()
        }
        flagged_joints = self._flagged_joints(errors)
        skeleton_ok = not flagged_joints

        for joint_a, joint_b in POSE_CONNECTIONS:
            if joint_a in pixel_coords and joint_b in pixel_coords:
                line_color = self.correct_color if skeleton_ok else self.default_color
                cv2.line(frame, pixel_coords[joint_a], pixel_coords[joint_b], line_color, self.thickness)

        for name, point in pixel_coords.items():
            color = self.error_color if name in flagged_joints else self.correct_color
            cv2.circle(frame, point, self.joint_radius, color, -1)

        if self.show_joint_angles:
            angles = compute_all_joint_angles(landmarks_xyz)
            self._draw_angle_labels(frame, pixel_coords, angles)

        return frame

    @staticmethod
    def _flagged_joints(errors: Optional[list]) -> Set[str]:
        if not errors:
            return set()
        flagged: Set[str] = set()
        for error in errors:
            error_type = error.error_type if isinstance(error, DetectedError) else error
            flagged |= _ERROR_TO_JOINTS.get(error_type, set())
        return flagged

    def _draw_angle_labels(self, frame: np.ndarray, pixel_coords: dict, angles: dict) -> None:
        import cv2

        label_map = {
            "left_elbow_angle": "left_elbow",
            "right_elbow_angle": "right_elbow",
            "left_knee_angle": "left_knee",
            "right_knee_angle": "right_knee",
        }
        for angle_name, joint_name in label_map.items():
            if joint_name in pixel_coords and angle_name in angles and not np.isnan(angles[angle_name]):
                x, y = pixel_coords[joint_name]
                cv2.putText(
                    frame, f"{angles[angle_name]:.0f}deg", (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.default_color, 1, cv2.LINE_AA,
                )
