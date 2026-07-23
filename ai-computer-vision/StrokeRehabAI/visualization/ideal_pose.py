"""
ideal_pose.py
===============
Generates an approximate "ideal pose" — the target arm position for
the current exercise — by rotating the relevant limb segment about its
proximal joint until it reaches the exercise's target angle, holding
everything else at the patient's current, actual position. This feeds
`visualization/ghost_skeleton.py`'s translucent reference overlay.

This is a geometric approximation (rotate the limb in the same plane
it's currently in), not a biomechanically precise inverse-kinematics
solve — good enough to give the patient an intuitive "move your arm
toward here" visual target, which is this feature's actual purpose.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from inference.exercise_library import ExerciseDefinition
from utils.geometry import normalize_vector
from utils.joint_angles import MEDIAPIPE_LANDMARK_INDEX
from utils.logger import get_logger

logger = get_logger(__name__)

# Which limb segment to rotate, and about which joint, per exercise's
# primary angle definition (vertex joint = the joint we rotate about).
_ANGLE_TO_SEGMENT = {
    "left_elbow_angle": ("left_elbow", "left_wrist"),
    "right_elbow_angle": ("right_elbow", "right_wrist"),
    "left_shoulder_angle": ("left_shoulder", "left_elbow"),
    "right_shoulder_angle": ("right_shoulder", "right_elbow"),
}


def generate_ideal_pose(landmarks_xyz: np.ndarray, definition: ExerciseDefinition, side: str = "left") -> Optional[np.ndarray]:
    """Return a (33, 3) landmark array with the relevant limb rotated to
    the exercise's target angle, or None if the exercise's primary angle
    isn't one this generator knows how to visualize (e.g. the rotation
    proxy angles, which don't correspond to a simple single-segment
    rotation)."""
    angle_name = definition.primary_angle if side == "left" else definition.secondary_angle
    if angle_name not in _ANGLE_TO_SEGMENT:
        return None

    vertex_name, endpoint_name = _ANGLE_TO_SEGMENT[angle_name]
    vertex_idx = MEDIAPIPE_LANDMARK_INDEX[vertex_name]
    endpoint_idx = MEDIAPIPE_LANDMARK_INDEX[endpoint_name]

    # Determine the reference joint the angle is measured from (the
    # "a" point in angle_between_points(a, vertex, endpoint)).
    reference_name = "left_hip" if "shoulder" in vertex_name else vertex_name.replace("elbow", "shoulder")
    reference_idx = MEDIAPIPE_LANDMARK_INDEX.get(reference_name, vertex_idx)

    vertex = landmarks_xyz[vertex_idx]
    endpoint = landmarks_xyz[endpoint_idx]
    reference = landmarks_xyz[reference_idx]

    reference_vec = normalize_vector(reference - vertex)
    current_vec = endpoint - vertex
    limb_length = np.linalg.norm(current_vec)
    if limb_length < 1e-6:
        return None

    # Rotate `reference_vec` by `target_deg` within the plane defined by
    # the current reference/limb vectors, to get the target limb direction.
    normal = np.cross(reference_vec, normalize_vector(current_vec))
    normal_norm = np.linalg.norm(normal)
    if normal_norm < 1e-6:
        normal = np.array([0.0, 0.0, 1.0])
    else:
        normal = normal / normal_norm

    target_rad = np.radians(definition.target_deg)
    rotated_direction = (
        reference_vec * np.cos(target_rad)
        + np.cross(normal, reference_vec) * np.sin(target_rad)
        + normal * np.dot(normal, reference_vec) * (1 - np.cos(target_rad))
    )
    rotated_direction = normalize_vector(rotated_direction)

    ideal_landmarks = landmarks_xyz.copy()
    ideal_landmarks[endpoint_idx] = vertex + rotated_direction * limb_length
    return ideal_landmarks
