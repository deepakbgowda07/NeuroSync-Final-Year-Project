"""
joint_angles.py
===============
Joint-angle computation from 3D (or 2D) body landmarks — the core
geometric primitive used for both real-time feedback (visualization
HUD) and offline exercise-quality scoring (evaluation metrics).

The functions here are pose-library agnostic: they operate on plain
(x, y, z) tuples / numpy arrays, so they work whether landmarks came
from MediaPipe, OpenPose, or a dataset's pre-extracted skeleton.

TODO (next development phase):
- Add per-exercise reference range-of-motion (ROM) tables sourced from
  clinical literature, to translate raw angles into pass/fail bands.
- Add temporal smoothing (e.g. one-euro filter) prior to angle calc.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from utils.geometry import to_np


def angle_between_points(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    """Angle at vertex `b`, formed by rays b->a and b->c, in degrees.

    This is the standard three-point joint-angle formula, e.g. for the
    elbow angle: a=shoulder, b=elbow, c=wrist.
    """
    a, b, c = to_np(a), to_np(b), to_np(c)
    ba = a - b
    bc = c - b

    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-8:
        return 0.0

    cosine_angle = np.dot(ba, bc) / denom
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine_angle)))


# MediaPipe Pose landmark indices for commonly used rehab-relevant joints.
# Reference: https://google.github.io/mediapipe/solutions/pose.html
MEDIAPIPE_LANDMARK_INDEX = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}

# (vertex_joint, endpoint_a, endpoint_b) triples for standard clinical angles.
JOINT_ANGLE_DEFINITIONS = {
    "left_elbow_angle": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow_angle": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_shoulder_angle": ("left_hip", "left_shoulder", "left_elbow"),
    "right_shoulder_angle": ("right_hip", "right_shoulder", "right_elbow"),
    "left_knee_angle": ("left_hip", "left_knee", "left_ankle"),
    "right_knee_angle": ("right_hip", "right_knee", "right_ankle"),
    "left_hip_angle": ("left_shoulder", "left_hip", "left_knee"),
    "right_hip_angle": ("right_shoulder", "right_hip", "right_knee"),
}


def shoulder_abduction_angle(hip: Sequence[float], shoulder: Sequence[float], elbow: Sequence[float]) -> float:
    """Shoulder abduction angle: elevation of the upper arm out to the
    side, measured in the frontal (x-y) plane — as opposed to
    `left_shoulder_angle` / flexion, which is unrestricted-plane and
    best interpreted from a side view. Abduction is best observed from
    the front view (see configs/exercises.yaml -> shoulder_abduction).

    Implemented by dropping the z (depth) coordinate before computing
    the standard three-point angle, so only left-right/up-down motion
    contributes.
    """
    hip_2d = to_np(hip)[:2]
    shoulder_2d = to_np(shoulder)[:2]
    elbow_2d = to_np(elbow)[:2]
    return angle_between_points(
        np.append(hip_2d, 0.0), np.append(shoulder_2d, 0.0), np.append(elbow_2d, 0.0)
    )


def forearm_rotation_proxy_deg(wrist: Sequence[float], thumb: Sequence[float], pinky: Sequence[float]) -> float:
    """Approximate forearm pronation/supination angle from MediaPipe
    Pose's coarse hand-landmark set (thumb + pinky tips relative to the
    wrist) — positive values trend toward pronation (palm-down), negative
    toward supination (palm-up), by convention of this project.

    IMPORTANT LIMITATION: MediaPipe *Pose* provides only 3 sparse points
    per hand (thumb, index, pinky), not the full 21-point MediaPipe
    *Hands* topology. This proxy is a coarse engineering approximation,
    not a clinically validated goniometer replacement — treat it as
    directionally useful (detecting *that* rotation is happening and
    roughly how far) rather than as a precise degree measurement. For
    clinical-grade pronation/supination tracking, integrate MediaPipe
    Hands in a future iteration (see docs/architecture.md).
    """
    wrist_v, thumb_v, pinky_v = to_np(wrist), to_np(thumb), to_np(pinky)
    thumb_vec = thumb_v - wrist_v
    pinky_vec = pinky_v - wrist_v

    # Signed angle between thumb and pinky vectors, projected onto the
    # x-z plane (roughly the plane the palm sweeps through during
    # pronation/supination when the forearm is held out in front).
    thumb_2d = np.array([thumb_vec[0], thumb_vec[2]])
    pinky_2d = np.array([pinky_vec[0], pinky_vec[2]])

    cross = thumb_2d[0] * pinky_2d[1] - thumb_2d[1] * pinky_2d[0]
    denom = (np.linalg.norm(thumb_2d) * np.linalg.norm(pinky_2d)) or 1e-8
    dot = np.dot(thumb_2d, pinky_2d)
    angle = np.degrees(np.arctan2(cross, dot))
    return float(angle)


def shoulder_rotation_proxy_deg(shoulder: Sequence[float], elbow: Sequence[float], wrist: Sequence[float]) -> float:
    """Approximate shoulder internal/external rotation angle, intended
    for use with the elbow flexed near 90 degrees at the side (the
    standard clinical rotation-testing position). Measures how far the
    forearm (elbow->wrist) has swept out of the sagittal plane defined
    by the upper arm (shoulder->elbow) and the vertical axis — a
    positive value trends toward external rotation, negative toward
    internal, by this project's convention.

    Like `forearm_rotation_proxy_deg`, this is an engineering
    approximation from body-only landmarks pending clinical validation
    (see that function's docstring for the same caveat).
    """
    shoulder_v, elbow_v, wrist_v = to_np(shoulder), to_np(elbow), to_np(wrist)
    upper_arm_vec = elbow_v - shoulder_v
    forearm_vec = wrist_v - elbow_v

    upper_arm_2d = np.array([upper_arm_vec[0], upper_arm_vec[2]])
    forearm_2d = np.array([forearm_vec[0], forearm_vec[2]])

    cross = upper_arm_2d[0] * forearm_2d[1] - upper_arm_2d[1] * forearm_2d[0]
    denom = (np.linalg.norm(upper_arm_2d) * np.linalg.norm(forearm_2d)) or 1e-8
    dot = np.dot(upper_arm_2d, forearm_2d)
    angle = np.degrees(np.arctan2(cross, dot))
    return float(angle)


def compute_all_joint_angles(landmarks_xyz: np.ndarray) -> Dict[str, float]:
    """Compute the standard set of clinical joint angles for one frame,
    plus the abduction and rotation/pronation proxy angles needed by
    the exercise library (see configs/exercises.yaml).

    Args:
        landmarks_xyz: (33, 3) array of MediaPipe Pose landmarks for a
            single frame (index order must match MediaPipe's convention).

    Returns:
        Dict mapping angle name -> degrees.
    """
    angles: Dict[str, float] = {}
    for angle_name, (j_a, j_b, j_c) in JOINT_ANGLE_DEFINITIONS.items():
        try:
            idx_a = MEDIAPIPE_LANDMARK_INDEX[j_a]
            idx_b = MEDIAPIPE_LANDMARK_INDEX[j_b]
            idx_c = MEDIAPIPE_LANDMARK_INDEX[j_c]
            angles[angle_name] = angle_between_points(
                landmarks_xyz[idx_a], landmarks_xyz[idx_b], landmarks_xyz[idx_c]
            )
        except (IndexError, KeyError):
            angles[angle_name] = float("nan")

    for side in ("left", "right"):
        try:
            hip = landmarks_xyz[MEDIAPIPE_LANDMARK_INDEX[f"{side}_hip"]]
            shoulder = landmarks_xyz[MEDIAPIPE_LANDMARK_INDEX[f"{side}_shoulder"]]
            elbow = landmarks_xyz[MEDIAPIPE_LANDMARK_INDEX[f"{side}_elbow"]]
            wrist = landmarks_xyz[MEDIAPIPE_LANDMARK_INDEX[f"{side}_wrist"]]
            thumb = landmarks_xyz[MEDIAPIPE_LANDMARK_INDEX[f"{side}_thumb"]]
            pinky = landmarks_xyz[MEDIAPIPE_LANDMARK_INDEX[f"{side}_pinky"]]

            angles[f"{side}_shoulder_abduction_angle"] = shoulder_abduction_angle(hip, shoulder, elbow)
            angles[f"{side}_forearm_rotation_proxy"] = forearm_rotation_proxy_deg(wrist, thumb, pinky)
            angles[f"{side}_shoulder_rotation_proxy"] = shoulder_rotation_proxy_deg(shoulder, elbow, wrist)
        except (IndexError, KeyError):
            angles[f"{side}_shoulder_abduction_angle"] = float("nan")
            angles[f"{side}_forearm_rotation_proxy"] = float("nan")
            angles[f"{side}_shoulder_rotation_proxy"] = float("nan")

    return angles


def angle_deviation(measured_deg: float, reference_deg: float) -> float:
    """Signed deviation (degrees) of a measured angle from a clinical reference."""
    return measured_deg - reference_deg
