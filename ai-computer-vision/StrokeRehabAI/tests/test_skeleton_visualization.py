"""Tests for visualization.skeleton_renderer (green/red joint coloring) and ideal_pose."""

import numpy as np
import pytest

from configs.config_loader import load_config
from inference.error_detector import DetectedError, ErrorType
from inference.exercise_library import ExerciseLibrary
from utils.joint_angles import MEDIAPIPE_LANDMARK_INDEX


@pytest.fixture
def cfg():
    return load_config(force_reload=True)


def _sample_landmarks():
    lm = np.random.default_rng(0).uniform(0.1, 0.9, size=(33, 3))
    return lm


def test_flagged_joints_empty_with_no_errors():
    from visualization.skeleton_renderer import SkeletonRenderer

    flagged = SkeletonRenderer._flagged_joints(None)
    assert flagged == set()
    flagged2 = SkeletonRenderer._flagged_joints([])
    assert flagged2 == set()


def test_flagged_joints_maps_shoulder_hiking_to_shoulders():
    from visualization.skeleton_renderer import SkeletonRenderer

    errors = [DetectedError(ErrorType.SHOULDER_HIKING, "minor", {})]
    flagged = SkeletonRenderer._flagged_joints(errors)
    assert "left_shoulder" in flagged
    assert "right_shoulder" in flagged


def test_draw_produces_same_shape_frame(cfg):
    from visualization.skeleton_renderer import SkeletonRenderer

    renderer = SkeletonRenderer(cfg.visualization)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    lm = _sample_landmarks()
    result = renderer.draw(frame, lm)
    assert result.shape == frame.shape


def test_draw_with_errors_does_not_crash(cfg):
    from visualization.skeleton_renderer import SkeletonRenderer

    renderer = SkeletonRenderer(cfg.visualization)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    lm = _sample_landmarks()
    errors = [DetectedError(ErrorType.TRUNK_COMPENSATION, "moderate", {})]
    result = renderer.draw(frame, lm, errors=errors)
    assert result.shape == frame.shape


def test_generate_ideal_pose_rotates_to_target_angle(cfg):
    from visualization.ideal_pose import generate_ideal_pose
    from utils.joint_angles import compute_all_joint_angles

    lib = ExerciseLibrary(cfg.exercises)
    defn = lib.get("elbow_flexion")

    lm = np.zeros((33, 3))
    lm[MEDIAPIPE_LANDMARK_INDEX["left_shoulder"]] = [0.4, 0.2, 0]
    lm[MEDIAPIPE_LANDMARK_INDEX["left_hip"]] = [0.4, 0.6, 0]
    lm[MEDIAPIPE_LANDMARK_INDEX["left_elbow"]] = [0.4, 0.4, 0]
    lm[MEDIAPIPE_LANDMARK_INDEX["left_wrist"]] = [0.4, 0.6, 0]

    ideal = generate_ideal_pose(lm, defn, side="left")
    assert ideal is not None
    angles = compute_all_joint_angles(ideal)
    assert angles["left_elbow_angle"] == pytest.approx(defn.target_deg, abs=1.0)


def test_generate_ideal_pose_returns_none_for_unmapped_angle(cfg):
    from visualization.ideal_pose import generate_ideal_pose

    lib = ExerciseLibrary(cfg.exercises)
    defn = lib.get("forearm_pronation")  # uses a rotation-proxy angle, not in _ANGLE_TO_SEGMENT
    lm = np.random.default_rng(0).uniform(0, 1, size=(33, 3))
    ideal = generate_ideal_pose(lm, defn, side="left")
    assert ideal is None
