"""Integration tests for inference.movement_analyzer.MovementAnalyzer."""

import numpy as np
import pytest

from configs.config_loader import load_config
from inference.exercise_library import ExerciseLibrary
from inference.movement_analyzer import MovementAnalyzer
from utils.joint_angles import MEDIAPIPE_LANDMARK_INDEX, compute_all_joint_angles


@pytest.fixture
def analyzer():
    cfg = load_config(force_reload=True)
    lib = ExerciseLibrary(cfg.exercises)
    return MovementAnalyzer(lib, fps=30, exercises_cfg=cfg.exercises)


def _make_landmarks(elbow_angle_deg: float):
    lm = np.zeros((33, 3))
    lm[MEDIAPIPE_LANDMARK_INDEX["left_shoulder"]] = [0.4, 0.2, 0]
    lm[MEDIAPIPE_LANDMARK_INDEX["right_shoulder"]] = [0.6, 0.2, 0]
    lm[MEDIAPIPE_LANDMARK_INDEX["left_hip"]] = [0.4, 0.6, 0]
    lm[MEDIAPIPE_LANDMARK_INDEX["right_hip"]] = [0.6, 0.6, 0]
    lm[MEDIAPIPE_LANDMARK_INDEX["left_elbow"]] = [0.4, 0.4, 0]
    rad = np.radians(180 - elbow_angle_deg)
    lm[MEDIAPIPE_LANDMARK_INDEX["left_wrist"]] = [0.4 + 0.2 * np.sin(rad), 0.4 + 0.2 * np.cos(rad), 0]
    lm[MEDIAPIPE_LANDMARK_INDEX["right_elbow"]] = [0.6, 0.4, 0]
    lm[MEDIAPIPE_LANDMARK_INDEX["right_wrist"]] = [0.6, 0.6, 0]
    return lm


def test_full_rep_recognized_tracked_and_counted(analyzer):
    sequence = list(np.linspace(170, 45, 20)) + list(np.linspace(45, 170, 20))
    result = None
    for angle in sequence:
        lm = _make_landmarks(angle)
        angles = compute_all_joint_angles(lm)
        result = analyzer.analyze_frame(lm, angles, pose_confidence=0.95)

    assert result.rep_count == 1
    assert result.exercise_key == "elbow_flexion"


def test_no_exercise_recognized_returns_safe_defaults(analyzer):
    lm = _make_landmarks(170.0)
    angles = compute_all_joint_angles(lm)
    result = analyzer.analyze_frame(lm, angles, pose_confidence=0.9)
    assert result.movement_quality == 1.0
    assert result.errors == []


def test_reset_clears_analyzer_state(analyzer):
    for angle in np.linspace(170, 45, 15):
        lm = _make_landmarks(angle)
        angles = compute_all_joint_angles(lm)
        analyzer.analyze_frame(lm, angles, pose_confidence=0.9)
    analyzer.reset()
    assert analyzer._active_exercise_key is None
