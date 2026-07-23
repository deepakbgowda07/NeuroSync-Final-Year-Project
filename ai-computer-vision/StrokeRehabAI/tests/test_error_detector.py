"""Tests for inference.error_detector.ErrorDetector."""

import numpy as np
import pytest

from configs.config_loader import load_config
from inference.exercise_library import ExerciseLibrary
from inference.error_detector import ErrorDetector, ErrorType
from inference.phase_detector import ExercisePhase, PhaseResult
from utils.joint_angles import MEDIAPIPE_LANDMARK_INDEX, compute_all_joint_angles


@pytest.fixture
def elbow_flexion_defn():
    cfg = load_config(force_reload=True)
    return ExerciseLibrary(cfg.exercises).get("elbow_flexion")


def _basic_landmarks():
    lm = np.zeros((33, 3))
    lm[MEDIAPIPE_LANDMARK_INDEX["left_shoulder"]] = [0.4, 0.2, 0]
    lm[MEDIAPIPE_LANDMARK_INDEX["right_shoulder"]] = [0.6, 0.2, 0]
    lm[MEDIAPIPE_LANDMARK_INDEX["left_hip"]] = [0.4, 0.6, 0]
    lm[MEDIAPIPE_LANDMARK_INDEX["right_hip"]] = [0.6, 0.6, 0]
    lm[MEDIAPIPE_LANDMARK_INDEX["left_elbow"]] = [0.4, 0.4, 0]
    lm[MEDIAPIPE_LANDMARK_INDEX["right_elbow"]] = [0.6, 0.4, 0]
    lm[MEDIAPIPE_LANDMARK_INDEX["left_wrist"]] = [0.4, 0.55, 0]
    lm[MEDIAPIPE_LANDMARK_INDEX["right_wrist"]] = [0.6, 0.55, 0]
    return lm


def test_detects_insufficient_rom_at_peak(elbow_flexion_defn):
    detector = ErrorDetector(fps=30)
    lm = _basic_landmarks()
    angles = compute_all_joint_angles(lm)
    phase_result = PhaseResult(phase=ExercisePhase.PEAK, progress_fraction=1.0, current_angle_deg=170.0, velocity_deg_per_sec=10.0)
    errors = detector.detect(lm, angles, elbow_flexion_defn, phase_result)
    assert any(e.error_type == ErrorType.INSUFFICIENT_ROM for e in errors)


def test_no_errors_for_good_form_at_target(elbow_flexion_defn):
    detector = ErrorDetector(fps=30)
    lm = _basic_landmarks()
    angles = compute_all_joint_angles(lm)
    phase_result = PhaseResult(phase=ExercisePhase.PEAK, progress_fraction=1.0, current_angle_deg=45.0, velocity_deg_per_sec=10.0)
    errors = detector.detect(lm, angles, elbow_flexion_defn, phase_result)
    rom_errors = [e for e in errors if e.error_type in (ErrorType.INSUFFICIENT_ROM, ErrorType.EXCESSIVE_ROM)]
    assert rom_errors == []


def test_detects_fast_movement(elbow_flexion_defn):
    detector = ErrorDetector(fps=30, fast_movement_deg_per_sec=100.0)
    lm = _basic_landmarks()
    angles = compute_all_joint_angles(lm)
    phase_result = PhaseResult(phase=ExercisePhase.MOVING_TO_TARGET, progress_fraction=0.5, current_angle_deg=100.0, velocity_deg_per_sec=300.0)
    errors = detector.detect(lm, angles, elbow_flexion_defn, phase_result)
    assert any(e.error_type == ErrorType.FAST_MOVEMENT for e in errors)


def test_detects_trunk_compensation_with_large_lean(elbow_flexion_defn):
    detector = ErrorDetector(fps=30, trunk_lean_threshold_deg=15.0)
    lm = _basic_landmarks()
    lm[MEDIAPIPE_LANDMARK_INDEX["left_shoulder"]] = [0.7, 0.2, 0]
    lm[MEDIAPIPE_LANDMARK_INDEX["right_shoulder"]] = [0.9, 0.2, 0]
    angles = compute_all_joint_angles(lm)
    phase_result = PhaseResult(phase=ExercisePhase.MOVING_TO_TARGET, progress_fraction=0.5, current_angle_deg=100.0, velocity_deg_per_sec=10.0)
    errors = detector.detect(lm, angles, elbow_flexion_defn, phase_result)
    assert any(e.error_type == ErrorType.TRUNK_COMPENSATION for e in errors)


def test_symmetry_check_only_applies_to_symmetric_exercises(elbow_flexion_defn):
    # elbow_flexion is symmetric=False per configs/exercises.yaml, so no
    # asymmetry check should ever fire for it even with very different sides.
    detector = ErrorDetector(fps=30)
    lm = _basic_landmarks()
    angles = compute_all_joint_angles(lm)
    angles["left_elbow_angle"] = 45.0
    angles["right_elbow_angle"] = 170.0
    phase_result = PhaseResult(phase=ExercisePhase.PEAK, progress_fraction=1.0, current_angle_deg=45.0, velocity_deg_per_sec=10.0)
    errors = detector.detect(lm, angles, elbow_flexion_defn, phase_result)
    assert not any(e.error_type == ErrorType.ASYMMETRICAL_MOTION for e in errors)


def test_incorrect_sequence_detected_when_mismatched(elbow_flexion_defn):
    detector = ErrorDetector(fps=30)
    lm = _basic_landmarks()
    angles = compute_all_joint_angles(lm)
    phase_result = PhaseResult(phase=ExercisePhase.NEUTRAL, progress_fraction=0.0, current_angle_deg=170.0, velocity_deg_per_sec=0.0)
    errors = detector.detect(
        lm, angles, elbow_flexion_defn, phase_result,
        recognized_exercise_key="elbow_flexion", expected_exercise_key="shoulder_flexion",
    )
    assert any(e.error_type == ErrorType.INCORRECT_SEQUENCE for e in errors)
