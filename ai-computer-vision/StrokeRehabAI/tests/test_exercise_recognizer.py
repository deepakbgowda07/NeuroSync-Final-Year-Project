"""Tests for inference.exercise_recognizer.ExerciseRecognizer."""

import numpy as np
import pytest

from configs.config_loader import load_config
from inference.exercise_library import ExerciseLibrary
from inference.exercise_recognizer import ExerciseRecognizer


@pytest.fixture
def library():
    cfg = load_config(force_reload=True)
    return ExerciseLibrary(cfg.exercises)


def test_recognizes_elbow_flexion_from_angle_trajectory(library):
    recognizer = ExerciseRecognizer(library, confidence_window_frames=10, min_recognition_confidence=0.5)
    result = None
    for angle in np.linspace(170, 45, 40):
        result = recognizer.update({"left_elbow_angle": angle, "right_elbow_angle": 170.0})
    assert result.exercise_key == "elbow_flexion"
    assert result.confidence > 0.5


def test_no_recognition_with_flat_angles(library):
    recognizer = ExerciseRecognizer(library, min_recognition_confidence=0.9)
    result = None
    for _ in range(30):
        result = recognizer.update({"left_elbow_angle": 170.0, "right_elbow_angle": 170.0})
    assert result.exercise_key is None


def test_reset_clears_history(library):
    recognizer = ExerciseRecognizer(library)
    for angle in np.linspace(170, 45, 20):
        recognizer.update({"left_elbow_angle": angle})
    recognizer.reset()
    result = recognizer.update({"left_elbow_angle": 170.0})
    assert result.confidence <= 1.0  # fresh state, no crash


def test_candidate_scores_returned_for_all_exercises(library):
    recognizer = ExerciseRecognizer(library)
    result = recognizer.update({"left_elbow_angle": 100.0})
    assert set(result.candidate_scores.keys()) == set(library.keys())
