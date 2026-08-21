"""Tests for dashboard.clinical_metrics.ClinicalMetricsCalculator."""

import numpy as np
import pytest

from dashboard.clinical_metrics import ClinicalMetricsCalculator


@pytest.fixture
def calculator():
    return ClinicalMetricsCalculator()


def _smooth_elbow_frames(quality=0.9):
    frames = []
    for angle in list(np.linspace(170, 45, 30)) + list(np.linspace(45, 170, 30)):
        frames.append({
            "joint_angles": {"left_elbow_angle": angle, "right_elbow_angle": 170.0},
            "movement_quality": quality,
        })
    return frames


def test_all_scores_within_0_100(calculator):
    frames = _smooth_elbow_frames()
    scores = calculator.compute_for_session(frames, events=[], reps=[{"completed": True}])
    for name, value in scores.to_dict().items():
        assert 0.0 <= value <= 100.0, f"{name}={value} out of range"


def test_smoothness_high_for_linear_ramp(calculator):
    frames = _smooth_elbow_frames()
    scores = calculator.compute_for_session(frames, events=[], reps=[])
    assert scores.movement_smoothness > 80.0


def test_smoothness_not_corrupted_by_stationary_second_side(calculator):
    """Regression test: right_elbow_angle stays constant while left moves;
    smoothness must reflect the moving side's actual trajectory, not an
    interleaved left/right sequence (a bug found during development)."""
    frames = _smooth_elbow_frames()
    scores = calculator.compute_for_session(frames, events=[], reps=[])
    assert scores.movement_smoothness > 50.0  # would be near 0 if interleaving bug reappears


def test_compensation_score_decreases_with_more_events(calculator):
    frames = _smooth_elbow_frames()
    no_events_score = calculator.compute_for_session(frames, events=[], reps=[]).compensation_score
    many_events = [{"error_type": "trunk_compensation"}] * 10
    with_events_score = calculator.compute_for_session(frames, events=many_events, reps=[]).compensation_score
    assert with_events_score < no_events_score


def test_exercise_quality_reflects_completion_rate(calculator):
    frames = _smooth_elbow_frames(quality=0.9)
    all_complete = calculator.compute_for_session(frames, events=[], reps=[{"completed": True}] * 5).exercise_quality_score
    half_complete = calculator.compute_for_session(frames, events=[], reps=[{"completed": True}] * 2 + [{"completed": False}] * 3).exercise_quality_score
    assert all_complete > half_complete


def test_symmetry_score_perfect_when_sides_equal(calculator):
    frames = []
    for angle in np.linspace(170, 45, 20):
        frames.append({"joint_angles": {"left_elbow_angle": angle, "right_elbow_angle": angle}, "movement_quality": 0.8})
    scores = calculator.compute_for_session(frames, events=[], reps=[])
    assert scores.symmetry_score > 95.0


def test_average_scores_aggregates_correctly():
    dicts = [{"a": 10.0, "b": 20.0}, {"a": 30.0, "b": 40.0}]
    avg = ClinicalMetricsCalculator.average_scores(dicts)
    assert avg["a"] == 20.0
    assert avg["b"] == 30.0


def test_empty_frames_returns_neutral_defaults(calculator):
    scores = calculator.compute_for_session([], events=[], reps=[])
    assert scores.movement_stability == 50.0
    assert scores.shoulder_mobility_score == 0.0
