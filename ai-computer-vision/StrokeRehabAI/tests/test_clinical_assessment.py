"""Tests for inference.clinical_assessment.ClinicalAssessmentEngine."""

import numpy as np
import pytest

from configs.config_loader import load_config
from inference.clinical_assessment import ClinicalAssessmentEngine, RepFrameSample
from inference.error_detector import DetectedError, ErrorType
from inference.exercise_library import ExerciseLibrary


@pytest.fixture
def elbow_flexion_defn():
    cfg = load_config(force_reload=True)
    return ExerciseLibrary(cfg.exercises).get("elbow_flexion")


@pytest.fixture
def engine():
    return ClinicalAssessmentEngine()


def _good_rep_frames(num_frames=30):
    return [
        RepFrameSample(angle_deg=a, movement_quality=0.9, trunk_lean_deg=5.0, shoulder_elevation=0.05, errors=[])
        for a in np.linspace(170, 45, num_frames)
    ]


def test_all_scores_within_0_100(engine, elbow_flexion_defn):
    frames = _good_rep_frames()
    assessment = engine.assess_repetition(frames, elbow_flexion_defn, rep_completed=True, peak_progress_fraction=1.0)
    for name, value in assessment.to_dict().items():
        assert 0.0 <= value <= 100.0, f"{name}={value} out of range"


def test_good_rep_scores_highly(engine, elbow_flexion_defn):
    frames = _good_rep_frames()
    assessment = engine.assess_repetition(frames, elbow_flexion_defn, rep_completed=True, peak_progress_fraction=1.0)
    assert assessment.overall_rehabilitation_score > 85.0


def test_empty_frames_returns_zeroed_assessment(engine, elbow_flexion_defn):
    assessment = engine.assess_repetition([], elbow_flexion_defn, rep_completed=False, peak_progress_fraction=0.0)
    assert assessment.overall_rehabilitation_score == 0.0


def test_incomplete_rep_scores_lower_completion(engine, elbow_flexion_defn):
    frames = _good_rep_frames()
    completed = engine.assess_repetition(frames, elbow_flexion_defn, rep_completed=True, peak_progress_fraction=1.0)
    incomplete = engine.assess_repetition(frames, elbow_flexion_defn, rep_completed=False, peak_progress_fraction=0.3)
    assert completed.exercise_completion_score > incomplete.exercise_completion_score


def test_compensation_errors_reduce_compensation_score(engine, elbow_flexion_defn):
    clean_frames = _good_rep_frames()
    compensated_frames = [
        RepFrameSample(
            angle_deg=f.angle_deg, movement_quality=f.movement_quality, trunk_lean_deg=20.0, shoulder_elevation=0.2,
            errors=[DetectedError(ErrorType.TRUNK_COMPENSATION, "moderate", {})],
        )
        for f in clean_frames
    ]
    clean_score = engine.assess_repetition(clean_frames, elbow_flexion_defn, True, 1.0).compensation_severity_score
    compensated_score = engine.assess_repetition(compensated_frames, elbow_flexion_defn, True, 1.0).compensation_severity_score
    assert compensated_score < clean_score


def test_jerky_movement_reduces_smoothness_score(engine, elbow_flexion_defn):
    smooth_frames = _good_rep_frames()
    rng = np.random.default_rng(0)
    jerky_angles = [a + rng.normal(0, 25) for a in np.linspace(170, 45, 30)]
    jerky_frames = [
        RepFrameSample(angle_deg=a, movement_quality=0.9, trunk_lean_deg=5.0, shoulder_elevation=0.05, errors=[])
        for a in jerky_angles
    ]
    smooth_score = engine.assess_repetition(smooth_frames, elbow_flexion_defn, True, 1.0).movement_smoothness_score
    jerky_score = engine.assess_repetition(jerky_frames, elbow_flexion_defn, True, 1.0).movement_smoothness_score
    assert jerky_score < smooth_score


def test_low_rom_reduces_range_of_motion_score(engine, elbow_flexion_defn):
    full_rom_frames = _good_rep_frames()
    partial_rom_frames = [
        RepFrameSample(angle_deg=a, movement_quality=0.9, trunk_lean_deg=5.0, shoulder_elevation=0.05, errors=[])
        for a in np.linspace(170, 140, 30)  # barely moved
    ]
    full_score = engine.assess_repetition(full_rom_frames, elbow_flexion_defn, True, 1.0).range_of_motion_score
    partial_score = engine.assess_repetition(partial_rom_frames, elbow_flexion_defn, True, 1.0).range_of_motion_score
    assert partial_score < full_score
