"""Tests for inference.explainable_ai.ExplainableAIEngine."""

import pytest

from configs.config_loader import load_config
from inference.clinical_assessment import RepetitionAssessment
from inference.error_detector import DetectedError, ErrorType
from inference.exercise_library import ExerciseLibrary
from inference.explainable_ai import ExplainableAIEngine
from inference.movement_analyzer import MovementAnalysisResult
from inference.phase_detector import ExercisePhase


@pytest.fixture
def library():
    cfg = load_config(force_reload=True)
    return ExerciseLibrary(cfg.exercises)


@pytest.fixture
def engine():
    return ExplainableAIEngine()


def _result(**overrides):
    defaults = dict(
        exercise_key="shoulder_flexion", exercise_display_name="Shoulder Flexion",
        exercise_recognition_confidence=0.97, phase=ExercisePhase.MOVING_TO_TARGET,
        expected_angle_deg=150, actual_angle_deg=135, progress_fraction=0.85,
        completion_percentage=85, movement_quality=0.84, overall_confidence=0.9,
        rep_count=2, errors=[],
    )
    defaults.update(overrides)
    return MovementAnalysisResult(**defaults)


def test_no_exercise_recognized_gives_begin_message(engine):
    result = _result(exercise_key=None, exercise_display_name=None, phase=None)
    explanation = engine.explain(result, None)
    assert explanation.exercise == "Unknown"
    assert "begin" in explanation.recommendation.lower()


def test_never_emits_bare_wrong_or_correct(engine, library):
    defn = library.get("shoulder_flexion")
    result = _result(errors=[DetectedError(ErrorType.TRUNK_COMPENSATION, "moderate", {"trunk_lean_deg": 18})])
    assessment = RepetitionAssessment(84, 78, 80, 85, 82, 65, 85, 80)
    explanation = engine.explain(result, defn, assessment=assessment, peak_velocity_deg_per_sec=80)

    full_text = explanation.to_display_string().lower()
    assert "wrong" not in full_text.split()
    assert full_text.split().count("correct") == 0


def test_explanation_contains_all_required_fields(engine, library):
    defn = library.get("shoulder_flexion")
    result = _result()
    explanation = engine.explain(result, defn)
    d = explanation.to_dict()
    assert set(d.keys()) == {"exercise", "confidence", "movement_quality", "reason", "recommendation"}
    assert d["confidence"] == "97%"
    assert d["movement_quality"] == "84%"


def test_reasoning_trace_condition_appears_in_reason(engine, library):
    defn = library.get("shoulder_flexion")
    result = _result(errors=[DetectedError(ErrorType.TRUNK_COMPENSATION, "moderate", {"trunk_lean_deg": 18})])
    assessment = RepetitionAssessment(84, 78, 80, 85, 82, 65, 85, 80)
    explanation = engine.explain(result, defn, assessment=assessment)
    assert any("trunk lean" in r.lower() for r in explanation.reasons)


def test_good_completed_rep_gives_positive_explanation(engine, library):
    defn = library.get("elbow_flexion")
    result = _result(exercise_key="elbow_flexion", exercise_display_name="Elbow Flexion", errors=[])
    assessment = RepetitionAssessment(95, 95, 95, 95, 95, 100, 100, 96)
    explanation = engine.explain(result, defn, assessment=assessment)
    assert "excellent" in explanation.recommendation.lower() or "completed" in explanation.reasons[0].lower()


def test_display_string_matches_required_format(engine, library):
    defn = library.get("shoulder_flexion")
    result = _result()
    explanation = engine.explain(result, defn)
    text = explanation.to_display_string()
    assert text.startswith("Exercise: Shoulder Flexion")
    assert "Confidence: 97%" in text
    assert "Movement Quality: 84%" in text
    assert "Reason:" in text
    assert "Recommendation:" in text
