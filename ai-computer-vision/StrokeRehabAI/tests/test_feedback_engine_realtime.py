"""Tests for inference.feedback_engine.FeedbackEngine (natural language feedback)."""

import random

from configs.config_loader import load_config
from inference.error_detector import DetectedError, ErrorType
from inference.exercise_library import ExerciseLibrary
from inference.feedback_engine import FeedbackEngine
from inference.movement_analyzer import MovementAnalysisResult
from inference.phase_detector import ExercisePhase
from inference.rep_tracker import RepEvent


def _base_result(**overrides):
    defaults = dict(
        exercise_key="elbow_flexion", exercise_display_name="Elbow Flexion",
        exercise_recognition_confidence=0.9, phase=ExercisePhase.PEAK,
        expected_angle_deg=45, actual_angle_deg=45, progress_fraction=1.0,
        completion_percentage=100, movement_quality=1.0, overall_confidence=0.9,
        rep_count=1, errors=[],
    )
    defaults.update(overrides)
    return MovementAnalysisResult(**defaults)


def _defn():
    cfg = load_config(force_reload=True)
    return ExerciseLibrary(cfg.exercises).get("elbow_flexion")


def test_never_uses_bare_wrong_or_correct():
    from inference.feedback_engine import _PHRASE_BANK, _POSITIVE_PHRASES, _ENCOURAGEMENT_PHRASES

    all_phrases = [p for bank in _PHRASE_BANK.values() for p in bank] + _POSITIVE_PHRASES + _ENCOURAGEMENT_PHRASES
    for phrase in all_phrases:
        normalized = phrase.strip().rstrip(".").lower()
        assert normalized not in ("wrong", "correct")


def test_generates_positive_feedback_on_completed_rep_with_no_errors():
    engine = FeedbackEngine(rng=random.Random(0))
    result = _base_result(rep_event=RepEvent(rep_number=1, completed=True, peak_progress_fraction=1.0, duration_frames=20))
    messages = engine.generate(result, _defn())
    assert any(m.severity == "info" for m in messages)


def test_generates_warning_for_active_error():
    engine = FeedbackEngine(rng=random.Random(0))
    result = _base_result(errors=[DetectedError(ErrorType.FAST_MOVEMENT, "minor", {})])
    messages = engine.generate(result, _defn())
    assert any(m.error_type == ErrorType.FAST_MOVEMENT for m in messages)
    assert all(m.text.strip().lower() not in ("wrong", "incorrect") for m in messages)


def test_no_exercise_recognized_gives_generic_prompt():
    engine = FeedbackEngine()
    result = _base_result(exercise_key=None)
    messages = engine.generate(result, None)
    assert len(messages) == 1
    assert messages[0].severity == "info"


def test_low_confidence_adds_visibility_warning():
    engine = FeedbackEngine(low_confidence_threshold=0.6)
    result = _base_result(overall_confidence=0.2)
    messages = engine.generate(result, _defn())
    assert any("visible" in m.text.lower() for m in messages)
