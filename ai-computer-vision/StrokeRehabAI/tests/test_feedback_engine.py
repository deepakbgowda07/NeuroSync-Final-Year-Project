"""Tests for inference.feedback_engine.FeedbackEngine (natural language feedback)."""

import random

import pytest

from configs.config_loader import load_config
from inference.error_detector import DetectedError, ErrorType
from inference.exercise_library import ExerciseLibrary
from inference.feedback_engine import FeedbackEngine, _PHRASE_BANK, _POSITIVE_PHRASES, _ENCOURAGEMENT_PHRASES
from inference.movement_analyzer import MovementAnalysisResult
from inference.phase_detector import ExercisePhase
from inference.rep_tracker import RepEvent


@pytest.fixture
def defn():
    cfg = load_config(force_reload=True)
    return ExerciseLibrary(cfg.exercises).get("elbow_flexion")


def _base_result(**overrides):
    defaults = dict(
        exercise_key="elbow_flexion", exercise_display_name="Elbow Flexion",
        exercise_recognition_confidence=0.9, phase=ExercisePhase.PEAK,
        expected_angle_deg=45, actual_angle_deg=100, progress_fraction=0.7,
        completion_percentage=70, movement_quality=0.7, overall_confidence=0.9,
        rep_count=0, errors=[],
    )
    defaults.update(overrides)
    return MovementAnalysisResult(**defaults)


def test_no_bare_wrong_or_correct_in_any_phrase_bank_entry():
    all_phrases = _POSITIVE_PHRASES + _ENCOURAGEMENT_PHRASES
    for phrases in _PHRASE_BANK.values():
        all_phrases.extend(phrases)

    for phrase in all_phrases:
        lowered = phrase.lower().replace("{direction}", "reach").replace("{joint}", "arm")
        assert lowered.strip(".") != "wrong"
        assert lowered.strip(".") != "correct"
        # also guard against the phrase being *only* those words with punctuation
        assert "wrong" not in lowered.split()
        assert "correct" not in lowered.split()


def test_generates_message_for_each_error(defn):
    engine = FeedbackEngine(rng=random.Random(0))
    result = _base_result(errors=[DetectedError(ErrorType.INSUFFICIENT_ROM, "moderate", {})])
    messages = engine.generate(result, defn)
    assert len(messages) >= 1
    assert any(m.error_type == ErrorType.INSUFFICIENT_ROM for m in messages)


def test_positive_feedback_on_completed_rep_with_no_errors(defn):
    engine = FeedbackEngine(rng=random.Random(0))
    result = _base_result(errors=[], rep_event=RepEvent(rep_number=1, completed=True, peak_progress_fraction=1.0, duration_frames=20))
    messages = engine.generate(result, defn)
    assert any(m.text in _POSITIVE_PHRASES for m in messages)


def test_low_confidence_triggers_visibility_warning(defn):
    engine = FeedbackEngine(low_confidence_threshold=0.5, rng=random.Random(0))
    result = _base_result(overall_confidence=0.2, errors=[])
    messages = engine.generate(result, defn)
    assert any("visible" in m.text.lower() for m in messages)


def test_no_exercise_prompts_to_begin():
    engine = FeedbackEngine(rng=random.Random(0))
    result = _base_result(exercise_key=None, exercise_display_name=None, phase=None)
    messages = engine.generate(result, None)
    assert len(messages) == 1
    assert "begin" in messages[0].text.lower()
