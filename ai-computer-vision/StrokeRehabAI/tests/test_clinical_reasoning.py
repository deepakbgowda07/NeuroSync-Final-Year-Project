"""Tests for inference.clinical_reasoning.ClinicalReasoningEngine."""

import pytest

from configs.config_loader import load_config
from inference.clinical_assessment import RepetitionAssessment
from inference.clinical_reasoning import ClinicalReasoningEngine
from inference.error_detector import DetectedError, ErrorType
from inference.exercise_library import ExerciseLibrary


@pytest.fixture
def library():
    cfg = load_config(force_reload=True)
    return ExerciseLibrary(cfg.exercises)


@pytest.fixture
def engine():
    return ClinicalReasoningEngine()


def _good_assessment(**overrides):
    defaults = dict(
        exercise_quality_score=90, range_of_motion_score=90, movement_smoothness_score=90,
        joint_stability_score=90, movement_consistency_score=90, compensation_severity_score=90,
        exercise_completion_score=90, overall_rehabilitation_score=90,
    )
    defaults.update(overrides)
    return RepetitionAssessment(**defaults)


def test_no_rules_fire_for_good_repetition(engine, library):
    defn = library.get("elbow_flexion")
    results = engine.evaluate(defn, _good_assessment(), errors=[], achieved_angle_deg=45, peak_velocity_deg_per_sec=50)
    assert results == []


def test_rom_deficit_rule_fires_with_recommendation_for_shoulder(engine, library):
    defn = library.get("shoulder_flexion")
    assessment = _good_assessment(range_of_motion_score=40.0)
    results = engine.evaluate(defn, assessment, errors=[])
    rom_results = [r for r in results if r.rule_id == "rom_deficit"]
    assert len(rom_results) == 1
    assert "shoulder flexion" in rom_results[0].recommendation.lower()
    assert "mobility" in rom_results[0].recommendation.lower()


def test_trunk_compensation_rule_fires_on_trunk_error(engine, library):
    defn = library.get("shoulder_flexion")
    errors = [DetectedError(ErrorType.TRUNK_COMPENSATION, "moderate", {"trunk_lean_deg": 22})]
    results = engine.evaluate(defn, _good_assessment(), errors=errors)
    trunk_results = [r for r in results if r.rule_id == "trunk_compensation"]
    assert len(trunk_results) == 1
    assert "upright" in trunk_results[0].recommendation.lower()
    assert "22" in trunk_results[0].condition


def test_elbow_extension_limited_rule_fires_only_for_that_exercise(engine, library):
    elbow_ext = library.get("elbow_extension")
    results = engine.evaluate(elbow_ext, _good_assessment(), errors=[], achieved_angle_deg=100)
    ext_results = [r for r in results if r.rule_id == "elbow_extension_limited"]
    assert len(ext_results) == 1

    elbow_flex = library.get("elbow_flexion")
    results2 = engine.evaluate(elbow_flex, _good_assessment(), errors=[], achieved_angle_deg=100)
    assert not any(r.rule_id == "elbow_extension_limited" for r in results2)


def test_fast_movement_rule_fires_above_threshold(engine, library):
    defn = library.get("elbow_flexion")
    results = engine.evaluate(defn, _good_assessment(), errors=[], peak_velocity_deg_per_sec=300)
    assert any(r.rule_id == "fast_movement" for r in results)


def test_low_smoothness_rule_fires(engine, library):
    defn = library.get("elbow_flexion")
    assessment = _good_assessment(movement_smoothness_score=30.0)
    results = engine.evaluate(defn, assessment, errors=[])
    assert any(r.rule_id == "low_smoothness" for r in results)


def test_compensation_severity_rule_fires(engine, library):
    defn = library.get("elbow_flexion")
    assessment = _good_assessment(compensation_severity_score=40.0)
    results = engine.evaluate(defn, assessment, errors=[])
    assert any(r.rule_id == "compensation_severity" for r in results)


def test_every_result_has_condition_recommendation_and_explanation(engine, library):
    defn = library.get("shoulder_flexion")
    assessment = _good_assessment(range_of_motion_score=30.0, movement_smoothness_score=30.0)
    results = engine.evaluate(defn, assessment, errors=[])
    assert len(results) >= 2
    for r in results:
        assert r.condition
        assert r.recommendation
        assert r.explanation
