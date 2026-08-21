"""Tests for dashboard.recommendation_engine.RecommendationEngine."""

from dashboard.analytics import RecoveryAnalytics
from dashboard.recommendation_engine import RecommendationEngine


def _analytics(**overrides):
    defaults = dict(
        patient_id=1, num_sessions=5, average_exercise_accuracy=90.0,
        average_rom_deg=100.0, max_rom_deg=110.0, min_rom_deg=90.0,
        average_joint_angle_error_deg=5.0, exercise_completion_rate=90.0,
        movement_smoothness=90.0, compensation_frequency=1.0, exercise_consistency=90.0,
        average_session_duration_seconds=180.0, fatigue_indicator=10.0,
        recovery_score=80.0, improvement_percentage=10.0, trend="improving",
    )
    defaults.update(overrides)
    return RecoveryAnalytics(**defaults)


def test_no_sessions_returns_general_recommendation():
    engine = RecommendationEngine()
    recs = engine.generate(_analytics(num_sessions=0))
    assert len(recs) == 1
    assert recs[0].category == "general"


def test_good_performance_returns_stable_recommendation():
    engine = RecommendationEngine()
    recs = engine.generate(_analytics())
    assert any("stable" in r.text.lower() or "continue" in r.text.lower() for r in recs)


def test_low_rom_triggers_high_priority_recommendation():
    engine = RecommendationEngine()
    recs = engine.generate(_analytics(average_rom_deg=40.0, max_rom_deg=110.0), exercise_display_name="Shoulder Flexion")
    rom_recs = [r for r in recs if r.category == "rom"]
    assert any(r.priority == "high" for r in rom_recs)
    assert any("Shoulder Flexion" in r.text for r in rom_recs)


def test_high_compensation_triggers_recommendation():
    engine = RecommendationEngine()
    recs = engine.generate(_analytics(compensation_frequency=10.0))
    assert any(r.category == "compensation" for r in recs)


def test_low_completion_rate_triggers_recommendation():
    engine = RecommendationEngine()
    recs = engine.generate(_analytics(exercise_completion_rate=40.0))
    assert any(r.category == "quality" and r.priority == "high" for r in recs)


def test_low_smoothness_triggers_speed_recommendation():
    engine = RecommendationEngine()
    recs = engine.generate(_analytics(movement_smoothness=30.0))
    assert any(r.category == "speed" for r in recs)


def test_declining_trend_triggers_rom_recommendation():
    engine = RecommendationEngine()
    recs = engine.generate(_analytics(trend="declining"))
    assert any(r.category == "rom" for r in recs)
