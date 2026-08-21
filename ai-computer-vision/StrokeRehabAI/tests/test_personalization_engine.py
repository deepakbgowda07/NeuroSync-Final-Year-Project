"""Tests for dashboard.personalization_engine.PersonalizationEngine."""

import json
from datetime import datetime, timedelta

import pytest

from dashboard.db import get_connection, init_db
from dashboard.personalization_engine import PersonalizationEngine


@pytest.fixture
def db_path(tmp_path):
    path = str(tmp_path / "test.db")
    init_db(path)
    with get_connection(path) as conn:
        conn.execute("INSERT INTO patients (patient_id, full_name) VALUES (1, 'Jane Doe')")
    return path


def _insert_session(db_path, session_id, exercise_name, quality, days_ago, compensation_events=0):
    started = (datetime.now() - timedelta(days=days_ago)).isoformat()
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO sessions (session_id, patient_id, exercise_name, started_at, duration_seconds, total_reps) "
            "VALUES (?, 1, ?, ?, 120, 6)",
            (session_id, exercise_name, started),
        )
        for f in range(20):
            conn.execute(
                "INSERT INTO session_frames (session_id, movement_quality, rom_deg, joint_angles_json) VALUES (?, ?, ?, ?)",
                (session_id, quality, 100 + f, json.dumps({"left_elbow_angle": 100 + f})),
            )
        for _ in range(compensation_events):
            conn.execute(
                "INSERT INTO session_events (session_id, event_type, error_type, severity) VALUES (?, 'error', 'trunk_compensation', 'moderate')",
                (session_id,),
            )


def test_no_sessions_returns_conservative_starter_plan(db_path):
    engine = PersonalizationEngine(db_path)
    plan = engine.build_plan(patient_id=1)
    assert plan.suggested_repetitions <= 8
    assert plan.suggested_sets <= 3
    assert len(plan.rationale) > 0


def test_consistently_good_exercise_is_flagged_for_reduction(db_path):
    for i in range(4):
        _insert_session(db_path, i + 1, "Elbow Flexion", quality=0.92, days_ago=(4 - i) * 2)
    engine = PersonalizationEngine(db_path)
    plan = engine.build_plan(patient_id=1)
    assert "Elbow Flexion" in plan.exercises_to_reduce


def test_consistently_poor_exercise_is_prioritized(db_path):
    for i in range(3):
        _insert_session(db_path, i + 1, "Shoulder Flexion", quality=0.4, days_ago=(3 - i) * 2, compensation_events=5)
    engine = PersonalizationEngine(db_path)
    plan = engine.build_plan(patient_id=1)
    assert "Shoulder Flexion" in plan.exercises_to_prioritize


def test_high_compensation_triggers_focus_area(db_path):
    for i in range(3):
        _insert_session(db_path, i + 1, "Shoulder Flexion", quality=0.6, days_ago=(3 - i) * 2, compensation_events=8)
    engine = PersonalizationEngine(db_path)
    plan = engine.build_plan(patient_id=1)
    assert any("compensat" in area.lower() for area in plan.recovery_focus_areas)


def test_plan_never_prescribes_medical_treatment_language(db_path):
    """Personalization is exercise-practice guidance only, never a medical
    diagnosis or treatment decision."""
    for i in range(3):
        _insert_session(db_path, i + 1, "Elbow Flexion", quality=0.5, days_ago=(3 - i) * 2)
    engine = PersonalizationEngine(db_path)
    plan = engine.build_plan(patient_id=1)
    full_text = " ".join(plan.rationale + plan.recovery_focus_areas).lower()
    for forbidden_word in ("diagnos", "prescri", "medication", "treatment plan"):
        assert forbidden_word not in full_text
