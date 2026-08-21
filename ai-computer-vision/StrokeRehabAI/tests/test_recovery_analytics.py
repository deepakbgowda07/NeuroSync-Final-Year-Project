"""Tests for dashboard.analytics.RecoveryAnalyticsEngine."""

import json
from datetime import datetime, timedelta

import numpy as np
import pytest

from dashboard.analytics import RecoveryAnalyticsEngine
from dashboard.db import get_connection, init_db


@pytest.fixture
def db_path(tmp_path):
    path = str(tmp_path / "test.db")
    init_db(path)
    with get_connection(path) as conn:
        conn.execute("INSERT INTO patients (patient_id, full_name) VALUES (1, 'Jane Doe')")
    return path


def _seed_improving_sessions(db_path, num_sessions=4):
    with get_connection(db_path) as conn:
        for i in range(num_sessions):
            sid = i + 1
            quality = 0.4 + i * 0.15  # improving over time
            started = (datetime.now() - timedelta(days=(num_sessions - i) * 3)).isoformat()
            conn.execute(
                "INSERT INTO sessions (session_id, patient_id, exercise_name, started_at, duration_seconds, total_reps) "
                "VALUES (?, 1, 'Elbow Flexion', ?, 120, 5)",
                (sid, started),
            )
            for f in range(20):
                angle = 170 - f * (2 + i)  # increasing ROM over sessions
                conn.execute(
                    "INSERT INTO session_frames (session_id, movement_quality, rom_deg, joint_angles_json) VALUES (?, ?, ?, ?)",
                    (sid, quality, angle, json.dumps({"left_elbow_angle": angle})),
                )
            for r in range(5):
                conn.execute("INSERT INTO session_reps (session_id, rep_number, completed) VALUES (?, ?, 1)", (sid, r))


def test_empty_patient_returns_zeroed_analytics(db_path):
    engine = RecoveryAnalyticsEngine(db_path)
    analytics = engine.compute_patient_analytics(patient_id=1)
    assert analytics.num_sessions == 0
    assert analytics.trend == "insufficient_data"


def test_improving_sessions_detected_as_improving_trend(db_path):
    _seed_improving_sessions(db_path)
    engine = RecoveryAnalyticsEngine(db_path)
    analytics = engine.compute_patient_analytics(patient_id=1)
    assert analytics.trend == "improving"
    assert analytics.improvement_percentage > 0


def test_all_metrics_are_finite_and_non_negative(db_path):
    _seed_improving_sessions(db_path)
    engine = RecoveryAnalyticsEngine(db_path)
    analytics = engine.compute_patient_analytics(patient_id=1)
    for key, value in analytics.to_dict().items():
        if isinstance(value, (int, float)):
            assert np.isfinite(value), f"{key} is not finite: {value}"


def test_save_session_scores_persists_recovery_scores(db_path):
    _seed_improving_sessions(db_path, num_sessions=1)
    engine = RecoveryAnalyticsEngine(db_path)
    scores = engine.save_session_scores(session_id=1, patient_id=1)
    assert "overall_rehabilitation_score" in scores

    with get_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM recovery_scores WHERE session_id = 1").fetchone()
    assert row is not None


def test_save_joint_metrics_persists_rom_data(db_path):
    _seed_improving_sessions(db_path, num_sessions=1)
    engine = RecoveryAnalyticsEngine(db_path)
    rows = engine.save_joint_metrics(session_id=1)
    assert len(rows) > 0
    assert all(r["rom_deg"] >= 0 for r in rows)

    with get_connection(db_path) as conn:
        db_rows = conn.execute("SELECT * FROM joint_metrics WHERE session_id = 1").fetchall()
    assert len(db_rows) == len(rows)


def test_save_joint_metrics_computes_angle_error_with_targets(db_path):
    _seed_improving_sessions(db_path, num_sessions=1)
    engine = RecoveryAnalyticsEngine(db_path)
    rows = engine.save_joint_metrics(session_id=1, target_angles={"left_elbow_angle": 45.0})
    left_elbow_row = next(r for r in rows if r["joint_name"] == "left_elbow_angle")
    assert left_elbow_row["angle_error_deg"] is not None
