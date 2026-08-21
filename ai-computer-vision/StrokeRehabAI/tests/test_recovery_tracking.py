"""Tests for dashboard.recovery_tracking.RecoveryTracker."""

import json
from datetime import datetime, timedelta

import pytest

from dashboard.db import get_connection, init_db
from dashboard.recovery_tracking import RecoveryTracker


@pytest.fixture
def db_path(tmp_path):
    path = str(tmp_path / "test.db")
    init_db(path)
    with get_connection(path) as conn:
        conn.execute("INSERT INTO patients (patient_id, full_name) VALUES (1, 'Jane Doe')")
    return path


def _seed_sessions(db_path, qualities_with_days_ago):
    with get_connection(db_path) as conn:
        for i, (quality, days_ago) in enumerate(qualities_with_days_ago):
            sid = i + 1
            started = (datetime.now() - timedelta(days=days_ago)).isoformat()
            conn.execute(
                "INSERT INTO sessions (session_id, patient_id, exercise_name, started_at, duration_seconds, total_reps) "
                "VALUES (?, 1, 'Elbow Flexion', ?, 120, 6)",
                (sid, started),
            )
            for f in range(20):
                conn.execute(
                    "INSERT INTO session_frames (session_id, movement_quality, rom_deg, joint_angles_json) VALUES (?, ?, ?, ?)",
                    (sid, quality, 100 + f, json.dumps({"left_elbow_angle": 100 + f})),
                )


def test_insufficient_data_for_single_session(db_path):
    _seed_sessions(db_path, [(0.7, 1)])
    tracker = RecoveryTracker(db_path)
    report = tracker.track(patient_id=1)
    assert report.recovery_trend == "insufficient_data"


def test_improving_scores_detected_as_improving(db_path):
    _seed_sessions(db_path, [(0.3, 20), (0.5, 15), (0.7, 10), (0.9, 5)])
    tracker = RecoveryTracker(db_path)
    report = tracker.track(patient_id=1)
    assert report.recovery_trend == "improving"
    assert report.recovery_velocity_per_week > 0


def test_declining_scores_detected_as_declining(db_path):
    _seed_sessions(db_path, [(0.9, 20), (0.7, 15), (0.5, 10), (0.3, 5)])
    tracker = RecoveryTracker(db_path)
    report = tracker.track(patient_id=1)
    assert report.recovery_trend == "declining"
    assert report.recovery_velocity_per_week < 0


def test_all_scores_are_finite(db_path):
    _seed_sessions(db_path, [(0.4, 20), (0.6, 10), (0.8, 2)])
    tracker = RecoveryTracker(db_path)
    report = tracker.track(patient_id=1)
    d = report.to_dict()
    for key in ("consistency_score", "compliance_score", "recovery_velocity_per_week"):
        assert d[key] is not None

def test_consistency_score_high_for_stable_performance(db_path):
    _seed_sessions(db_path, [(0.7, 15), (0.71, 10), (0.69, 5)])
    tracker = RecoveryTracker(db_path)
    report = tracker.track(patient_id=1)
    assert report.consistency_score > 80.0


def test_report_includes_non_diagnostic_note(db_path):
    _seed_sessions(db_path, [(0.5, 10), (0.6, 5)])
    tracker = RecoveryTracker(db_path)
    report = tracker.track(patient_id=1)
    assert "not a medical diagnosis" in report.note.lower()
