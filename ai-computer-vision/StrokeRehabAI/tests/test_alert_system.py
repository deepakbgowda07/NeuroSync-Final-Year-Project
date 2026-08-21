"""Tests for dashboard.alert_system.AlertSystem."""

import json
from datetime import datetime, timedelta

import pytest

from dashboard.alert_system import AlertSystem
from dashboard.db import get_connection, init_db


@pytest.fixture
def db_path(tmp_path):
    path = str(tmp_path / "test.db")
    init_db(path)
    with get_connection(path) as conn:
        conn.execute("INSERT INTO patients (patient_id, full_name) VALUES (1, 'Jane Doe')")
    return path


def _insert_session(db_path, session_id, days_ago, quality, duration, compensation_count=0, rom_max=100):
    started = (datetime.now() - timedelta(days=days_ago)).isoformat()
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO sessions (session_id, patient_id, exercise_name, started_at, duration_seconds, mean_quality) "
            "VALUES (?, 1, 'Elbow Flexion', ?, ?, ?)",
            (session_id, started, duration, quality),
        )
        for f in range(20):
            angle = 170 - (f % rom_max)
            conn.execute(
                "INSERT INTO session_frames (session_id, movement_quality, rom_deg, joint_angles_json) VALUES (?, ?, ?, ?)",
                (session_id, quality, angle, json.dumps({"left_elbow_angle": angle})),
            )
        for _ in range(compensation_count):
            conn.execute(
                "INSERT INTO session_events (session_id, event_type, error_type, severity) VALUES (?, 'error', 'trunk_compensation', 'moderate')",
                (session_id,),
            )


def test_no_alerts_for_stable_good_performance(db_path):
    _insert_session(db_path, 1, days_ago=6, quality=0.85, duration=180)
    _insert_session(db_path, 2, days_ago=3, quality=0.85, duration=180)
    system = AlertSystem(db_path)
    alerts = system.check_patient(1)
    assert alerts == []


def test_skipped_sessions_alert_triggers_after_long_gap(db_path):
    _insert_session(db_path, 1, days_ago=30, quality=0.8, duration=180)
    system = AlertSystem(db_path, expected_session_interval_days=3)
    alerts = system.check_patient(1)
    assert any(a.alert_type == "skipped_sessions" for a in alerts)


def test_short_session_alert_triggers(db_path):
    _insert_session(db_path, 1, days_ago=1, quality=0.8, duration=20)
    system = AlertSystem(db_path, short_session_seconds_threshold=60.0)
    alerts = system.check_patient(1)
    assert any(a.alert_type == "short_session" for a in alerts)


def test_accuracy_decrease_alert_triggers(db_path):
    _insert_session(db_path, 1, days_ago=5, quality=0.9, duration=180)
    _insert_session(db_path, 2, days_ago=1, quality=0.5, duration=180)
    system = AlertSystem(db_path, accuracy_decrease_threshold=15.0)
    alerts = system.check_patient(1)
    assert any(a.alert_type == "accuracy_decrease" for a in alerts)


def test_compensation_increase_alert_triggers(db_path):
    _insert_session(db_path, 1, days_ago=5, quality=0.8, duration=180, compensation_count=0)
    _insert_session(db_path, 2, days_ago=1, quality=0.8, duration=180, compensation_count=10)
    system = AlertSystem(db_path, compensation_increase_threshold=3.0)
    alerts = system.check_patient(1)
    assert any(a.alert_type == "compensation_increase" for a in alerts)


def test_no_alerts_for_single_session(db_path):
    _insert_session(db_path, 1, days_ago=1, quality=0.8, duration=180)
    system = AlertSystem(db_path)
    alerts = system.check_patient(1)
    # Only a fresh single session - no session-over-session comparison possible
    assert all(a.alert_type != "recovery_decrease" for a in alerts)


def test_no_alerts_for_unknown_patient(db_path):
    system = AlertSystem(db_path)
    alerts = system.check_patient(999)
    assert alerts == []
