"""Tests for dashboard.session_manager.SessionManager."""

from datetime import datetime, timedelta

import pytest

from dashboard.db import get_connection, init_db
from dashboard.session_manager import SessionManager


@pytest.fixture
def db_path(tmp_path):
    path = str(tmp_path / "test.db")
    init_db(path)
    with get_connection(path) as conn:
        conn.execute("INSERT INTO patients (patient_id, full_name) VALUES (1, 'Jane Doe')")
    return path


def _insert_session(db_path, session_id, days_ago, quality, reps, total_reps):
    started = (datetime.now() - timedelta(days=days_ago)).isoformat()
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO sessions (session_id, patient_id, exercise_name, started_at, total_reps, mean_quality, mean_confidence) "
            "VALUES (?, 1, 'Elbow Flexion', ?, ?, ?, ?)",
            (session_id, started, total_reps, quality, quality),
        )
        for i in range(reps):
            conn.execute("INSERT INTO session_reps (session_id, rep_number, completed) VALUES (?, ?, 1)", (session_id, i))


def test_list_sessions_returns_newest_first(db_path):
    _insert_session(db_path, 1, days_ago=5, quality=0.5, reps=3, total_reps=3)
    _insert_session(db_path, 2, days_ago=1, quality=0.8, reps=5, total_reps=5)
    manager = SessionManager(db_path)
    sessions = manager.list_sessions(patient_id=1)
    assert sessions[0].session_id == 2
    assert sessions[1].session_id == 1


def test_get_session_detail_counts_reps(db_path):
    _insert_session(db_path, 1, days_ago=1, quality=0.8, reps=4, total_reps=4)
    manager = SessionManager(db_path)
    detail = manager.get_session_detail(1)
    assert detail["successful_repetitions"] == 4
    assert detail["incorrect_repetitions"] == 0


def test_get_session_detail_unknown_session_raises(db_path):
    manager = SessionManager(db_path)
    with pytest.raises(ValueError):
        manager.get_session_detail(999)


def test_compare_sessions_previous_session_detects_improvement(db_path):
    _insert_session(db_path, 1, days_ago=5, quality=0.5, reps=3, total_reps=5)
    _insert_session(db_path, 2, days_ago=1, quality=0.9, reps=5, total_reps=5)
    manager = SessionManager(db_path)
    result = manager.compare_sessions(2, comparison="previous_session")
    assert result["baseline"]["session_id"] == 1
    assert result["verdicts"]["quality"] == "improvement"


def test_compare_sessions_no_baseline_available(db_path):
    _insert_session(db_path, 1, days_ago=1, quality=0.8, reps=5, total_reps=5)
    manager = SessionManager(db_path)
    result = manager.compare_sessions(1, comparison="previous_session")
    assert result["baseline"] is None


def test_get_recent_sessions_in_window(db_path):
    _insert_session(db_path, 1, days_ago=40, quality=0.5, reps=3, total_reps=5)
    _insert_session(db_path, 2, days_ago=2, quality=0.8, reps=5, total_reps=5)
    manager = SessionManager(db_path)
    recent = manager.get_recent_sessions_in_window(patient_id=1, days=7)
    assert len(recent) == 1
    assert recent[0].session_id == 2
