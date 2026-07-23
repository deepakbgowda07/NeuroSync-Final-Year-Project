"""Tests for inference.session_logger.SessionLogger (dashboard communication)."""

import sqlite3

import pytest

from inference.error_detector import DetectedError, ErrorType
from inference.movement_analyzer import MovementAnalysisResult
from inference.phase_detector import ExercisePhase
from inference.rep_tracker import RepEvent
from inference.session_logger import SessionLogger


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test_session.db")


def _result(**overrides):
    defaults = dict(
        exercise_key="elbow_flexion", exercise_display_name="Elbow Flexion",
        exercise_recognition_confidence=0.9, phase=ExercisePhase.PEAK,
        expected_angle_deg=45, actual_angle_deg=50, progress_fraction=0.95,
        completion_percentage=95, movement_quality=0.85, overall_confidence=0.88,
        rep_count=1, errors=[],
    )
    defaults.update(overrides)
    return MovementAnalysisResult(**defaults)


def test_start_session_creates_row(db_path):
    logger = SessionLogger(db_path, patient_id=1, exercise_name="Elbow Flexion")
    session_id = logger.start_session()
    assert session_id is not None

    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    assert row is not None


def test_log_frame_writes_frame_and_error_events(db_path):
    logger = SessionLogger(db_path, patient_id=1)
    logger.start_session()
    result = _result(errors=[DetectedError(ErrorType.FAST_MOVEMENT, "minor", {"velocity_deg_per_sec": 250})])
    logger.log_frame(result, {"left_elbow_angle": 50.0})

    conn = sqlite3.connect(db_path)
    frames = conn.execute("SELECT * FROM session_frames").fetchall()
    events = conn.execute("SELECT * FROM session_events").fetchall()
    assert len(frames) == 1
    assert len(events) == 1


def test_log_frame_with_rep_event_writes_rep_row(db_path):
    logger = SessionLogger(db_path, patient_id=1)
    logger.start_session()
    rep_event = RepEvent(rep_number=1, completed=True, peak_progress_fraction=1.0, duration_frames=25)
    result = _result(rep_event=rep_event)
    logger.log_frame(result, {"left_elbow_angle": 50.0})

    conn = sqlite3.connect(db_path)
    reps = conn.execute("SELECT * FROM session_reps").fetchall()
    assert len(reps) == 1


def test_end_session_updates_summary_stats(db_path):
    logger = SessionLogger(db_path, patient_id=1)
    session_id = logger.start_session()
    logger.log_frame(_result(), {"left_elbow_angle": 50.0})
    logger.log_frame(_result(movement_quality=0.5), {"left_elbow_angle": 60.0})
    logger.end_session()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    assert row["ended_at"] is not None
    assert row["mean_quality"] == pytest.approx(0.675, abs=1e-6)


def test_log_frame_before_start_raises(db_path):
    logger = SessionLogger(db_path)
    with pytest.raises(RuntimeError):
        logger.log_frame(_result(), {})
