"""
session_logger.py
====================
Streams live session data to the dashboard's SQLite database as the
real-time pipeline runs — timestamp, exercise, rep number, joint
angles, movement quality, error list, confidence, ROM, compensation
events, and session duration — so the Streamlit dashboard's Exercise
History / Recovery Analytics pages (dashboard/pages/) can reflect an
in-progress or just-finished session without any separate export step.

Writes are batched per frame but kept lightweight (a handful of small
INSERTs) so logging never becomes the real-time loop's bottleneck; see
`docs/inference_guide.md` for the measured overhead.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from typing import Dict, List, Optional

from dashboard.db import get_connection, init_db
from inference.error_detector import DetectedError
from inference.movement_analyzer import MovementAnalysisResult
from utils.logger import get_logger

logger = get_logger(__name__)


class SessionLogger:
    """Opens (or creates) a session record and streams frame/rep/error
    events into it as the real-time pipeline produces them."""

    def __init__(self, db_path: str, patient_id: Optional[int] = None, exercise_name: str = "unassigned"):
        self.db_path = db_path
        self.patient_id = patient_id
        self.exercise_name = exercise_name
        self.session_id: Optional[int] = None
        self._session_start_time: Optional[float] = None
        self._quality_values: List[float] = []
        self._confidence_values: List[float] = []
        self._total_reps = 0

    def start_session(self) -> int:
        init_db(self.db_path)
        self._session_start_time = time.time()

        with get_connection(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO sessions (patient_id, exercise_name, started_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
                (self.patient_id, self.exercise_name),
            )
            self.session_id = cursor.lastrowid

        logger.info("Session logging started: session_id=%d, patient_id=%s", self.session_id, self.patient_id)
        return self.session_id

    def log_frame(self, result: MovementAnalysisResult, angles: Dict[str, float]) -> None:
        if self.session_id is None:
            raise RuntimeError("SessionLogger.log_frame() called before start_session().")

        self._quality_values.append(result.movement_quality)
        self._confidence_values.append(result.overall_confidence)

        rom_deg = None
        if result.expected_angle_deg is not None and result.actual_angle_deg is not None:
            rom_deg = abs(result.actual_angle_deg)

        with get_connection(self.db_path) as conn:
            conn.execute(
                """INSERT INTO session_frames
                   (session_id, frame_timestamp, exercise_name, phase, predicted_class,
                    confidence, movement_quality, rom_deg, joint_angles_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    self.session_id, time.time(), result.exercise_display_name,
                    result.phase.value if result.phase else None, None,
                    result.overall_confidence, result.movement_quality, rom_deg,
                    json.dumps(angles),
                ),
            )

            for error in result.errors:
                self._insert_event(conn, "error", error)

        if result.rep_event is not None:
            self._log_rep(result)

    def _log_rep(self, result: MovementAnalysisResult) -> None:
        rep_event = result.rep_event
        if rep_event.completed:
            self._total_reps += 1

        with get_connection(self.db_path) as conn:
            conn.execute(
                """INSERT INTO session_reps
                   (session_id, rep_number, exercise_name, completed, peak_progress_fraction, duration_frames, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    self.session_id, rep_event.rep_number, result.exercise_display_name,
                    int(rep_event.completed), rep_event.peak_progress_fraction,
                    rep_event.duration_frames, time.time(),
                ),
            )

    def _insert_event(self, conn, event_type: str, error: DetectedError) -> None:
        conn.execute(
            """INSERT INTO session_events (session_id, event_type, error_type, severity, detail_json, timestamp)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (self.session_id, event_type, error.error_type.value, error.severity, json.dumps(error.detail), time.time()),
        )

    def log_calibration(self, calibration_summary: Dict) -> None:
        if self.session_id is None:
            raise RuntimeError("SessionLogger.log_calibration() called before start_session().")
        with get_connection(self.db_path) as conn:
            conn.execute(
                """INSERT INTO session_events (session_id, event_type, error_type, severity, detail_json, timestamp)
                   VALUES (?, 'calibration', NULL, 'info', ?, ?)""",
                (self.session_id, json.dumps(calibration_summary), time.time()),
            )

    def end_session(self) -> None:
        if self.session_id is None:
            return

        duration = time.time() - self._session_start_time if self._session_start_time else 0.0
        mean_quality = sum(self._quality_values) / len(self._quality_values) if self._quality_values else None
        mean_confidence = sum(self._confidence_values) / len(self._confidence_values) if self._confidence_values else None

        with get_connection(self.db_path) as conn:
            conn.execute(
                """UPDATE sessions SET ended_at = CURRENT_TIMESTAMP, duration_seconds = ?,
                   total_reps = ?, mean_quality = ?, mean_confidence = ? WHERE session_id = ?""",
                (duration, self._total_reps, mean_quality, mean_confidence, self.session_id),
            )

        logger.info(
            "Session %d ended: duration=%.1fs, total_reps=%d, mean_quality=%s",
            self.session_id, duration, self._total_reps, mean_quality,
        )
