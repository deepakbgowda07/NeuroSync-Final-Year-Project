"""
session_manager.py
=====================
Session-level query operations: history listing, single-session
detail retrieval (for "replay"), and comparison between a current
session and a previous session / previous week / previous month.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from dashboard.db import get_connection
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SessionSummary:
    session_id: int
    patient_id: Optional[int]
    exercise_name: str
    started_at: str
    ended_at: Optional[str]
    duration_seconds: Optional[float]
    total_reps: Optional[int]
    mean_quality: Optional[float]
    mean_confidence: Optional[float]
    original_video_path: Optional[str] = None
    processed_video_path: Optional[str] = None

    @classmethod
    def from_row(cls, row) -> "SessionSummary":
        row_dict = dict(row)
        return cls(
            session_id=row["session_id"], patient_id=row["patient_id"], exercise_name=row["exercise_name"],
            started_at=row["started_at"], ended_at=row["ended_at"], duration_seconds=row["duration_seconds"],
            total_reps=row["total_reps"], mean_quality=row["mean_quality"], mean_confidence=row["mean_confidence"],
            original_video_path=row_dict.get("original_video_path"),
            processed_video_path=row_dict.get("processed_video_path"),
        )


class SessionManager:
    """Query layer over sessions, session_reps, session_events, and session_frames."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def list_sessions(self, patient_id: Optional[int] = None, limit: int = 100) -> List[SessionSummary]:
        query = "SELECT * FROM sessions"
        params: list = []
        if patient_id is not None:
            query += " WHERE patient_id = ?"
            params.append(patient_id)
        query += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)

        with get_connection(self.db_path) as conn:
            rows = conn.execute(query, params).fetchall()
        return [SessionSummary.from_row(r) for r in rows]

    def get_session_detail(self, session_id: int) -> Dict:
        """Full replay-ready detail for one session: summary, all reps,
        all frames (for a joint-angle timeline), and all events."""
        with get_connection(self.db_path) as conn:
            session_row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
            if session_row is None:
                raise ValueError(f"No session found with id={session_id}")

            reps = conn.execute(
                "SELECT * FROM session_reps WHERE session_id = ? ORDER BY rep_number", (session_id,)
            ).fetchall()
            frames = conn.execute(
                "SELECT * FROM session_frames WHERE session_id = ? ORDER BY frame_id", (session_id,)
            ).fetchall()
            events = conn.execute(
                "SELECT * FROM session_events WHERE session_id = ? ORDER BY timestamp", (session_id,)
            ).fetchall()
            exercise_results = conn.execute(
                "SELECT * FROM exercise_results WHERE session_id = ?", (session_id,)
            ).fetchall()

        successful = sum(1 for r in reps if r["completed"])
        incorrect = sum(1 for r in reps if not r["completed"])

        return {
            "session": dict(session_row),
            "reps": [dict(r) for r in reps],
            "frames": [
                {
                    **dict(f),
                    "joint_angles": json.loads(f["joint_angles_json"]) if f["joint_angles_json"] else {},
                    "landmarks": json.loads(f["landmarks_json"]) if ("landmarks_json" in dict(f) and f["landmarks_json"]) else None
                }
                for f in frames
            ],
            "events": [dict(e) for e in events],
            "exercise_results": [dict(r) for r in exercise_results],
            "successful_repetitions": successful,
            "incorrect_repetitions": incorrect,
        }

    def get_recent_sessions_in_window(self, patient_id: int, days: int) -> List[SessionSummary]:
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM sessions WHERE patient_id = ? AND started_at >= ? ORDER BY started_at DESC",
                (patient_id, cutoff),
            ).fetchall()
        return [SessionSummary.from_row(r) for r in rows]

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare_sessions(self, current_session_id: int, comparison: str = "previous_session") -> Dict:
        """Compare `current_session_id` against another session selected
        by `comparison`: "previous_session" | "previous_week" | "previous_month".

        Returns a dict with both sessions' summary stats and a verdict
        per metric: "improvement" | "regression" | "stable".
        """
        with get_connection(self.db_path) as conn:
            current_row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (current_session_id,)).fetchone()
            if current_row is None:
                raise ValueError(f"No session found with id={current_session_id}")
            current = SessionSummary.from_row(current_row)

            baseline_row = self._find_baseline_session(conn, current, comparison)

        if baseline_row is None:
            return {
                "current": current.__dict__,
                "baseline": None,
                "comparison": comparison,
                "verdicts": {},
                "note": f"No baseline session found for comparison type '{comparison}'.",
            }

        baseline = SessionSummary.from_row(baseline_row)
        verdicts = {
            "quality": self._verdict(current.mean_quality, baseline.mean_quality),
            "reps": self._verdict(current.total_reps, baseline.total_reps),
            "confidence": self._verdict(current.mean_confidence, baseline.mean_confidence),
        }

        return {
            "current": current.__dict__,
            "baseline": baseline.__dict__,
            "comparison": comparison,
            "verdicts": verdicts,
        }

    def _find_baseline_session(self, conn, current: SessionSummary, comparison: str):
        if current.patient_id is None:
            return None

        if comparison == "previous_session":
            return conn.execute(
                "SELECT * FROM sessions WHERE patient_id = ? AND session_id != ? AND started_at < ? "
                "ORDER BY started_at DESC LIMIT 1",
                (current.patient_id, current.session_id, current.started_at),
            ).fetchone()

        if comparison in ("previous_week", "previous_month"):
            days = 7 if comparison == "previous_week" else 30
            current_dt = self._parse_datetime(current.started_at)
            window_start = (current_dt - timedelta(days=2 * days)).isoformat()
            window_end = (current_dt - timedelta(days=days)).isoformat()
            return conn.execute(
                "SELECT * FROM sessions WHERE patient_id = ? AND session_id != ? "
                "AND started_at BETWEEN ? AND ? ORDER BY started_at DESC LIMIT 1",
                (current.patient_id, current.session_id, window_start, window_end),
            ).fetchone()

        raise ValueError(f"Unknown comparison type: {comparison}")

    @staticmethod
    def _parse_datetime(value: str) -> datetime:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        return datetime.now()

    @staticmethod
    def _verdict(current_value: Optional[float], baseline_value: Optional[float], tolerance: float = 0.03) -> str:
        if current_value is None or baseline_value is None:
            return "stable"
        if baseline_value == 0:
            return "stable"
        relative_change = (current_value - baseline_value) / abs(baseline_value)
        if relative_change > tolerance:
            return "improvement"
        if relative_change < -tolerance:
            return "regression"
        return "stable"
