"""
alert_system.py
==================
Generates clinician-facing alerts when a patient's session data crosses
concerning thresholds: recovery decreasing, compensation increasing,
exercise accuracy decreasing, ROM decreasing, skipped sessions, or
unusually short sessions.

Alerts are computed on demand (not stored) from
`dashboard.session_manager.SessionManager` +
`dashboard.analytics.RecoveryAnalyticsEngine` data, so they always
reflect the current state of the database.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

from dashboard.analytics import RecoveryAnalyticsEngine
from dashboard.session_manager import SessionManager
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Alert:
    alert_type: str    # "recovery_decrease" | "compensation_increase" | "accuracy_decrease" |
                        # "rom_decrease" | "skipped_sessions" | "short_session"
    severity: str        # "info" | "warning" | "critical"
    message: str


class AlertSystem:
    """Computes alert conditions for a patient from recent session trends."""

    def __init__(
        self,
        db_path: str,
        recovery_decrease_threshold: float = 10.0,
        compensation_increase_threshold: float = 3.0,
        accuracy_decrease_threshold: float = 15.0,
        rom_decrease_threshold: float = 15.0,
        expected_session_interval_days: int = 3,
        skipped_sessions_grace_multiplier: float = 2.0,
        short_session_seconds_threshold: float = 60.0,
    ):
        self.db_path = db_path
        self.session_manager = SessionManager(db_path)
        self.analytics_engine = RecoveryAnalyticsEngine(db_path)

        self.recovery_decrease_threshold = recovery_decrease_threshold
        self.compensation_increase_threshold = compensation_increase_threshold
        self.accuracy_decrease_threshold = accuracy_decrease_threshold
        self.rom_decrease_threshold = rom_decrease_threshold
        self.expected_session_interval_days = expected_session_interval_days
        self.skipped_sessions_grace_multiplier = skipped_sessions_grace_multiplier
        self.short_session_seconds_threshold = short_session_seconds_threshold

    def check_patient(self, patient_id: int) -> List[Alert]:
        alerts: List[Alert] = []
        sessions = self.session_manager.list_sessions(patient_id=patient_id, limit=10)

        alerts += self._check_skipped_sessions(sessions)
        alerts += self._check_short_sessions(sessions)
        alerts += self._check_session_over_session_trends(patient_id, sessions)

        return alerts

    def _check_skipped_sessions(self, sessions) -> List[Alert]:
        if not sessions:
            return []

        most_recent = sessions[0]
        try:
            last_date = self.session_manager._parse_datetime(most_recent.started_at)
        except Exception:  # noqa: BLE001
            return []

        days_since = (datetime.now() - last_date).days
        grace_period = self.expected_session_interval_days * self.skipped_sessions_grace_multiplier

        if days_since > grace_period:
            return [Alert(
                "skipped_sessions", "warning",
                f"No session recorded in {days_since} days (expected roughly every "
                f"{self.expected_session_interval_days} days) — patient may be missing scheduled sessions.",
            )]
        return []

    def _check_short_sessions(self, sessions) -> List[Alert]:
        alerts = []
        for session in sessions[:3]:  # check the most recent few
            if session.duration_seconds is not None and 0 < session.duration_seconds < self.short_session_seconds_threshold:
                alerts.append(Alert(
                    "short_session", "info",
                    f"Session on {session.started_at[:10]} lasted only {session.duration_seconds:.0f}s — "
                    "unusually short; confirm the patient completed the intended exercise set.",
                ))
        return alerts

    def _check_session_over_session_trends(self, patient_id: int, sessions) -> List[Alert]:
        alerts = []
        if len(sessions) < 2:
            return alerts

        current, previous = sessions[0], sessions[1]

        if current.mean_quality is not None and previous.mean_quality is not None and previous.mean_quality > 0:
            change_pct = (current.mean_quality - previous.mean_quality) / previous.mean_quality * 100.0
            if change_pct < -self.accuracy_decrease_threshold:
                alerts.append(Alert(
                    "accuracy_decrease", "warning",
                    f"Exercise accuracy dropped {abs(change_pct):.0f}% compared to the previous session.",
                ))

        current_detail = self.session_manager.get_session_detail(current.session_id)
        previous_detail = self.session_manager.get_session_detail(previous.session_id)

        current_compensation = self._compensation_rate(current_detail)
        previous_compensation = self._compensation_rate(previous_detail)
        if current_compensation - previous_compensation > self.compensation_increase_threshold:
            alerts.append(Alert(
                "compensation_increase", "warning",
                f"Compensation events increased from {previous_compensation:.1f} to "
                f"{current_compensation:.1f} per 100 frames compared to the previous session.",
            ))

        current_rom = self._session_rom(current_detail)
        previous_rom = self._session_rom(previous_detail)
        if previous_rom > 0 and (previous_rom - current_rom) / previous_rom * 100.0 > self.rom_decrease_threshold:
            alerts.append(Alert(
                "rom_decrease", "warning",
                f"Range of motion decreased from {previous_rom:.0f}\u00b0 to {current_rom:.0f}\u00b0 "
                "compared to the previous session.",
            ))

        recovery_current = self.analytics_engine._session_recovery_score(current_detail, self.analytics_engine.metrics_calculator)
        recovery_previous = self.analytics_engine._session_recovery_score(previous_detail, self.analytics_engine.metrics_calculator)
        if recovery_previous > 0 and (recovery_previous - recovery_current) / recovery_previous * 100.0 > self.recovery_decrease_threshold:
            alerts.append(Alert(
                "recovery_decrease", "critical",
                f"Recovery score dropped from {recovery_previous:.0f} to {recovery_current:.0f} "
                "compared to the previous session — consider clinical review.",
            ))

        return alerts

    @staticmethod
    def _compensation_rate(detail) -> float:
        compensation_types = {"trunk_compensation", "shoulder_hiking", "body_lean", "poor_alignment"}
        events = [e for e in detail["events"] if e.get("error_type") in compensation_types]
        num_frames = max(1, len(detail["frames"]))
        return (len(events) / num_frames) * 100.0

    @staticmethod
    def _session_rom(detail) -> float:
        values = [f["rom_deg"] for f in detail["frames"] if f.get("rom_deg") is not None]
        return (max(values) - min(values)) if len(values) > 1 else (values[0] if values else 0.0)
