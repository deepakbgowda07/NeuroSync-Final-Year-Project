"""
recovery_tracking.py
=======================
Tracks rehabilitation recovery over time: daily, weekly, and monthly
improvement, a consistency score, a compliance score (against the
patient's assigned rehabilitation plan frequency), recovery velocity
(rate of recovery-score change per week), and an overall trend
classification.

This module estimates *progress* from measured session data only. It
does not make medical diagnoses or treatment decisions — those remain
the physiotherapist's responsibility. Every number here is a
descriptive statistic over logged sessions, not a clinical judgment.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np

from dashboard.analytics import RecoveryAnalyticsEngine
from dashboard.clinical_metrics import ClinicalMetricsCalculator
from dashboard.patient_manager import PatientManager
from dashboard.session_manager import SessionManager
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RecoveryTrackingReport:
    patient_id: int
    daily_improvement_percent: Optional[float]
    weekly_improvement_percent: Optional[float]
    monthly_improvement_percent: Optional[float]
    consistency_score: float
    compliance_score: float
    recovery_velocity_per_week: float
    recovery_trend: str  # "improving" | "declining" | "stable" | "insufficient_data"
    note: str = (
        "Progress estimate derived from movement-quality measurements only. "
        "This is not a medical diagnosis or a treatment recommendation."
    )

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}


class RecoveryTracker:
    """Computes the recovery-over-time reporting suite for one patient."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.patient_manager = PatientManager(db_path)
        self.session_manager = SessionManager(db_path)
        self.analytics_engine = RecoveryAnalyticsEngine(db_path)
        self.metrics_calculator = ClinicalMetricsCalculator()

    def track(self, patient_id: int) -> RecoveryTrackingReport:
        sessions = self.session_manager.list_sessions(patient_id=patient_id, limit=500)
        sessions = list(reversed(sessions))  # chronological

        if len(sessions) < 2:
            return RecoveryTrackingReport(
                patient_id=patient_id, daily_improvement_percent=None, weekly_improvement_percent=None,
                monthly_improvement_percent=None, consistency_score=0.0, compliance_score=0.0,
                recovery_velocity_per_week=0.0, recovery_trend="insufficient_data",
            )

        dated_scores = self._dated_recovery_scores(sessions)

        daily = self._improvement_over_window(dated_scores, days=1)
        weekly = self._improvement_over_window(dated_scores, days=7)
        monthly = self._improvement_over_window(dated_scores, days=30)

        consistency = self._consistency_score(dated_scores)
        compliance = self._compliance_score(patient_id, sessions)
        velocity = self._recovery_velocity(dated_scores)
        trend = self._trend(dated_scores)

        return RecoveryTrackingReport(
            patient_id=patient_id,
            daily_improvement_percent=daily,
            weekly_improvement_percent=weekly,
            monthly_improvement_percent=monthly,
            consistency_score=consistency,
            compliance_score=compliance,
            recovery_velocity_per_week=velocity,
            recovery_trend=trend,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _dated_recovery_scores(self, sessions) -> List["tuple[datetime, float]"]:
        results = []
        for session in sessions:
            detail = self.session_manager.get_session_detail(session.session_id)
            scores = self.metrics_calculator.compute_for_session(detail["frames"], detail["events"], detail["reps"])
            dt = self.session_manager._parse_datetime(session.started_at)
            results.append((dt, scores.recovery_index))
        return results

    @staticmethod
    def _improvement_over_window(dated_scores: List["tuple[datetime, float]"], days: int) -> Optional[float]:
        """Percentage change between the earliest and latest score within
        the most recent `days`-day window, or None if fewer than two
        sessions fall in that window."""
        if not dated_scores:
            return None
        latest_dt = dated_scores[-1][0]
        window_start = latest_dt - timedelta(days=days)
        in_window = [(dt, score) for dt, score in dated_scores if dt >= window_start]

        if len(in_window) < 2:
            return None

        first_score = in_window[0][1]
        last_score = in_window[-1][1]
        if first_score < 1e-6:
            return 0.0 if last_score < 1e-6 else 100.0
        return ((last_score - first_score) / first_score) * 100.0

    @staticmethod
    def _consistency_score(dated_scores: List["tuple[datetime, float]"]) -> float:
        """100 - normalized variance of recovery scores across all
        sessions — a patient whose scores don't swing wildly session to
        session is judged more consistent."""
        if len(dated_scores) < 2:
            return 100.0
        scores = [s for _, s in dated_scores]
        std = float(np.std(scores))
        return float(np.clip(100.0 - std, 0.0, 100.0))

    def _compliance_score(self, patient_id: int, sessions) -> float:
        """Ratio of actual sessions logged to expected sessions, per the
        patient's active rehabilitation plan's sessions_per_week (falls
        back to a default of 3/week if no plan is assigned)."""
        plan = self.patient_manager.get_active_plan(patient_id)
        expected_per_week = plan["sessions_per_week"] if plan else 3

        if len(sessions) < 2:
            return 100.0 if sessions else 0.0

        first_dt = self.session_manager._parse_datetime(sessions[0].started_at)
        last_dt = self.session_manager._parse_datetime(sessions[-1].started_at)
        elapsed_weeks = max((last_dt - first_dt).days / 7.0, 1.0 / 7.0)

        expected_sessions = expected_per_week * elapsed_weeks
        actual_sessions = len(sessions)

        return float(np.clip((actual_sessions / max(expected_sessions, 1e-6)) * 100.0, 0.0, 100.0))

    @staticmethod
    def _recovery_velocity(dated_scores: List["tuple[datetime, float]"]) -> float:
        """Rate of recovery-score change per week, via a simple linear
        fit over session dates — positive means improving, negative
        means declining, magnitude indicates how fast."""
        if len(dated_scores) < 2:
            return 0.0

        t0 = dated_scores[0][0]
        x_days = np.array([(dt - t0).total_seconds() / 86400.0 for dt, _ in dated_scores])
        y_scores = np.array([score for _, score in dated_scores])

        if np.ptp(x_days) < 1e-6:
            return 0.0

        slope_per_day = float(np.polyfit(x_days, y_scores, 1)[0])
        return slope_per_day * 7.0

    @staticmethod
    def _trend(dated_scores: List["tuple[datetime, float]"]) -> str:
        if len(dated_scores) < 2:
            return "insufficient_data"
        velocity = RecoveryTracker._recovery_velocity(dated_scores)
        if velocity > 1.0:
            return "improving"
        if velocity < -1.0:
            return "declining"
        return "stable"
