"""
analytics.py
==============
Recovery analytics engine: computes the full per-patient aggregate
metric suite across sessions —

    Average Exercise Accuracy      Compensation Frequency
    Average / Max / Min ROM         Exercise Consistency
    Average Joint Angle Error        Average Session Duration
    Exercise Completion Rate          Fatigue Indicator
    Movement Smoothness                Recovery Score
                                          Improvement Percentage
                                          Trend Analysis

Built on top of `dashboard.session_manager.SessionManager` and
`dashboard.clinical_metrics.ClinicalMetricsCalculator` — this module
is the aggregation layer that turns per-session data into the
patient-level recovery picture the dashboard displays.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from dashboard.clinical_metrics import ClinicalMetricsCalculator
from dashboard.db import get_connection
from dashboard.session_manager import SessionManager
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RecoveryAnalytics:
    patient_id: int
    num_sessions: int
    average_exercise_accuracy: float
    average_rom_deg: float
    max_rom_deg: float
    min_rom_deg: float
    average_joint_angle_error_deg: float
    exercise_completion_rate: float
    movement_smoothness: float
    compensation_frequency: float
    exercise_consistency: float
    average_session_duration_seconds: float
    fatigue_indicator: float
    recovery_score: float
    improvement_percentage: float
    trend: str  # "improving" | "declining" | "stable" | "insufficient_data"

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}


class RecoveryAnalyticsEngine:
    """Computes patient-level recovery analytics from stored session data."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.session_manager = SessionManager(db_path)
        self.metrics_calculator = ClinicalMetricsCalculator()

    def compute_patient_analytics(self, patient_id: int, session_limit: int = 200) -> RecoveryAnalytics:
        sessions = self.session_manager.list_sessions(patient_id=patient_id, limit=session_limit)
        # Chronological order for trend analysis (list_sessions returns newest-first).
        sessions = list(reversed(sessions))

        if not sessions:
            return self._empty_analytics(patient_id)

        details = [self.session_manager.get_session_detail(s.session_id) for s in sessions]

        accuracies = [self._session_accuracy(d) for d in details]
        rom_values = self._collect_all_rom(details)
        angle_errors = self._collect_all_angle_errors(details)
        completion_rates = [self._session_completion_rate(d) for d in details]
        smoothness_scores = [self.metrics_calculator._movement_smoothness(d["frames"]) for d in details]
        compensation_rates = [self._session_compensation_rate(d) for d in details]
        durations = [s.duration_seconds for s in sessions if s.duration_seconds is not None]
        quality_series = [self._session_mean_quality(d) for d in details]

        recovery_scores = [self._session_recovery_score(d, self.metrics_calculator) for d in details]

        improvement_pct, trend = self._trend_and_improvement(recovery_scores)
        consistency = self._consistency_score(quality_series)
        fatigue = self._fatigue_indicator(details)

        return RecoveryAnalytics(
            patient_id=patient_id,
            num_sessions=len(sessions),
            average_exercise_accuracy=self._safe_mean(accuracies),
            average_rom_deg=self._safe_mean(rom_values),
            max_rom_deg=float(np.max(rom_values)) if rom_values else 0.0,
            min_rom_deg=float(np.min(rom_values)) if rom_values else 0.0,
            average_joint_angle_error_deg=self._safe_mean(angle_errors),
            exercise_completion_rate=self._safe_mean(completion_rates),
            movement_smoothness=self._safe_mean(smoothness_scores),
            compensation_frequency=self._safe_mean(compensation_rates),
            exercise_consistency=consistency,
            average_session_duration_seconds=self._safe_mean(durations),
            fatigue_indicator=fatigue,
            recovery_score=self._safe_mean(recovery_scores),
            improvement_percentage=improvement_pct,
            trend=trend,
        )

    # ------------------------------------------------------------------
    # Per-session helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _session_accuracy(detail: Dict) -> float:
        """Exercise accuracy = fraction of reps completed correctly, 0-100."""
        reps = detail["reps"]
        if not reps:
            return 100.0
        return 100.0 * sum(1 for r in reps if r.get("completed")) / len(reps)

    @staticmethod
    def _session_completion_rate(detail: Dict) -> float:
        return RecoveryAnalyticsEngine._session_accuracy(detail)  # same definition, kept as a distinct named metric per spec

    @staticmethod
    def _session_mean_quality(detail: Dict) -> float:
        qualities = [f.get("movement_quality") for f in detail["frames"] if f.get("movement_quality") is not None]
        return float(np.mean(qualities)) if qualities else 0.0

    @staticmethod
    def _collect_all_rom(details: List[Dict]) -> List[float]:
        """Per-session ROM (max-min of the primary tracked angle), one value per session."""
        rom_values = []
        for detail in details:
            values = [f["rom_deg"] for f in detail["frames"] if f.get("rom_deg") is not None]
            if values:
                rom_values.append(max(values) - min(values) if len(values) > 1 else values[0])
        return rom_values

    def _collect_all_angle_errors(self, details: List[Dict]) -> List[float]:
        """Per-session mean joint-angle error, pulled from the
        `joint_metrics` table (populated by the analytics/reporting
        pipeline from calibration-baseline comparisons — see
        dashboard/analytics.py:save_joint_metrics)."""
        session_ids = [d["session"]["session_id"] for d in details]
        if not session_ids:
            return []

        placeholders = ",".join("?" * len(session_ids))
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                f"SELECT session_id, AVG(angle_error_deg) as avg_error FROM joint_metrics "
                f"WHERE session_id IN ({placeholders}) AND angle_error_deg IS NOT NULL GROUP BY session_id",
                session_ids,
            ).fetchall()
        return [row["avg_error"] for row in rows if row["avg_error"] is not None]

    @staticmethod
    def _session_compensation_rate(detail: Dict) -> float:
        """Compensation events per 100 frames for this session."""
        compensation_types = {"trunk_compensation", "shoulder_hiking", "body_lean", "poor_alignment"}
        events = [e for e in detail["events"] if e.get("error_type") in compensation_types]
        num_frames = max(1, len(detail["frames"]))
        return (len(events) / num_frames) * 100.0

    @staticmethod
    def _session_recovery_score(detail: Dict, calculator: ClinicalMetricsCalculator) -> float:
        scores = calculator.compute_for_session(detail["frames"], detail["events"], detail["reps"])
        return scores.recovery_index

    # ------------------------------------------------------------------
    # Aggregate-level metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _consistency_score(quality_series: List[float]) -> float:
        """Exercise consistency: 100 - normalized variance across
        sessions' mean quality. High consistency = performance doesn't
        swing wildly session to session (distinct from within-session
        `movement_stability`, which is frame-to-frame)."""
        if len(quality_series) < 2:
            return 100.0
        std = float(np.std(quality_series))
        return float(np.clip(100.0 * (1.0 - min(std * 2.0, 1.0)), 0.0, 100.0))

    @staticmethod
    def _fatigue_indicator(details: List[Dict]) -> float:
        """0-100, where higher = more evidence of fatigue: computed from
        the average *within-session* decline in movement quality from
        the first third to the last third of each session, averaged
        across sessions. A patient who reliably tires late in a session
        will show a consistent late-session quality drop."""
        declines = []
        for detail in details:
            qualities = [f.get("movement_quality") for f in detail["frames"] if f.get("movement_quality") is not None]
            if len(qualities) < 6:
                continue
            third = len(qualities) // 3
            first_third_mean = float(np.mean(qualities[:third]))
            last_third_mean = float(np.mean(qualities[-third:]))
            decline = first_third_mean - last_third_mean
            declines.append(decline)

        if not declines:
            return 0.0
        avg_decline = float(np.mean(declines))
        return float(np.clip(avg_decline * 200.0, 0.0, 100.0))  # scale a 0.5-quality-point drop -> 100

    @staticmethod
    def _trend_and_improvement(recovery_scores: List[float]) -> "tuple[float, str]":
        """Improvement percentage: relative change from the first-quarter
        average to the last-quarter average of the recovery-score series.
        Trend: a simple linear-regression-slope-sign classification."""
        if len(recovery_scores) < 2:
            return 0.0, "insufficient_data"

        quarter = max(1, len(recovery_scores) // 4)
        first_quarter_mean = float(np.mean(recovery_scores[:quarter]))
        last_quarter_mean = float(np.mean(recovery_scores[-quarter:]))

        if first_quarter_mean < 1e-6:
            improvement_pct = 0.0 if last_quarter_mean < 1e-6 else 100.0
        else:
            improvement_pct = ((last_quarter_mean - first_quarter_mean) / first_quarter_mean) * 100.0

        x = np.arange(len(recovery_scores))
        slope = float(np.polyfit(x, recovery_scores, 1)[0])
        if slope > 0.5:
            trend = "improving"
        elif slope < -0.5:
            trend = "declining"
        else:
            trend = "stable"

        return improvement_pct, trend

    @staticmethod
    def _empty_analytics(patient_id: int) -> RecoveryAnalytics:
        return RecoveryAnalytics(
            patient_id=patient_id, num_sessions=0, average_exercise_accuracy=0.0,
            average_rom_deg=0.0, max_rom_deg=0.0, min_rom_deg=0.0, average_joint_angle_error_deg=0.0,
            exercise_completion_rate=0.0, movement_smoothness=0.0, compensation_frequency=0.0,
            exercise_consistency=0.0, average_session_duration_seconds=0.0, fatigue_indicator=0.0,
            recovery_score=0.0, improvement_percentage=0.0, trend="insufficient_data",
        )

    def save_joint_metrics(self, session_id: int, target_angles: Optional[Dict[str, float]] = None) -> List[Dict]:
        """Compute and persist per-joint ROM + angle-error metrics for a
        session into the `joint_metrics` table. `target_angles` (e.g. the
        exercise's target_deg per angle name) enables angle-error
        computation; without it, only ROM is stored.
        """
        detail = self.session_manager.get_session_detail(session_id)
        angle_names = set()
        for frame in detail["frames"]:
            angle_names.update((frame.get("joint_angles") or {}).keys())

        saved_rows = []
        with get_connection(self.db_path) as conn:
            for angle_name in sorted(angle_names):
                values = [
                    frame["joint_angles"][angle_name]
                    for frame in detail["frames"]
                    if angle_name in (frame.get("joint_angles") or {})
                    and frame["joint_angles"][angle_name] is not None
                    and not (isinstance(frame["joint_angles"][angle_name], float) and np.isnan(frame["joint_angles"][angle_name]))
                ]
                if not values:
                    continue

                avg_angle = float(np.mean(values))
                min_angle = float(np.min(values))
                max_angle = float(np.max(values))
                rom = max_angle - min_angle

                angle_error = None
                if target_angles and angle_name in target_angles:
                    angle_error = abs(max_angle - target_angles[angle_name]) if max_angle >= min_angle else None

                conn.execute(
                    """INSERT INTO joint_metrics
                       (session_id, joint_name, avg_angle_deg, min_angle_deg, max_angle_deg, rom_deg, angle_error_deg)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (session_id, angle_name, avg_angle, min_angle, max_angle, rom, angle_error),
                )
                saved_rows.append({
                    "joint_name": angle_name, "avg_angle_deg": avg_angle, "min_angle_deg": min_angle,
                    "max_angle_deg": max_angle, "rom_deg": rom, "angle_error_deg": angle_error,
                })

        logger.info("Saved %d joint metric rows for session_id=%d.", len(saved_rows), session_id)
        return saved_rows

    @staticmethod
    def _safe_mean(values: List[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    # ------------------------------------------------------------------
    # Persist computed recovery scores back to the database
    # ------------------------------------------------------------------

    def save_session_scores(self, session_id: int, patient_id: int) -> Dict:
        """Compute and persist a session's clinical scores into
        `recovery_scores` — called after a session ends (see
        dashboard/pages usage) so historical trend queries don't need to
        recompute from raw frames every time."""
        detail = self.session_manager.get_session_detail(session_id)
        scores = self.metrics_calculator.compute_for_session(detail["frames"], detail["events"], detail["reps"])

        with get_connection(self.db_path) as conn:
            conn.execute(
                """INSERT INTO recovery_scores
                   (session_id, patient_id, shoulder_mobility_score, elbow_mobility_score,
                    movement_stability, movement_smoothness, symmetry_score, compensation_score,
                    exercise_quality_score, recovery_index, overall_rehabilitation_score)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id, patient_id, scores.shoulder_mobility_score, scores.elbow_mobility_score,
                    scores.movement_stability, scores.movement_smoothness, scores.symmetry_score,
                    scores.compensation_score, scores.exercise_quality_score, scores.recovery_index,
                    scores.overall_rehabilitation_score,
                ),
            )
        logger.info("Saved recovery scores for session_id=%d, patient_id=%d.", session_id, patient_id)
        return scores.to_dict()
