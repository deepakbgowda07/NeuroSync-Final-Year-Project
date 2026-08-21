"""
personalization_engine.py
============================
The Personalized Rehabilitation Recommendation Engine: generates
patient-specific guidance from calibration measurements, historical
sessions, recovery trends, current movement quality, ROM improvement,
and compensation history — recommending which exercises to prioritize
or reduce, suggested reps/sets, recommended movement speed, and
recovery focus areas.

Distinct from `dashboard/recommendation_engine.py` (session-level,
single-exercise recommendations from one session's recovery metrics)
— this module operates across a patient's *entire exercise history*
and rehabilitation plan to produce a longer-horizon personalized plan
adjustment, the kind a physiotherapist would review weekly rather than
after every session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from dashboard.analytics import RecoveryAnalyticsEngine
from dashboard.patient_manager import PatientManager
from dashboard.session_manager import SessionManager
from inference.exercise_library import SUPPORTED_EXERCISES
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PersonalizedPlan:
    patient_id: int
    exercises_to_prioritize: List[str] = field(default_factory=list)
    exercises_to_reduce: List[str] = field(default_factory=list)
    suggested_repetitions: int = 8
    suggested_sets: int = 3
    recommended_movement_speed: str = "controlled"  # "slower" | "controlled" | "current pace"
    recovery_focus_areas: List[str] = field(default_factory=list)
    rationale: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "patient_id": self.patient_id,
            "exercises_to_prioritize": self.exercises_to_prioritize,
            "exercises_to_reduce": self.exercises_to_reduce,
            "suggested_repetitions": self.suggested_repetitions,
            "suggested_sets": self.suggested_sets,
            "recommended_movement_speed": self.recommended_movement_speed,
            "recovery_focus_areas": self.recovery_focus_areas,
            "rationale": self.rationale,
        }


class PersonalizationEngine:
    """Builds a PersonalizedPlan from a patient's calibration data,
    session history, and computed recovery analytics.

    This module estimates progress and suggests *exercise-practice*
    adjustments only — it never makes a medical diagnosis or a
    treatment decision; those remain the physiotherapist's judgment
    (see the module-level note in dashboard/recovery_tracking.py for
    the same principle applied to trend reporting).
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.patient_manager = PatientManager(db_path)
        self.session_manager = SessionManager(db_path)
        self.analytics_engine = RecoveryAnalyticsEngine(db_path)

    def build_plan(self, patient_id: int) -> PersonalizedPlan:
        analytics = self.analytics_engine.compute_patient_analytics(patient_id)
        sessions = self.session_manager.list_sessions(patient_id=patient_id, limit=200)
        active_plan = self.patient_manager.get_active_plan(patient_id)

        plan = PersonalizedPlan(patient_id=patient_id)

        if analytics.num_sessions == 0:
            plan.rationale.append("No session history yet — starting with a conservative introductory plan.")
            plan.exercises_to_prioritize = (active_plan["exercises"] if active_plan else SUPPORTED_EXERCISES[:3])
            plan.suggested_repetitions = 6
            plan.suggested_sets = 2
            return plan

        per_exercise_stats = self._per_exercise_stats(sessions)

        plan.exercises_to_prioritize, plan.exercises_to_reduce = self._exercise_selection(per_exercise_stats)
        plan.suggested_repetitions, plan.suggested_sets = self._suggested_volume(analytics)
        plan.recommended_movement_speed = self._recommended_speed(analytics)
        plan.recovery_focus_areas = self._recovery_focus_areas(analytics, per_exercise_stats)
        plan.rationale = self._rationale(analytics, per_exercise_stats)

        return plan

    # ------------------------------------------------------------------
    # Per-exercise history stats
    # ------------------------------------------------------------------

    def _per_exercise_stats(self, sessions) -> Dict[str, Dict]:
        stats: Dict[str, Dict] = {}
        for session in sessions:
            key = session.exercise_name
            entry = stats.setdefault(key, {"sessions": 0, "qualities": [], "rom_values": []})
            entry["sessions"] += 1

            detail = self.session_manager.get_session_detail(session.session_id)
            qualities = [f.get("movement_quality") for f in detail["frames"] if f.get("movement_quality") is not None]
            if qualities:
                entry["qualities"].append(float(np.mean(qualities)))

            rom_values = [f["rom_deg"] for f in detail["frames"] if f.get("rom_deg") is not None]
            if len(rom_values) > 1:
                entry["rom_values"].append(max(rom_values) - min(rom_values))

        return stats

    # ------------------------------------------------------------------
    # Plan components
    # ------------------------------------------------------------------

    @staticmethod
    def _exercise_selection(per_exercise_stats: Dict[str, Dict]) -> "tuple[List[str], List[str]]":
        """Prioritize exercises with low average quality or a declining
        recent trend (need more practice); reduce exercises already
        performing consistently well (avoid over-practicing a mastered
        movement at the expense of others)."""
        prioritize, reduce = [], []

        for exercise_name, entry in per_exercise_stats.items():
            qualities = entry["qualities"]
            if not qualities:
                continue
            mean_quality = float(np.mean(qualities))

            if mean_quality < 0.6:
                prioritize.append(exercise_name)
            elif mean_quality > 0.88 and len(qualities) >= 3:
                reduce.append(exercise_name)

        return prioritize, reduce

    @staticmethod
    def _suggested_volume(analytics) -> "tuple[int, int]":
        """Higher completion rate / lower fatigue -> can handle more
        volume; struggling patients get a lighter, more achievable load."""
        base_reps, base_sets = 8, 3

        if analytics.exercise_completion_rate > 85 and analytics.fatigue_indicator < 20:
            return base_reps + 2, base_sets
        if analytics.exercise_completion_rate < 60 or analytics.fatigue_indicator > 50:
            return max(4, base_reps - 3), max(1, base_sets - 1)
        return base_reps, base_sets

    @staticmethod
    def _recommended_speed(analytics) -> str:
        if analytics.movement_smoothness < 55:
            return "slower"
        if analytics.movement_smoothness > 85:
            return "current pace"
        return "controlled"

    @staticmethod
    def _recovery_focus_areas(analytics, per_exercise_stats: Dict[str, Dict]) -> List[str]:
        focus_areas = []

        if analytics.compensation_frequency > 4.0:
            focus_areas.append("Reducing compensatory movement (trunk lean, shoulder hiking)")
        if analytics.average_rom_deg > 0 and analytics.max_rom_deg > 0 and (analytics.average_rom_deg / analytics.max_rom_deg) < 0.7:
            focus_areas.append("Range-of-motion consistency across sessions")
        if analytics.exercise_consistency < 65:
            focus_areas.append("Session-to-session consistency (regular practice schedule)")
        if analytics.movement_smoothness < 60:
            focus_areas.append("Movement smoothness and motor control")

        low_quality_exercises = [
            name for name, entry in per_exercise_stats.items()
            if entry["qualities"] and float(np.mean(entry["qualities"])) < 0.55
        ]
        if low_quality_exercises:
            focus_areas.append(f"Foundational form on: {', '.join(low_quality_exercises)}")

        if not focus_areas:
            focus_areas.append("Maintaining current gains — continue the established plan")

        return focus_areas

    @staticmethod
    def _rationale(analytics, per_exercise_stats: Dict[str, Dict]) -> List[str]:
        rationale = [
            f"Based on {analytics.num_sessions} recorded session(s), recovery trend is '{analytics.trend}' "
            f"with a {analytics.improvement_percentage:+.1f}% change in recovery score.",
        ]
        if analytics.compensation_frequency > 4.0:
            rationale.append(f"Compensation frequency ({analytics.compensation_frequency:.1f} events/100 frames) is above the comfortable range.")
        if analytics.fatigue_indicator > 40:
            rationale.append("Quality consistently declines later in sessions, suggesting reduced volume may help.")
        return rationale
