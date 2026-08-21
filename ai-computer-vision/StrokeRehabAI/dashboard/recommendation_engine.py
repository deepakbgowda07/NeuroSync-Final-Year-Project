"""
recommendation_engine.py
===========================
Generates specific, actionable clinical recommendations from a
patient's computed analytics (dashboard/analytics.py) and clinical
scores (dashboard/clinical_metrics.py) — e.g. "Increase Shoulder
Flexion repetitions", "Reduce trunk compensation", "Slow movement
speed". Every recommendation is derived from a specific calculated
metric crossing a threshold; nothing is generic filler text.

TODO (next development phase): thresholds below are principled
defaults, not clinician-validated — same caveat as
inference/exercise_library.py and dashboard/clinical_metrics.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from dashboard.analytics import RecoveryAnalytics
from dashboard.clinical_metrics import ClinicalScores
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Recommendation:
    category: str      # "rom" | "compensation" | "speed" | "consistency" | "quality" | "fatigue"
    priority: str       # "low" | "medium" | "high"
    text: str


class RecommendationEngine:
    """Rule-based recommendation generator over analytics + clinical scores."""

    def __init__(
        self,
        low_rom_threshold: float = 60.0,
        high_compensation_threshold: float = 5.0,
        low_completion_rate_threshold: float = 70.0,
        low_consistency_threshold: float = 60.0,
        high_fatigue_threshold: float = 40.0,
        low_smoothness_threshold: float = 60.0,
    ):
        self.low_rom_threshold = low_rom_threshold
        self.high_compensation_threshold = high_compensation_threshold
        self.low_completion_rate_threshold = low_completion_rate_threshold
        self.low_consistency_threshold = low_consistency_threshold
        self.high_fatigue_threshold = high_fatigue_threshold
        self.low_smoothness_threshold = low_smoothness_threshold

    def generate(
        self,
        analytics: RecoveryAnalytics,
        clinical_scores: Optional[ClinicalScores] = None,
        exercise_display_name: str = "the assigned exercise",
    ) -> List[Recommendation]:
        recommendations: List[Recommendation] = []

        if analytics.num_sessions == 0:
            return [Recommendation("general", "low", "No session data yet — begin with a calibration session and a short introductory exercise set.")]

        recommendations += self._rom_recommendations(analytics, exercise_display_name)
        recommendations += self._compensation_recommendations(analytics, clinical_scores)
        recommendations += self._completion_recommendations(analytics, exercise_display_name)
        recommendations += self._consistency_recommendations(analytics)
        recommendations += self._fatigue_recommendations(analytics)
        recommendations += self._smoothness_recommendations(analytics)

        if not recommendations:
            recommendations.append(Recommendation("general", "low", "Performance is stable across recent sessions — continue the current plan."))

        return recommendations

    def _rom_recommendations(self, analytics: RecoveryAnalytics, exercise_name: str) -> List[Recommendation]:
        recs = []
        rom_ratio_of_max = (analytics.average_rom_deg / analytics.max_rom_deg * 100.0) if analytics.max_rom_deg else 100.0

        if analytics.average_rom_deg > 0 and rom_ratio_of_max < self.low_rom_threshold:
            recs.append(Recommendation(
                "rom", "high",
                f"Increase {exercise_name} range of motion gradually — recent sessions are averaging well "
                f"below the best range achieved ({analytics.average_rom_deg:.0f}\u00b0 vs. a peak of {analytics.max_rom_deg:.0f}\u00b0).",
            ))
        if analytics.trend == "declining":
            recs.append(Recommendation("rom", "medium", "Recovery trend has declined over recent sessions — consider reviewing exercise difficulty or checking for pain/fatigue."))
        return recs

    def _compensation_recommendations(self, analytics: RecoveryAnalytics, clinical_scores: Optional[ClinicalScores]) -> List[Recommendation]:
        recs = []
        if analytics.compensation_frequency > self.high_compensation_threshold:
            recs.append(Recommendation(
                "compensation", "high",
                f"Reduce trunk compensation — compensation events are occurring at "
                f"{analytics.compensation_frequency:.1f} per 100 frames, above the recommended threshold.",
            ))
        if clinical_scores is not None and clinical_scores.symmetry_score < 60.0:
            recs.append(Recommendation(
                "compensation", "medium",
                "Movement asymmetry detected between sides — focus on even, bilateral effort during exercises.",
            ))
        return recs

    def _completion_recommendations(self, analytics: RecoveryAnalytics, exercise_name: str) -> List[Recommendation]:
        recs = []
        if analytics.exercise_completion_rate < self.low_completion_rate_threshold:
            recs.append(Recommendation(
                "quality", "high",
                f"Improve {exercise_name} completion — only {analytics.exercise_completion_rate:.0f}% of "
                f"repetitions are reaching full range. Consider more supervised practice or reduced target reps per set.",
            ))
        return recs

    def _consistency_recommendations(self, analytics: RecoveryAnalytics) -> List[Recommendation]:
        recs = []
        if analytics.exercise_consistency < self.low_consistency_threshold:
            recs.append(Recommendation(
                "consistency", "medium",
                "Session-to-session performance is inconsistent — a more regular practice schedule may help stabilize progress.",
            ))
        return recs

    def _fatigue_recommendations(self, analytics: RecoveryAnalytics) -> List[Recommendation]:
        recs = []
        if analytics.fatigue_indicator > self.high_fatigue_threshold:
            recs.append(Recommendation(
                "fatigue", "medium",
                "Quality consistently drops later in sessions — consider shorter sessions with more frequent rest breaks.",
            ))
        return recs

    def _smoothness_recommendations(self, analytics: RecoveryAnalytics) -> List[Recommendation]:
        recs = []
        if analytics.movement_smoothness < self.low_smoothness_threshold:
            recs.append(Recommendation("speed", "medium", "Slow movement speed slightly — motion is registering as jerky rather than smooth and controlled."))
        return recs
