"""
explainable_ai.py
====================
Wraps every real-time prediction in a structured explanation, matching
the required format:

    Exercise: Shoulder Flexion
    Confidence: 97%
    Movement Quality: 84%
    Reason: Shoulder angle remained below target by approximately 15
            degrees during the lifting phase. Trunk lean increased
            during the final third of the movement.
    Recommendation: Lift the arm higher while keeping the torso upright.

Built by combining `inference/movement_analyzer.py`'s per-frame result,
`inference/clinical_assessment.py`'s per-repetition scores, and
`inference/clinical_reasoning.py`'s rule firings — this module's job is
purely to render those into one coherent, non-generic explanation.
Never emits bare "Wrong" / "Correct" (see `_reason_sentences`, which
always cites a specific measured quantity).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

from inference.clinical_assessment import RepetitionAssessment
from inference.clinical_reasoning import ClinicalReasoningEngine, ClinicalReasoningResult
from inference.error_detector import DetectedError
from inference.exercise_library import ExerciseDefinition
from utils.logger import get_logger

if TYPE_CHECKING:
    from inference.movement_analyzer import MovementAnalysisResult

logger = get_logger(__name__)


@dataclass
class Explanation:
    exercise: str
    confidence_percent: float
    movement_quality_percent: float
    reasons: List[str] = field(default_factory=list)
    recommendation: str = ""
    reasoning_trace: List[ClinicalReasoningResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "exercise": self.exercise,
            "confidence": f"{self.confidence_percent:.0f}%",
            "movement_quality": f"{self.movement_quality_percent:.0f}%",
            "reason": " ".join(self.reasons) if self.reasons else "Movement matched expected form for this repetition.",
            "recommendation": self.recommendation or "Continue with the current technique.",
        }

    def to_display_string(self) -> str:
        """Renders the exact spec-format block for logging/CLI/demo display."""
        lines = [
            f"Exercise: {self.exercise}",
            f"Confidence: {self.confidence_percent:.0f}%",
            f"Movement Quality: {self.movement_quality_percent:.0f}%",
            f"Reason: {' '.join(self.reasons) if self.reasons else 'Movement matched expected form for this repetition.'}",
            f"Recommendation: {self.recommendation or 'Continue with the current technique.'}",
        ]
        return "\n".join(lines)


class ExplainableAIEngine:
    """Produces a structured Explanation for a movement-analysis result."""

    def __init__(self, reasoning_engine: Optional[ClinicalReasoningEngine] = None):
        self.reasoning_engine = reasoning_engine or ClinicalReasoningEngine()

    def explain(
        self,
        result: MovementAnalysisResult,
        definition: Optional[ExerciseDefinition],
        assessment: Optional[RepetitionAssessment] = None,
        peak_velocity_deg_per_sec: Optional[float] = None,
    ) -> Explanation:
        if result.exercise_key is None or definition is None:
            return Explanation(
                exercise="Unknown", confidence_percent=0.0, movement_quality_percent=0.0,
                reasons=["No exercise has been recognized yet."],
                recommendation="Begin the exercise whenever you're ready.",
            )

        reasoning_trace: List[ClinicalReasoningResult] = []
        if assessment is not None:
            reasoning_trace = self.reasoning_engine.evaluate(
                definition, assessment, result.errors,
                achieved_angle_deg=result.actual_angle_deg,
                peak_velocity_deg_per_sec=peak_velocity_deg_per_sec,
            )

        reasons = self._reason_sentences(result, definition, reasoning_trace, assessment)
        recommendation = self._recommendation(result, reasoning_trace, assessment)

        return Explanation(
            exercise=result.exercise_display_name or definition.display_name,
            confidence_percent=result.exercise_recognition_confidence * 100.0,
            movement_quality_percent=result.movement_quality * 100.0,
            reasons=reasons,
            recommendation=recommendation,
            reasoning_trace=reasoning_trace,
        )

    def _reason_sentences(
        self, result: MovementAnalysisResult, definition: ExerciseDefinition,
        reasoning_trace: List[ClinicalReasoningResult], assessment: Optional[RepetitionAssessment],
    ) -> List[str]:
        """Builds specific, quantified reason sentences — never a bare
        'wrong'/'correct' judgement. Prefers reasoning-engine conditions
        (already quantified). If a repetition just completed cleanly
        (high assessment score, no rules fired), states that positively
        rather than falling back to a raw current-frame angle deficit —
        which, right after a rep completes, reflects the patient having
        returned to neutral rather than any actual problem.
        """
        if reasoning_trace:
            return [r.condition for r in reasoning_trace]

        if assessment is not None:
            if assessment.overall_rehabilitation_score >= 80.0:
                return [
                    f"{definition.display_name} repetition completed with "
                    f"{assessment.range_of_motion_score:.0f}% of expected range of motion and smooth, controlled movement."
                ]
            return [
                f"{definition.display_name} repetition scored {assessment.overall_rehabilitation_score:.0f}/100 overall, "
                f"with range-of-motion at {assessment.range_of_motion_score:.0f}% of the expected span."
            ]

        if result.phase is not None and result.phase.value != "neutral" and result.actual_angle_deg is not None and result.expected_angle_deg is not None:
            deficit = abs(result.expected_angle_deg - result.actual_angle_deg)
            if deficit > 5.0:
                direction = "below" if (
                    (definition.rep_direction == "increasing" and result.actual_angle_deg < result.expected_angle_deg)
                    or (definition.rep_direction == "decreasing" and result.actual_angle_deg > result.expected_angle_deg)
                ) else "beyond"
                return [
                    f"{definition.display_name} angle remained {direction} target by approximately "
                    f"{deficit:.0f} degrees during the {result.phase.value.replace('_', ' ')} phase."
                ]

        return []

    def _recommendation(
        self, result: MovementAnalysisResult, reasoning_trace: List[ClinicalReasoningResult],
        assessment: Optional[RepetitionAssessment] = None,
    ) -> str:
        if reasoning_trace:
            # Lead with the highest-priority (first-evaluated) rule's recommendation;
            # ClinicalReasoningEngine.evaluate() orders rules by clinical significance.
            return reasoning_trace[0].recommendation

        if assessment is not None and assessment.overall_rehabilitation_score >= 80.0:
            return "Excellent repetition — continue at this pace and range."

        if result.movement_quality > 0.9 and not result.errors:
            return "Excellent form — continue at this pace and range."

        return "Continue the movement, keeping your torso upright and motion controlled."
