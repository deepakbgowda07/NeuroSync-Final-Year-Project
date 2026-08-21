"""
clinical_reasoning.py
========================
A rule-based clinical reasoning engine implementing established
physiotherapy principles as explicit IF-THEN rules — each rule
produces a recommendation *and* the reasoning behind it, so every
output is traceable to a specific measured condition rather than being
a black-box judgment.

Examples of the rules implemented:
    IF shoulder flexion ROM < expected
        THEN recommend increasing shoulder flexion mobility exercises.
    IF trunk lean exceeds threshold
        THEN flag trunk compensation, advise upright posture.
    IF elbow extension remains limited
        THEN suggest controlled elbow extension exercises.
    IF movement is too fast
        THEN recommend slower controlled repetitions.

This is the reasoning layer `inference/explainable_ai.py` draws on to
build patient-facing explanations, and is deliberately separate from
`inference/error_detector.py` (which *detects* conditions per frame) —
this module *interprets* a repetition's or session's condition set
into clinical guidance, citing which condition triggered it.

TODO (next development phase): rule thresholds are principled defaults
from general physiotherapy literature, not this project's own
clinician-validated data — same caveat as
inference/exercise_library.py and dashboard/clinical_metrics.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from inference.clinical_assessment import RepetitionAssessment
from inference.error_detector import DetectedError, ErrorType
from inference.exercise_library import ExerciseDefinition
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ClinicalReasoningResult:
    condition: str          # the specific measured condition that triggered this rule
    recommendation: str      # the actionable clinical guidance
    explanation: str          # why this recommendation follows from the condition
    rule_id: str


class ClinicalReasoningEngine:
    """Evaluates a fixed set of physiotherapy IF-THEN rules against the
    current repetition/session's measured data and returns every rule
    that fired, each with its own explanation."""

    def __init__(
        self,
        rom_deficit_ratio_threshold: float = 0.75,
        trunk_lean_threshold_deg: float = 15.0,
        elbow_extension_limited_deg: float = 150.0,
        fast_movement_deg_per_sec: float = 200.0,
        low_smoothness_threshold: float = 60.0,
        low_symmetry_threshold: float = 60.0,
    ):
        self.rom_deficit_ratio_threshold = rom_deficit_ratio_threshold
        self.trunk_lean_threshold_deg = trunk_lean_threshold_deg
        self.elbow_extension_limited_deg = elbow_extension_limited_deg
        self.fast_movement_deg_per_sec = fast_movement_deg_per_sec
        self.low_smoothness_threshold = low_smoothness_threshold
        self.low_symmetry_threshold = low_symmetry_threshold

    def evaluate(
        self,
        definition: ExerciseDefinition,
        assessment: RepetitionAssessment,
        errors: List[DetectedError],
        achieved_angle_deg: Optional[float] = None,
        peak_velocity_deg_per_sec: Optional[float] = None,
    ) -> List[ClinicalReasoningResult]:
        """Run every applicable rule and return the ones that fired."""
        results: List[ClinicalReasoningResult] = []

        results += self._rule_rom_deficit(definition, assessment, achieved_angle_deg)
        results += self._rule_trunk_compensation(definition, errors)
        results += self._rule_elbow_extension_limited(definition, achieved_angle_deg)
        results += self._rule_fast_movement(peak_velocity_deg_per_sec)
        results += self._rule_low_smoothness(assessment)
        results += self._rule_compensation_severity(assessment)

        return results

    # ------------------------------------------------------------------
    # Individual rules
    # ------------------------------------------------------------------

    def _rule_rom_deficit(
        self, definition: ExerciseDefinition, assessment: RepetitionAssessment, achieved_angle_deg: Optional[float],
    ) -> List[ClinicalReasoningResult]:
        if assessment.range_of_motion_score / 100.0 >= self.rom_deficit_ratio_threshold:
            return []

        exercise_name = definition.display_name
        deficit_pct = 100.0 - assessment.range_of_motion_score
        condition = f"{exercise_name} range of motion reached only {assessment.range_of_motion_score:.0f}% of the expected range."

        # Give shoulder-specific and general phrasing, matching the spec's example.
        if "shoulder" in definition.key:
            recommendation = f"Recommend increasing {exercise_name.lower()} mobility exercises."
        elif "elbow" in definition.key:
            recommendation = f"Suggest controlled {exercise_name.lower()} exercises to gradually extend range."
        else:
            recommendation = f"Gradually increase the range of motion for {exercise_name.lower()}."

        explanation = (
            f"The measured range of motion fell {deficit_pct:.0f}% short of the exercise's expected "
            f"neutral-to-target span — consistent with reduced mobility rather than a single off repetition."
        )
        return [ClinicalReasoningResult(condition, recommendation, explanation, rule_id="rom_deficit")]

    def _rule_trunk_compensation(self, definition: ExerciseDefinition, errors: List[DetectedError]) -> List[ClinicalReasoningResult]:
        trunk_errors = [e for e in errors if e.error_type == ErrorType.TRUNK_COMPENSATION]
        if not trunk_errors:
            return []

        lean_values = [e.detail.get("trunk_lean_deg", self.trunk_lean_threshold_deg) for e in trunk_errors]
        max_lean = max(lean_values)
        condition = f"Trunk lean reached {max_lean:.0f}\u00b0, exceeding the {self.trunk_lean_threshold_deg:.0f}\u00b0 compensation threshold."
        recommendation = "Flag trunk compensation and advise maintaining an upright posture throughout the movement."
        explanation = (
            "Excessive trunk lean during an arm movement usually indicates the patient is using torso "
            "motion to substitute for limited shoulder/arm range, rather than isolating the target joint."
        )
        return [ClinicalReasoningResult(condition, recommendation, explanation, rule_id="trunk_compensation")]

    def _rule_elbow_extension_limited(self, definition: ExerciseDefinition, achieved_angle_deg: Optional[float]) -> List[ClinicalReasoningResult]:
        if definition.key != "elbow_extension" or achieved_angle_deg is None:
            return []
        if achieved_angle_deg >= self.elbow_extension_limited_deg:
            return []

        condition = f"Elbow extension reached only {achieved_angle_deg:.0f}\u00b0, below the {self.elbow_extension_limited_deg:.0f}\u00b0 functional threshold."
        recommendation = "Suggest controlled elbow extension exercises, progressing range gradually within a comfortable limit."
        explanation = "Limited terminal elbow extension restricts reaching and functional arm use in daily activities."
        return [ClinicalReasoningResult(condition, recommendation, explanation, rule_id="elbow_extension_limited")]

    def _rule_fast_movement(self, peak_velocity_deg_per_sec: Optional[float]) -> List[ClinicalReasoningResult]:
        if peak_velocity_deg_per_sec is None or abs(peak_velocity_deg_per_sec) <= self.fast_movement_deg_per_sec:
            return []

        condition = f"Peak movement speed reached {abs(peak_velocity_deg_per_sec):.0f}\u00b0/s, above the {self.fast_movement_deg_per_sec:.0f}\u00b0/s controlled-movement threshold."
        recommendation = "Recommend slower, controlled repetitions rather than rapid movement."
        explanation = "Fast, ballistic movement reduces neuromuscular control and increases risk of compensatory patterns and injury during rehabilitation."
        return [ClinicalReasoningResult(condition, recommendation, explanation, rule_id="fast_movement")]

    def _rule_low_smoothness(self, assessment: RepetitionAssessment) -> List[ClinicalReasoningResult]:
        if assessment.movement_smoothness_score >= self.low_smoothness_threshold:
            return []

        condition = f"Movement smoothness score was {assessment.movement_smoothness_score:.0f}/100, indicating jerky rather than fluid motion."
        recommendation = "Recommend slower, more controlled repetitions with steady pacing throughout the movement."
        explanation = "Jerky, non-smooth trajectories are associated with impaired motor control and are a common target for improvement in early-stage stroke rehabilitation."
        return [ClinicalReasoningResult(condition, recommendation, explanation, rule_id="low_smoothness")]

    def _rule_compensation_severity(self, assessment: RepetitionAssessment) -> List[ClinicalReasoningResult]:
        if assessment.compensation_severity_score >= 70.0:
            return []

        condition = f"Compensation severity score was {assessment.compensation_severity_score:.0f}/100, indicating frequent or significant compensatory movement."
        recommendation = "Reduce repetitions per set temporarily and prioritize form correction over volume."
        explanation = "Persistent compensation patterns can reinforce non-functional movement strategies if practiced repeatedly without correction."
        return [ClinicalReasoningResult(condition, recommendation, explanation, rule_id="compensation_severity")]
