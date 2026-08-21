"""
clinical_assessment.py
=========================
The AI Clinical Assessment Engine: evaluates rehabilitation *quality*
for each completed repetition — not just "which exercise" (that's
`inference/exercise_recognizer.py`) — producing eight normalized 0-100
scores:

    Exercise Quality Score        Movement Consistency
    Range of Motion Score          Compensation Severity
    Movement Smoothness Score       Exercise Completion Score
    Joint Stability Score            Overall Rehabilitation Score

This operates on the buffered per-frame data collected *during one
repetition* (see `inference/movement_analyzer.py`'s `_rep_buffers`),
distinct from `dashboard/clinical_metrics.py`, which scores an entire
*session* after the fact from stored database rows. The two are
complementary: this module is the real-time, single-repetition
assessment the patient sees immediately; the dashboard module
aggregates many stored sessions for longitudinal tracking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from inference.error_detector import DetectedError
from inference.exercise_library import ExerciseDefinition
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RepFrameSample:
    """One frame's worth of data buffered during a repetition, the
    minimal input ClinicalAssessmentEngine needs — decoupled from
    MovementAnalyzer's internals so this module stays independently
    testable."""

    angle_deg: float
    movement_quality: float           # 0-1, from MovementAnalyzer
    trunk_lean_deg: float
    shoulder_elevation: float
    errors: List[DetectedError]


@dataclass
class RepetitionAssessment:
    exercise_quality_score: float
    range_of_motion_score: float
    movement_smoothness_score: float
    joint_stability_score: float
    movement_consistency_score: float
    compensation_severity_score: float
    exercise_completion_score: float
    overall_rehabilitation_score: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "exercise_quality_score": self.exercise_quality_score,
            "range_of_motion_score": self.range_of_motion_score,
            "movement_smoothness_score": self.movement_smoothness_score,
            "joint_stability_score": self.joint_stability_score,
            "movement_consistency_score": self.movement_consistency_score,
            "compensation_severity_score": self.compensation_severity_score,
            "exercise_completion_score": self.exercise_completion_score,
            "overall_rehabilitation_score": self.overall_rehabilitation_score,
        }


def _clip_0_100(value: float) -> float:
    return float(np.clip(value, 0.0, 100.0))


class ClinicalAssessmentEngine:
    """Computes the eight-score assessment suite for one completed repetition.

    TODO (next development phase): scoring weights are principled
    engineering defaults, not clinician-validated — same caveat as
    dashboard/clinical_metrics.py and inference/exercise_library.py.
    """

    COMPENSATION_ERROR_TYPES = {"trunk_compensation", "shoulder_hiking", "body_lean", "poor_alignment"}

    def assess_repetition(
        self,
        frames: List[RepFrameSample],
        definition: ExerciseDefinition,
        rep_completed: bool,
        peak_progress_fraction: float,
    ) -> RepetitionAssessment:
        if not frames:
            return self._empty_assessment()

        rom_score = self._range_of_motion_score(frames, definition)
        smoothness_score = self._movement_smoothness_score(frames)
        stability_score = self._joint_stability_score(frames)
        consistency_score = self._movement_consistency_score(frames)
        compensation_score = self._compensation_severity_score(frames)
        completion_score = self._exercise_completion_score(rep_completed, peak_progress_fraction)
        quality_score = self._exercise_quality_score(frames, completion_score)

        overall = _clip_0_100(
            0.25 * quality_score + 0.20 * rom_score + 0.15 * smoothness_score +
            0.15 * stability_score + 0.10 * consistency_score +
            0.10 * compensation_score + 0.05 * completion_score
        )

        return RepetitionAssessment(
            exercise_quality_score=quality_score,
            range_of_motion_score=rom_score,
            movement_smoothness_score=smoothness_score,
            joint_stability_score=stability_score,
            movement_consistency_score=consistency_score,
            compensation_severity_score=compensation_score,
            exercise_completion_score=completion_score,
            overall_rehabilitation_score=overall,
        )

    # ------------------------------------------------------------------
    # Individual scores
    # ------------------------------------------------------------------

    @staticmethod
    def _range_of_motion_score(frames: List[RepFrameSample], definition: ExerciseDefinition) -> float:
        """Ratio of the repetition's achieved ROM to the exercise's
        expected neutral-to-target span."""
        angles = [f.angle_deg for f in frames]
        achieved_rom = max(angles) - min(angles)
        expected_rom = max(definition.rom_span(), 1e-6)
        return _clip_0_100((achieved_rom / expected_rom) * 100.0)

    @staticmethod
    def _movement_smoothness_score(frames: List[RepFrameSample]) -> float:
        """Inverse of angle jerk (third derivative) — smoother
        repetitions show low frame-to-frame acceleration change."""
        angles = np.array([f.angle_deg for f in frames])
        if len(angles) < 4:
            return 50.0
        velocity = np.diff(angles)
        jerk = np.diff(np.diff(velocity))
        jerk_std = float(np.std(jerk)) if len(jerk) else 0.0
        return _clip_0_100(100.0 * max(0.0, 1.0 - jerk_std / 25.0))

    @staticmethod
    def _joint_stability_score(frames: List[RepFrameSample]) -> float:
        """Penalizes erratic frame-to-frame quality swings — a
        repetition where movement quality bounces around suggests
        unstable joint control, distinct from smoothness (which
        measures the angle trajectory itself)."""
        qualities = [f.movement_quality for f in frames]
        if len(qualities) < 2:
            return 50.0
        std = float(np.std(qualities))
        return _clip_0_100(100.0 * (1.0 - min(std * 3.0, 1.0)))

    @staticmethod
    def _movement_consistency_score(frames: List[RepFrameSample]) -> float:
        """Consistency of movement quality across the repetition's
        duration — distinct from stability (frame-to-frame variance) in
        that this measures whether quality trended/degraded across
        distinct phases of the rep (start vs. middle vs. end)."""
        qualities = [f.movement_quality for f in frames]
        if len(qualities) < 6:
            return 50.0
        third = len(qualities) // 3
        segments = [qualities[:third], qualities[third:2 * third], qualities[2 * third:]]
        segment_means = [float(np.mean(seg)) for seg in segments if seg]
        if len(segment_means) < 2:
            return 50.0
        spread = max(segment_means) - min(segment_means)
        return _clip_0_100(100.0 * (1.0 - min(spread * 2.0, 1.0)))

    @classmethod
    def _compensation_severity_score(cls, frames: List[RepFrameSample]) -> float:
        """100 = no compensation; decreases with both frequency and
        severity of compensation-type errors observed during the rep."""
        severity_weight = {"minor": 1.0, "moderate": 2.0, "severe": 3.5}
        total_weight = 0.0
        for frame in frames:
            for error in frame.errors:
                if error.error_type.value in cls.COMPENSATION_ERROR_TYPES:
                    total_weight += severity_weight.get(error.severity, 1.5)

        rate = total_weight / max(1, len(frames))
        return _clip_0_100(100.0 - rate * 40.0)

    @staticmethod
    def _exercise_completion_score(rep_completed: bool, peak_progress_fraction: float) -> float:
        if rep_completed:
            return _clip_0_100(80.0 + peak_progress_fraction * 20.0)
        return _clip_0_100(peak_progress_fraction * 60.0)  # partial credit for a genuine attempt

    @staticmethod
    def _exercise_quality_score(frames: List[RepFrameSample], completion_score: float) -> float:
        """Blend of average frame-level movement quality and completion —
        the single most representative "how good was this rep" number."""
        mean_quality = float(np.mean([f.movement_quality for f in frames])) * 100.0
        return _clip_0_100(0.7 * mean_quality + 0.3 * completion_score)

    @staticmethod
    def _empty_assessment() -> RepetitionAssessment:
        return RepetitionAssessment(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
