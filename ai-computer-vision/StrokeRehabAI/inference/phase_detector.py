"""
phase_detector.py
===================
Tracks which phase of a repetition the patient is currently in, for the
active exercise: neutral (starting position) -> moving-toward-target ->
peak (near target, holding) -> returning-to-neutral -> back at neutral
(rep complete). Phase is derived from the primary tracked angle's value
and its recent velocity, using the exercise's neutral/target angles
from `configs/exercises.yaml`.

Phase drives both the rep counter (a completed rep = a full
neutral -> peak -> neutral cycle) and the "Movement Progress" /
"Completion Percentage" outputs required by the real-time analysis.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Optional

from inference.exercise_library import ExerciseDefinition
from utils.logger import get_logger

logger = get_logger(__name__)


class ExercisePhase(str, Enum):
    NEUTRAL = "neutral"
    MOVING_TO_TARGET = "moving_to_target"
    PEAK = "peak"
    RETURNING = "returning"
    UNKNOWN = "unknown"


@dataclass
class PhaseResult:
    phase: ExercisePhase
    progress_fraction: float          # 0.0 (neutral) .. 1.0 (at target)
    current_angle_deg: float
    velocity_deg_per_sec: float


class PhaseDetector:
    """Per-exercise phase state machine driven by angle value + velocity."""

    def __init__(
        self,
        fps: float = 30.0,
        near_neutral_fraction: float = 0.15,
        near_target_fraction: float = 0.85,
        velocity_still_threshold_deg_per_sec: float = 15.0,
        velocity_window: int = 5,
    ):
        self.dt = 1.0 / fps if fps > 0 else 1.0 / 30.0
        self.near_neutral_fraction = near_neutral_fraction
        self.near_target_fraction = near_target_fraction
        self.velocity_still_threshold = velocity_still_threshold_deg_per_sec
        self._angle_history: Deque[float] = deque(maxlen=velocity_window)
        self._phase = ExercisePhase.NEUTRAL

    def reset(self) -> None:
        self._angle_history.clear()
        self._phase = ExercisePhase.NEUTRAL

    def update(self, current_angle_deg: float, definition: ExerciseDefinition) -> PhaseResult:
        self._angle_history.append(current_angle_deg)
        velocity = self._estimate_velocity()
        progress = definition.progress_fraction(current_angle_deg)

        self._phase = self._next_phase(progress, velocity, definition)

        return PhaseResult(
            phase=self._phase,
            progress_fraction=progress,
            current_angle_deg=current_angle_deg,
            velocity_deg_per_sec=velocity,
        )

    def _estimate_velocity(self) -> float:
        if len(self._angle_history) < 2:
            return 0.0
        values = list(self._angle_history)
        return (values[-1] - values[0]) / (self.dt * (len(values) - 1))

    def _next_phase(self, progress: float, velocity: float, definition: ExerciseDefinition) -> ExercisePhase:
        if progress <= self.near_neutral_fraction:
            return ExercisePhase.NEUTRAL

        if progress >= self.near_target_fraction:
            return ExercisePhase.PEAK

        # Mid-range: is the patient still moving toward the target, or
        # coming back? "Toward target" means the angle is changing in the
        # same direction the exercise's rep_direction expects.
        moving_toward_target = (velocity > 0) == (definition.rep_direction == "increasing")
        is_still = abs(velocity) < self.velocity_still_threshold

        if is_still:
            # No clear velocity signal — stay in whatever phase we were
            # already progressing through, defaulting to "moving toward
            # target" if we have no prior phase to anchor on.
            return self._phase if self._phase != ExercisePhase.UNKNOWN else ExercisePhase.MOVING_TO_TARGET

        return ExercisePhase.MOVING_TO_TARGET if moving_toward_target else ExercisePhase.RETURNING
