"""
rep_tracker.py
================
Intelligent, exercise-specific repetition counting built on top of
`PhaseDetector`'s phase stream. A rep is only counted on a complete
neutral -> peak -> neutral cycle, which naturally:

    - ignores accidental/incidental small movements (never reaches PEAK,
      so never starts a countable rep),
    - handles pauses at any phase (phase just stops changing; the state
      machine waits rather than timing out mid-rep), and
    - handles incomplete repetitions explicitly (a movement that reaches
      PEAK but then loses tracking, or reverses direction without truly
      returning to neutral, is flagged rather than silently counted).

This is distinct from `feature_extraction/rep_counter.py`, which
peak-detects over a *complete, offline* angle sequence — this module is
the streaming, real-time counterpart used by the live inference loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from inference.phase_detector import ExercisePhase, PhaseResult
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RepEvent:
    rep_number: int
    completed: bool
    peak_progress_fraction: float
    duration_frames: int
    flagged_incomplete: bool = False
    flagged_premature_return: bool = False


@dataclass
class RepTrackerState:
    rep_count: int = 0
    incomplete_rep_count: int = 0
    current_rep_frames: int = 0
    reached_peak_this_rep: bool = False
    peak_progress_this_rep: float = 0.0
    premature_return_flagged_this_rep: bool = False
    last_event: Optional[RepEvent] = None


class RepTracker:
    """Streaming rep counter driven by per-frame PhaseResult updates."""

    def __init__(
        self,
        min_peak_progress: float = 0.7,
        min_rep_duration_frames: int = 8,
        max_rep_duration_frames: int = 900,
        premature_return_progress_threshold: float = 0.3,
    ):
        self.min_peak_progress = min_peak_progress
        self.min_rep_duration_frames = min_rep_duration_frames
        self.max_rep_duration_frames = max_rep_duration_frames
        self.premature_return_progress_threshold = premature_return_progress_threshold

        self.state = RepTrackerState()
        self._previous_phase: Optional[ExercisePhase] = None
        self._events: List[RepEvent] = []

    def reset(self) -> None:
        self.state = RepTrackerState()
        self._previous_phase = None
        self._events.clear()

    def update(self, phase_result: PhaseResult) -> Optional[RepEvent]:
        """Feed one frame's phase result. Returns a RepEvent only on the
        frame a rep (complete or flagged-incomplete) is finalized."""
        phase = phase_result.phase
        event: Optional[RepEvent] = None

        if phase != ExercisePhase.NEUTRAL:
            self.state.current_rep_frames += 1

        if phase == ExercisePhase.PEAK and phase_result.progress_fraction >= self.min_peak_progress:
            self.state.reached_peak_this_rep = True
            self.state.peak_progress_this_rep = max(self.state.peak_progress_this_rep, phase_result.progress_fraction)

        # Detect a "premature return": moving back toward neutral without
        # ever having gotten reasonably close to the target first.
        premature = (
            phase == ExercisePhase.RETURNING
            and not self.state.reached_peak_this_rep
            and phase_result.progress_fraction < self.premature_return_progress_threshold
            and self._previous_phase == ExercisePhase.MOVING_TO_TARGET
        )
        if premature:
            self.state.premature_return_flagged_this_rep = True

        # A rep finalizes the moment we arrive back at NEUTRAL after having
        # left it (current_rep_frames > 0 means we were mid-movement).
        if phase == ExercisePhase.NEUTRAL and self._previous_phase in (
            ExercisePhase.RETURNING, ExercisePhase.PEAK, ExercisePhase.MOVING_TO_TARGET,
        ) and self.state.current_rep_frames > 0:
            event = self._finalize_rep()

        if self.state.current_rep_frames > self.max_rep_duration_frames:
            logger.warning("Rep exceeded max duration (%d frames); force-finalizing as incomplete.", self.max_rep_duration_frames)
            event = self._finalize_rep(force_incomplete=True)

        self._previous_phase = phase
        return event

    def _finalize_rep(self, force_incomplete: bool = False) -> RepEvent:
        duration = self.state.current_rep_frames
        reached_peak = self.state.reached_peak_this_rep
        long_enough = duration >= self.min_rep_duration_frames

        completed = reached_peak and long_enough and not force_incomplete
        flagged_incomplete = not completed

        if completed:
            self.state.rep_count += 1
            rep_number = self.state.rep_count
            logger.info("Rep #%d completed (peak progress=%.0f%%, duration=%d frames).", rep_number, self.state.peak_progress_this_rep * 100, duration)
        else:
            self.state.incomplete_rep_count += 1
            rep_number = self.state.rep_count  # incomplete reps don't advance the count
            logger.info(
                "Incomplete repetition detected (reached_peak=%s, duration=%d frames) — not counted.",
                reached_peak, duration,
            )

        event = RepEvent(
            rep_number=rep_number,
            completed=completed,
            peak_progress_fraction=self.state.peak_progress_this_rep,
            duration_frames=duration,
            flagged_incomplete=flagged_incomplete,
            flagged_premature_return=self.state.premature_return_flagged_this_rep,
        )
        self._events.append(event)
        self.state.last_event = event

        # Reset per-rep tracking for the next cycle.
        self.state.current_rep_frames = 0
        self.state.reached_peak_this_rep = False
        self.state.peak_progress_this_rep = 0.0
        self.state.premature_return_flagged_this_rep = False

        return event

    @property
    def rep_count(self) -> int:
        return self.state.rep_count

    @property
    def events(self) -> List[RepEvent]:
        return list(self._events)
