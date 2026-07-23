"""
feedback_engine.py
====================
Translates MovementAnalysisResult + DetectedError lists into natural
language feedback for the patient — never bare "Wrong"/"Correct", and
never simply restating an error code. Each error type maps to a small
set of specific, actionable phrasings (e.g. "Raise your elbow
slightly", "Avoid leaning your torso") chosen based on the error's
detail and the exercise's rep direction, so the same error type reads
differently for, say, elbow flexion vs. shoulder abduction.

TODO (next development phase): once labeled session data + clinician
feedback exists, consider generating phrasing with a small template
-selection model rather than the current rule-based phrase bank, to
better vary wording across a long session.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from inference.error_detector import DetectedError, ErrorType
from inference.exercise_library import ExerciseDefinition
from inference.movement_analyzer import MovementAnalysisResult
from inference.phase_detector import ExercisePhase
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FeedbackMessage:
    severity: str  # "info" | "warning" | "error"
    text: str
    error_type: Optional[ErrorType] = None


# Phrase banks keyed by error type; {joint} / {direction} are filled in
# from the exercise definition and error detail at generation time.
_PHRASE_BANK: Dict[ErrorType, List[str]] = {
    ErrorType.INSUFFICIENT_ROM: [
        "Try to {direction} a little further.",
        "You can push a bit more into the movement.",
        "Reach a little farther if it's comfortable.",
    ],
    ErrorType.EXCESSIVE_ROM: [
        "That's further than needed — ease back slightly.",
        "No need to push quite that far.",
    ],
    ErrorType.INCOMPLETE_REPETITION: [
        "That one didn't quite finish — let's try the full movement again.",
        "Almost there — try completing the full range next time.",
    ],
    ErrorType.FAST_MOVEMENT: [
        "Slow down your movement a bit.",
        "Try a slower, more controlled pace.",
    ],
    ErrorType.JERKY_MOVEMENT: [
        "Try to keep the motion smooth and steady.",
        "Aim for a smoother, more even movement.",
    ],
    ErrorType.TRUNK_COMPENSATION: [
        "Avoid leaning your torso — keep your trunk upright.",
        "Try to keep your body still and let your arm do the work.",
    ],
    ErrorType.BODY_LEAN: [
        "Straighten up a little — you're leaning to one side.",
    ],
    ErrorType.SHOULDER_HIKING: [
        "Relax your shoulder — try not to lift it toward your ear.",
        "Lower your shoulder a little and keep it relaxed.",
    ],
    ErrorType.POOR_ALIGNMENT: [
        "Try to keep your shoulders and hips facing forward.",
        "Square up your shoulders with your hips.",
    ],
    ErrorType.ASYMMETRICAL_MOTION: [
        "Try to move both sides evenly.",
        "Keep the movement balanced on both sides.",
    ],
    ErrorType.INCORRECT_SEQUENCE: [
        "This looks like a different exercise than expected — let's switch back.",
    ],
    ErrorType.LATE_MOVEMENT: [
        "Whenever you're ready, go ahead and begin the movement.",
    ],
    ErrorType.PREMATURE_RETURN: [
        "Try holding a little longer before returning.",
        "Give it a moment at the top before coming back down.",
    ],
    ErrorType.INCORRECT_JOINT_ANGLE: [
        "Check your starting position against how we calibrated.",
    ],
}

_POSITIVE_PHRASES = [
    "Excellent repetition.",
    "Great job — nice and controlled.",
    "Good posture, keep it up.",
    "That was a smooth, complete movement.",
]

_ENCOURAGEMENT_PHRASES = [
    "Almost complete.",
    "You're getting closer — keep going.",
    "Nice progress on that rep.",
]


class FeedbackEngine:
    """Generates natural-language feedback from movement analysis + errors."""

    def __init__(self, quality_excellent_threshold: float = 0.92, low_confidence_threshold: float = 0.5, rng: Optional[random.Random] = None):
        self.quality_excellent_threshold = quality_excellent_threshold
        self.low_confidence_threshold = low_confidence_threshold
        self._rng = rng or random.Random()
        self._last_phrase_by_error: Dict[ErrorType, str] = {}

    def generate(self, result: MovementAnalysisResult, definition: Optional[ExerciseDefinition] = None) -> List[FeedbackMessage]:
        messages: List[FeedbackMessage] = []

        if result.exercise_key is None:
            messages.append(FeedbackMessage("info", "Begin your exercise whenever you're ready."))
            return messages

        if result.overall_confidence < self.low_confidence_threshold:
            messages.append(FeedbackMessage("warning", "Make sure your full upper body is visible to the camera."))

        for error in result.errors:
            phrase = self._phrase_for(error, definition)
            severity = "error" if error.severity == "severe" else "warning"
            messages.append(FeedbackMessage(severity, phrase, error_type=error.error_type))

        if not result.errors:
            if result.rep_event is not None and result.rep_event.completed:
                messages.append(FeedbackMessage("info", self._pick(_POSITIVE_PHRASES)))
            elif result.phase == ExercisePhase.PEAK:
                messages.append(FeedbackMessage("info", self._pick(_ENCOURAGEMENT_PHRASES)))

        return messages

    def _phrase_for(self, error: DetectedError, definition: Optional[ExerciseDefinition]) -> str:
        bank = _PHRASE_BANK.get(error.error_type, ["Adjust your form slightly."])
        # Avoid repeating the exact same phrase twice in a row for the same error type.
        previous = self._last_phrase_by_error.get(error.error_type)
        candidates = [p for p in bank if p != previous] or bank
        phrase = self._pick(candidates)
        self._last_phrase_by_error[error.error_type] = phrase

        direction = "reach further" if (definition and definition.rep_direction == "increasing") else "bend further"
        return phrase.format(direction=direction, joint=definition.display_name if definition else "joint")

    def _pick(self, options: List[str]) -> str:
        return self._rng.choice(options)
