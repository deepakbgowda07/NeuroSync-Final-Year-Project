"""
movement_analyzer.py
======================
The per-frame real-time analysis core: given the current landmarks and
joint angles, determines the current exercise, current phase, expected
vs. actual joint angles, movement progress / completion percentage,
a movement-quality score, and an overall confidence — the exact set of
outputs required by the real-time movement-analysis spec.

This is the orchestration layer that `inference/realtime_pipeline.py`
calls once per frame; it composes `ExerciseRecognizer`,
`PhaseDetector`, `ErrorDetector`, and `RepTracker` rather than
duplicating their logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from inference.calibration import CalibrationProfile
from inference.error_detector import DetectedError, ErrorDetector
from inference.exercise_library import ExerciseLibrary
from inference.exercise_recognizer import ExerciseRecognizer
from inference.phase_detector import ExercisePhase, PhaseDetector, PhaseResult
from inference.rep_tracker import RepEvent, RepTracker
from mediapipe_pipeline.view_detector import ViewDetectionResult
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MovementAnalysisResult:
    exercise_key: Optional[str]
    exercise_display_name: Optional[str]
    exercise_recognition_confidence: float
    phase: Optional[ExercisePhase]
    expected_angle_deg: Optional[float]
    actual_angle_deg: Optional[float]
    progress_fraction: float
    completion_percentage: float
    movement_quality: float          # 0-1, 1.0 = no errors detected
    overall_confidence: float        # combines recognition confidence + pose confidence
    rep_count: int
    errors: List[DetectedError] = field(default_factory=list)
    rep_event: Optional[RepEvent] = None


class MovementAnalyzer:
    """Combines exercise recognition, phase detection, rep tracking, and
    error detection into one per-frame analysis result."""

    def __init__(self, exercise_library: ExerciseLibrary, fps: float = 30.0, exercises_cfg=None):
        self.library = exercise_library
        self.fps = fps

        recognition_cfg = exercises_cfg.recognition if exercises_cfg and hasattr(exercises_cfg, "recognition") else None
        self.recognizer = ExerciseRecognizer(
            exercise_library,
            confidence_window_frames=recognition_cfg.confidence_window_frames if recognition_cfg else 20,
            min_recognition_confidence=recognition_cfg.min_recognition_confidence if recognition_cfg else 0.55,
        )
        self.error_detector = ErrorDetector(fps=fps)

        self._phase_detectors: Dict[str, PhaseDetector] = {}
        self._rep_trackers: Dict[str, RepTracker] = {}
        self._active_exercise_key: Optional[str] = None
        self._expected_exercise_key: Optional[str] = None

    def reset(self) -> None:
        self.recognizer.reset()
        self.error_detector.reset()
        self._phase_detectors.clear()
        self._rep_trackers.clear()
        self._active_exercise_key = None

    def set_expected_exercise(self, exercise_key: Optional[str]) -> None:
        """Optionally tell the analyzer which exercise a structured
        program expects next, enabling the "Incorrect Exercise Sequence"
        error check. Leave as None for free-form/unstructured sessions."""
        self._expected_exercise_key = exercise_key

    def analyze_frame(
        self,
        landmarks_xyz: np.ndarray,
        angles: Dict[str, float],
        pose_confidence: float,
        view: Optional[ViewDetectionResult] = None,
        calibration: Optional[CalibrationProfile] = None,
    ) -> MovementAnalysisResult:
        """Run the full per-frame analysis. `pose_confidence` should come
        from `LandmarkExtractor.overall_confidence()`."""
        recognition = self.recognizer.update(angles, view=view)
        exercise_key = recognition.exercise_key
        self._active_exercise_key = exercise_key or self._active_exercise_key

        if exercise_key is None:
            return MovementAnalysisResult(
                exercise_key=None, exercise_display_name=None,
                exercise_recognition_confidence=recognition.confidence,
                phase=None, expected_angle_deg=None, actual_angle_deg=None,
                progress_fraction=0.0, completion_percentage=0.0,
                movement_quality=1.0, overall_confidence=pose_confidence * recognition.confidence,
                rep_count=0, errors=[],
            )

        definition = self.library.get(exercise_key)
        phase_detector = self._phase_detectors.setdefault(exercise_key, PhaseDetector(fps=self.fps))
        rep_tracker = self._rep_trackers.setdefault(exercise_key, RepTracker())

        current_angle = angles.get(definition.primary_angle, float("nan"))
        if np.isnan(current_angle):
            current_angle = angles.get(definition.secondary_angle, float("nan"))

        phase_result = phase_detector.update(current_angle, definition)
        rep_event = rep_tracker.update(phase_result)

        errors = self.error_detector.detect(
            landmarks_xyz, angles, definition, phase_result,
            calibration=calibration, recognized_exercise_key=exercise_key,
            expected_exercise_key=self._expected_exercise_key,
        )
        if rep_event is not None:
            incomplete_error = self.error_detector.register_incomplete_rep(rep_event)
            if incomplete_error is not None:
                errors.append(incomplete_error)
            premature_error = self.error_detector.check_premature_return(rep_event)
            if premature_error is not None:
                errors.append(premature_error)

        quality = self._compute_quality_score(errors)
        overall_confidence = float(np.clip(pose_confidence * recognition.confidence, 0.0, 1.0))

        return MovementAnalysisResult(
            exercise_key=exercise_key,
            exercise_display_name=definition.display_name,
            exercise_recognition_confidence=recognition.confidence,
            phase=phase_result.phase,
            expected_angle_deg=definition.target_deg,
            actual_angle_deg=current_angle,
            progress_fraction=phase_result.progress_fraction,
            completion_percentage=phase_result.progress_fraction * 100.0,
            movement_quality=quality,
            overall_confidence=overall_confidence,
            rep_count=rep_tracker.rep_count,
            errors=errors,
            rep_event=rep_event,
        )

    @staticmethod
    def _compute_quality_score(errors: List[DetectedError]) -> float:
        """Simple, transparent quality score: start at 1.0 and deduct a
        severity-weighted penalty per active error, floored at 0.0.
        TODO: replace with a model-driven quality score (the trained
        ST-GCN classifier, see models/stgcn_model.py) once labeled
        session data is available to validate against.
        """
        penalty_by_severity = {"minor": 0.08, "moderate": 0.18, "severe": 0.35}
        penalty = sum(penalty_by_severity.get(e.severity, 0.1) for e in errors)
        return float(np.clip(1.0 - penalty, 0.0, 1.0))
