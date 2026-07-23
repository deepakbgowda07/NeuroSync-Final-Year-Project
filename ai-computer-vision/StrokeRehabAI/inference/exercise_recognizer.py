"""
exercise_recognizer.py
========================
Automatically recognizes which of the 10 supported exercises the
patient is currently performing, from a rolling buffer of per-frame
joint angles — the patient never selects an exercise manually.

Approach: for each candidate exercise, score how well the recent angle
trajectory's *range* and *direction of change* on that exercise's
primary tracked angle matches the exercise's defined characteristics
(neutral/target span, expected view). This is a transparent, tunable
rule-based recognizer that requires no labeled training data — a
natural place to later swap in the trained ST-GCN classifier
(models/stgcn_model.py, see training/trainer.py) as an additional or
replacement signal once labeled sessions exist.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

import numpy as np

from inference.exercise_library import ExerciseDefinition, ExerciseLibrary
from mediapipe_pipeline.view_detector import ViewDetectionResult
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RecognitionResult:
    exercise_key: Optional[str]
    confidence: float
    candidate_scores: Dict[str, float]


class ExerciseRecognizer:
    """Scores each candidate exercise against a rolling angle-observation
    buffer and returns the best match, temporally smoothed so the
    recognized exercise doesn't flicker frame-to-frame.
    """

    def __init__(
        self,
        library: ExerciseLibrary,
        buffer_size: int = 45,
        confidence_window_frames: int = 20,
        min_recognition_confidence: float = 0.55,
    ):
        self.library = library
        self.buffer_size = buffer_size
        self.confidence_window_frames = confidence_window_frames
        self.min_recognition_confidence = min_recognition_confidence

        self._angle_buffers: Dict[str, Deque[float]] = {}
        self._recent_predictions: Deque[Optional[str]] = deque(maxlen=confidence_window_frames)

    def reset(self) -> None:
        self._angle_buffers.clear()
        self._recent_predictions.clear()

    def update(self, angles: Dict[str, float], view: Optional[ViewDetectionResult] = None) -> RecognitionResult:
        """Feed one frame's joint angles (from utils.joint_angles.compute_all_joint_angles)
        and return the current best-guess exercise, with per-candidate scores."""
        for name, value in angles.items():
            if not np.isnan(value):
                self._angle_buffers.setdefault(name, deque(maxlen=self.buffer_size)).append(value)

        scores = {key: self._score_candidate(defn, view) for key, defn in self.library.all().items()}
        best_key = max(scores, key=scores.get) if scores else None
        best_score = scores.get(best_key, 0.0) if best_key else 0.0

        predicted = best_key if best_score >= self.min_recognition_confidence else None
        self._recent_predictions.append(predicted)

        smoothed_key, smoothed_confidence = self._majority_vote()
        return RecognitionResult(exercise_key=smoothed_key, confidence=smoothed_confidence, candidate_scores=scores)

    def _score_candidate(self, defn: ExerciseDefinition, view: Optional[ViewDetectionResult]) -> float:
        left_buffer = self._angle_buffers.get(defn.primary_angle)
        right_buffer = self._angle_buffers.get(defn.secondary_angle)

        buffers = [b for b in (left_buffer, right_buffer) if b and len(b) >= 5]
        if not buffers:
            return 0.0

        observed_range = max(max(b) - min(b) for b in buffers)
        expected_span = max(defn.rom_span(), 1e-6)
        range_score = float(np.clip(observed_range / expected_span, 0.0, 1.0))

        # Direction agreement: does the most recent trend move toward the target?
        direction_score = 0.0
        for b in buffers:
            if len(b) < 2:
                continue
            delta = list(b)[-1] - list(b)[0]
            moving_toward_target = (delta > 0) == (defn.rep_direction == "increasing")
            direction_score = max(direction_score, 1.0 if moving_toward_target else 0.3)

        view_score = 1.0
        if view is not None:
            from mediapipe_pipeline.view_detector import ViewDetector

            view_score = 1.0 if ViewDetector.matches_required_view(view, defn.required_view) else 0.5

        return float(np.clip(0.55 * range_score + 0.25 * direction_score + 0.20 * view_score, 0.0, 1.0))

    def _majority_vote(self) -> "tuple[Optional[str], float]":
        if not self._recent_predictions:
            return None, 0.0

        counts: Dict[Optional[str], int] = {}
        for pred in self._recent_predictions:
            counts[pred] = counts.get(pred, 0) + 1

        best = max(counts, key=counts.get)
        confidence = counts[best] / len(self._recent_predictions)
        return best, confidence
