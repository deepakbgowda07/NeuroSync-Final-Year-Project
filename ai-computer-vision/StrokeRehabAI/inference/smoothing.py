"""
smoothing.py
==============
Consolidated smoothing utilities for the real-time pipeline, beyond
the landmark-level EMA already provided by
`mediapipe_pipeline.pose_smoothing.LandmarkSmoother`:

    - ConfidenceSmoother: damps frame-to-frame confidence/visibility jitter.
    - AngleSmoother: per-angle EMA smoothing (keyed by angle name), so
      each tracked joint angle gets independent, appropriately-scaled
      smoothing rather than one blanket landmark-level filter.
    - PredictionSmoother: majority-vote smoothing over a rolling window
      for categorical predictions (recognized exercise, error presence),
      so a single noisy frame can't flip a displayed prediction.

`ExerciseRecognizer` and `ViewDetector` already implement their own
majority-vote smoothing internally; `PredictionSmoother` here is the
general-purpose version for any other categorical signal the pipeline
wants to stabilize (e.g. a per-frame movement-quality label).
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Generic, Hashable, Optional, TypeVar

from utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=Hashable)


class ConfidenceSmoother:
    """Simple EMA smoother for a scalar confidence/visibility value."""

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self._value: Optional[float] = None

    def reset(self) -> None:
        self._value = None

    def smooth(self, value: float) -> float:
        if self._value is None:
            self._value = value
        else:
            self._value = self.alpha * value + (1 - self.alpha) * self._value
        return self._value


class AngleSmoother:
    """Per-angle-name EMA smoothing, so e.g. `left_elbow_angle` and
    `left_forearm_rotation_proxy` (a noisier signal) can be tuned
    independently without one shared smoothing factor."""

    def __init__(self, default_alpha: float = 0.4, per_angle_alpha: Optional[Dict[str, float]] = None):
        self.default_alpha = default_alpha
        self.per_angle_alpha = per_angle_alpha or {}
        self._state: Dict[str, float] = {}

    def reset(self) -> None:
        self._state.clear()

    def smooth(self, angles: Dict[str, float]) -> Dict[str, float]:
        smoothed = {}
        for name, value in angles.items():
            alpha = self.per_angle_alpha.get(name, self.default_alpha)
            if name not in self._state:
                self._state[name] = value
            else:
                self._state[name] = alpha * value + (1 - alpha) * self._state[name]
            smoothed[name] = self._state[name]
        return smoothed


class PredictionSmoother(Generic[T]):
    """Majority-vote smoothing over a rolling window for any hashable
    categorical prediction (exercise key, discrete quality label, etc.)."""

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self._history: Deque[T] = deque(maxlen=window_size)

    def reset(self) -> None:
        self._history.clear()

    def smooth(self, prediction: T) -> "tuple[T, float]":
        self._history.append(prediction)
        counts: Dict[T, int] = {}
        for pred in self._history:
            counts[pred] = counts.get(pred, 0) + 1
        best = max(counts, key=counts.get)
        confidence = counts[best] / len(self._history)
        return best, confidence
