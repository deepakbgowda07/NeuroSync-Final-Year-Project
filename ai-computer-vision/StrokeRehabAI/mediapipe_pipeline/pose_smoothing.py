"""
pose_smoothing.py
==================
Temporal smoothing of landmark sequences to reduce frame-to-frame
jitter before joint-angle computation and visualization. MediaPipe's
built-in `smooth_landmarks` helps, but an additional configurable
filter is useful for noisy webcam conditions.

TODO (next development phase): implement a proper One-Euro filter
(Casiez et al. 2012) for adaptive smoothing that preserves fast
movements while damping jitter during near-static poses.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


class LandmarkSmoother:
    """Exponential-moving-average smoother over a rolling landmark buffer."""

    def __init__(self, alpha: float = 0.5, window_size: int = 5):
        self.alpha = alpha
        self.window_size = window_size
        self._history: Deque[np.ndarray] = deque(maxlen=window_size)
        self._ema_state: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._history.clear()
        self._ema_state = None

    def smooth(self, landmarks_xyz: np.ndarray) -> np.ndarray:
        """Apply EMA smoothing to a (33, 3) landmark array and return the
        smoothed result. Call once per frame with the latest raw landmarks."""
        if self._ema_state is None:
            self._ema_state = landmarks_xyz.copy()
        else:
            self._ema_state = self.alpha * landmarks_xyz + (1 - self.alpha) * self._ema_state

        self._history.append(self._ema_state.copy())
        return self._ema_state

    def windowed_average(self) -> Optional[np.ndarray]:
        """Return the simple average of the smoothing history window,
        useful for a more heavily damped display (e.g. ghost skeleton)."""
        if not self._history:
            return None
        return np.mean(np.stack(list(self._history)), axis=0)


class PoseGapHandler:
    """Bridges brief pose-detection dropouts (e.g. a single frame of
    motion blur or momentary occlusion) by holding the last known-good
    landmarks, while declaring the pose genuinely "lost" once the gap
    exceeds `max_hold_frames` — at which point callers should stop
    trusting held data (freeze feedback, warn the patient) rather than
    silently analyzing stale landmarks indefinitely.
    """

    def __init__(self, max_hold_frames: int = 10):
        self.max_hold_frames = max_hold_frames
        self._last_valid: Optional[np.ndarray] = None
        self._frames_since_valid = 0

    def reset(self) -> None:
        self._last_valid = None
        self._frames_since_valid = 0

    def update(self, landmarks_xyz: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], bool]:
        """Feed one frame's landmarks (None if detection failed).

        Returns (landmarks_to_use, is_held): `landmarks_to_use` is either
        the fresh detection, the held last-known-good pose (while within
        `max_hold_frames`), or None once the gap has exceeded the hold
        limit. `is_held` is True whenever the returned landmarks are not
        from the current frame's own detection.
        """
        if landmarks_xyz is not None:
            self._last_valid = landmarks_xyz
            self._frames_since_valid = 0
            return landmarks_xyz, False

        self._frames_since_valid += 1
        if self._last_valid is not None and self._frames_since_valid <= self.max_hold_frames:
            logger.debug("Holding last known pose (%d/%d frames since valid detection).", self._frames_since_valid, self.max_hold_frames)
            return self._last_valid, True

        if self._frames_since_valid == self.max_hold_frames + 1:
            logger.warning("Pose lost for more than %d frames; no longer holding stale landmarks.", self.max_hold_frames)
        return None, False

    @property
    def is_lost(self) -> bool:
        return self._frames_since_valid > self.max_hold_frames
