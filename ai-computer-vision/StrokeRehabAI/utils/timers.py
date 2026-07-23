"""
timers.py
=========
Lightweight timing utilities for profiling pipeline stages
(camera capture, pose inference, model prediction, rendering, etc.)
"""

from __future__ import annotations

import time
from collections import deque
from contextlib import contextmanager
from typing import Deque, Dict, Iterator, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


class FPSCounter:
    """Rolling FPS counter based on a sliding window of frame timestamps."""

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self._timestamps: Deque[float] = deque(maxlen=window_size)

    def tick(self) -> float:
        """Record a frame timestamp and return the current smoothed FPS."""
        now = time.perf_counter()
        self._timestamps.append(now)
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed


class StageTimer:
    """Accumulates elapsed time per named pipeline stage.

    Usage:
        timer = StageTimer()
        with timer.time("pose_estimation"):
            run_pose_estimation(frame)
        print(timer.summary())
    """

    def __init__(self):
        self._durations_ms: Dict[str, Deque[float]] = {}
        self._window = 100

    @contextmanager
    def time(self, stage_name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            bucket = self._durations_ms.setdefault(stage_name, deque(maxlen=self._window))
            bucket.append(elapsed_ms)

    def average_ms(self, stage_name: str) -> Optional[float]:
        bucket = self._durations_ms.get(stage_name)
        if not bucket:
            return None
        return sum(bucket) / len(bucket)

    def summary(self) -> Dict[str, float]:
        return {name: (sum(vals) / len(vals)) for name, vals in self._durations_ms.items() if vals}
