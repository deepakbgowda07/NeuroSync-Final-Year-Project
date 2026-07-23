"""
frame_queue.py
================
Thread-safe, bounded frame queue used to decouple camera capture from
frame processing (pose estimation + analysis), so a slow processing
stage never blocks the capture thread and — critically for real-time
feedback — the consumer always sees the *newest* frame rather than
working through a backlog of stale ones.

Two drop policies are supported when the queue is full:
    "oldest" (default): discard the oldest queued frame, keep filling
                          with new ones — minimizes end-to-end latency.
    "newest": discard the incoming frame, keep what's already queued —
               useful if every frame must eventually be processed
               (e.g. session recording) at the cost of added latency.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TimestampedFrame:
    frame: np.ndarray
    timestamp: float
    frame_index: int


class FrameQueue:
    """A small, bounded, thread-safe queue of the most recent camera frames."""

    def __init__(self, max_size: int = 2, drop_policy: str = "oldest"):
        if drop_policy not in ("oldest", "newest"):
            raise ValueError(f"Unknown drop_policy: {drop_policy}")
        self.max_size = max_size
        self.drop_policy = drop_policy
        self._queue: "queue.Queue[TimestampedFrame]" = queue.Queue(maxsize=max_size)
        self._lock = threading.Lock()
        self._frame_counter = 0
        self._dropped_frames = 0

    def put(self, frame: np.ndarray) -> None:
        """Enqueue a new frame, applying the configured drop policy if full."""
        with self._lock:
            timestamped = TimestampedFrame(frame=frame, timestamp=time.perf_counter(), frame_index=self._frame_counter)
            self._frame_counter += 1

            if self._queue.full():
                if self.drop_policy == "oldest":
                    try:
                        self._queue.get_nowait()
                        self._dropped_frames += 1
                    except queue.Empty:
                        pass
                    self._queue.put_nowait(timestamped)
                else:  # "newest" — drop the incoming frame instead
                    self._dropped_frames += 1
                    return
            else:
                self._queue.put_nowait(timestamped)

    def get(self, timeout: Optional[float] = 1.0) -> Optional[TimestampedFrame]:
        """Dequeue the next frame (blocking up to `timeout` seconds).
        Returns None on timeout rather than raising, so callers can loop
        cleanly without a try/except on every iteration."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_latest(self) -> Optional[TimestampedFrame]:
        """Drain the queue and return only the single newest frame —
        useful when a consumer fell behind and wants to skip straight
        to the most current frame rather than processing a backlog."""
        latest = None
        while True:
            try:
                latest = self._queue.get_nowait()
            except queue.Empty:
                break
        return latest

    def qsize(self) -> int:
        return self._queue.qsize()

    @property
    def dropped_frames(self) -> int:
        return self._dropped_frames

    def clear(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
