"""
fps_controller.py
===================
Measures actual achieved FPS and provides automatic frame-rate
adjustment: if the processing pipeline can't keep up with the camera's
native rate, the controller recommends a lower target FPS (down to a
configured floor) so the system degrades gracefully rather than
accumulating latency; if headroom reappears, it recommends stepping
back up.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque

from utils.logger import get_logger

logger = get_logger(__name__)


class FPSController:
    """Tracks recent frame timings and adaptively adjusts the target FPS.

    Usage:
        controller = FPSController(target_fps=30, min_fps=15)
        while True:
            controller.frame_start()
            ... process frame ...
            controller.frame_end()
            current_fps = controller.measured_fps
            if controller.should_skip_frame():
                continue  # let the capture thread advance without processing
    """

    def __init__(self, target_fps: int = 30, min_fps: int = 15, window_size: int = 30, adjust_every_n_frames: int = 30):
        self.target_fps = target_fps
        self.min_fps = min_fps
        self.current_fps_cap = target_fps
        self.window_size = window_size
        self.adjust_every_n_frames = adjust_every_n_frames

        self._frame_durations: Deque[float] = deque(maxlen=window_size)
        self._frame_start_time: float = 0.0
        self._frames_since_adjustment = 0

    def frame_start(self) -> None:
        self._frame_start_time = time.perf_counter()

    def frame_end(self) -> None:
        duration = time.perf_counter() - self._frame_start_time
        self._frame_durations.append(duration)
        self._frames_since_adjustment += 1

        if self._frames_since_adjustment >= self.adjust_every_n_frames:
            self._adjust_target_fps()
            self._frames_since_adjustment = 0

    @property
    def measured_fps(self) -> float:
        if not self._frame_durations:
            return 0.0
        mean_duration = sum(self._frame_durations) / len(self._frame_durations)
        return 1.0 / mean_duration if mean_duration > 0 else 0.0

    def should_skip_frame(self) -> bool:
        """Recommend skipping the current frame if the measured FPS has
        fallen meaningfully behind the current adaptive cap — lets a
        capture thread outrun a temporarily slow processing stage
        without the whole system falling further behind."""
        measured = self.measured_fps
        return measured > 0 and measured < (self.current_fps_cap * 0.6)

    def _adjust_target_fps(self) -> None:
        measured = self.measured_fps
        if measured <= 0:
            return

        if measured < self.current_fps_cap * 0.8 and self.current_fps_cap > self.min_fps:
            new_cap = max(self.min_fps, self.current_fps_cap - 5)
            if new_cap != self.current_fps_cap:
                logger.warning(
                    "FPS controller stepping target down: %d -> %d fps (measured %.1f fps).",
                    self.current_fps_cap, new_cap, measured,
                )
            self.current_fps_cap = new_cap
        elif measured > self.target_fps * 0.95 and self.current_fps_cap < self.target_fps:
            new_cap = min(self.target_fps, self.current_fps_cap + 5)
            if new_cap != self.current_fps_cap:
                logger.info(
                    "FPS controller stepping target back up: %d -> %d fps (measured %.1f fps).",
                    self.current_fps_cap, new_cap, measured,
                )
            self.current_fps_cap = new_cap

    def sleep_to_maintain_fps(self) -> None:
        """Optional pacing helper: sleeps just enough to cap the loop at
        `current_fps_cap`, for sources (e.g. video files) that would
        otherwise run faster than real time."""
        if self.current_fps_cap <= 0:
            return
        target_duration = 1.0 / self.current_fps_cap
        elapsed = time.perf_counter() - self._frame_start_time
        remaining = target_duration - elapsed
        if remaining > 0:
            time.sleep(remaining)
