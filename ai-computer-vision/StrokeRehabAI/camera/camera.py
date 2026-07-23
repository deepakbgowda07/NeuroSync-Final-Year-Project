"""
camera.py
=========
Abstract base interface that all frame sources (live webcam, recorded
video file, future IP camera / RTSP support) implement. This lets the
inference pipeline stay agnostic to where frames actually come from.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Optional, Tuple

import numpy as np


class FrameSource(ABC):
    """Common interface for anything that yields BGR video frames."""

    @abstractmethod
    def open(self) -> None:
        """Open/initialize the underlying capture device or file."""

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a single frame. Returns (success, frame_or_None)."""

    @abstractmethod
    def release(self) -> None:
        """Release any underlying resources (device handles, file handles)."""

    @property
    @abstractmethod
    def fps(self) -> float:
        """Reported or configured frames-per-second of this source."""

    def frames(self) -> Iterator[np.ndarray]:
        """Convenience generator: yields frames until the source is exhausted."""
        self.open()
        try:
            while True:
                ok, frame = self.read()
                if not ok or frame is None:
                    break
                yield frame
        finally:
            self.release()

    def __enter__(self) -> "FrameSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
