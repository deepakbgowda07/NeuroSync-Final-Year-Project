"""
webcam.py
=========
Live webcam frame source, built on OpenCV VideoCapture. Designed for
the integrated laptop webcam described in the project's target
hardware profile, but works with any OpenCV-compatible camera index.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from camera.camera import FrameSource
from utils.logger import get_logger

logger = get_logger(__name__)

# Maps friendly backend names from configs/camera.yaml to OpenCV constants.
_BACKEND_MAP = {
    "ANY": 0,        # cv2.CAP_ANY
    "DSHOW": 700,    # cv2.CAP_DSHOW (Windows DirectShow, low latency)
    "MSMF": 1400,    # cv2.CAP_MSMF (Windows Media Foundation)
}


class Webcam(FrameSource):
    """Live camera capture with configurable resolution, FPS, and backend."""

    def __init__(
        self,
        source: int = 0,
        width: int = 1280,
        height: int = 720,
        fps_target: int = 30,
        backend: str = "DSHOW",
        flip_horizontal: bool = True,
        buffer_size: int = 1,
    ):
        self.source = source
        self.width = width
        self.height = height
        self.fps_target = fps_target
        self.backend = backend
        self.flip_horizontal = flip_horizontal
        self.buffer_size = buffer_size
        self._cap = None

    def open(self) -> None:
        import cv2

        backend_flag = _BACKEND_MAP.get(self.backend, 0)
        self._cap = cv2.VideoCapture(self.source, backend_flag)

        if not self._cap.isOpened():
            # Retry with default backend as a fallback (e.g. on non-Windows hosts).
            logger.warning("Backend '%s' failed to open, retrying with CAP_ANY.", self.backend)
            self._cap = cv2.VideoCapture(self.source)

        if not self._cap.isOpened():
            raise RuntimeError(f"Unable to open webcam at source={self.source}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps_target)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

        actual_w = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        logger.info("Webcam opened: requested %dx%d, actual %dx%d", self.width, self.height, actual_w, actual_h)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None:
            raise RuntimeError("Webcam.read() called before open().")
        ok, frame = self._cap.read()
        if not ok:
            return False, None
        if self.flip_horizontal:
            import cv2

            frame = cv2.flip(frame, 1)
        return True, frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            logger.info("Webcam released.")
            self._cap = None

    @property
    def fps(self) -> float:
        if self._cap is None:
            return float(self.fps_target)
        reported = self._cap.get(5)  # cv2.CAP_PROP_FPS == 5
        return reported if reported > 0 else float(self.fps_target)
