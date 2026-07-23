"""
video_loader.py
================
Frame source for pre-recorded video files (used both for offline
dataset processing and for demoing the inference pipeline without a
live webcam).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from camera.camera import FrameSource
from utils.logger import get_logger

logger = get_logger(__name__)


class VideoLoader(FrameSource):
    """Reads frames sequentially from a video file on disk."""

    def __init__(self, video_path: str, loop: bool = False):
        self.video_path = str(video_path)
        self.loop = loop
        self._cap = None
        self._reported_fps = 30.0

    def open(self) -> None:
        import cv2

        if not Path(self.video_path).exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Unable to open video file: {self.video_path}")

        self._reported_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info("Opened video: %s (%d frames @ %.1f fps)", self.video_path, frame_count, self._reported_fps)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None:
            raise RuntimeError("VideoLoader.read() called before open().")

        ok, frame = self._cap.read()
        if not ok and self.loop:
            import cv2

            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()

        return (ok, frame) if ok else (False, None)

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            logger.info("VideoLoader released: %s", self.video_path)
            self._cap = None

    @property
    def fps(self) -> float:
        return self._reported_fps
