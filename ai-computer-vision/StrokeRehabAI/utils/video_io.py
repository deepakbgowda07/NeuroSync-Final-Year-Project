"""
video_io.py
===========
Video read/write helpers built on OpenCV, used by both the offline
dataset pipeline (extracting frames from recorded videos) and the
inference pipeline (optionally recording annotated sessions).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


class VideoWriter:
    """Thin wrapper around cv2.VideoWriter with sane defaults for
    recording annotated rehab sessions."""

    def __init__(self, output_path: str, fps: float = 30.0, frame_size: Optional[Tuple[int, int]] = None,
                 fourcc: str = "mp4v"):
        self.output_path = str(output_path)
        self.fps = fps
        self.frame_size = frame_size
        self.fourcc = fourcc
        self._writer = None

    def _lazy_init(self, frame_shape: Tuple[int, int]) -> None:
        import cv2

        if self.frame_size is None:
            self.frame_size = (frame_shape[1], frame_shape[0])  # (width, height)
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc_code = cv2.VideoWriter_fourcc(*self.fourcc)
        self._writer = cv2.VideoWriter(self.output_path, fourcc_code, self.fps, self.frame_size)
        logger.info("VideoWriter opened: %s (%dx%d @ %.1f fps)", self.output_path, *self.frame_size, self.fps)

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            self._lazy_init(frame.shape[:2])
        self._writer.write(frame)

    def release(self) -> None:
        if self._writer is not None:
            self._writer.release()
            logger.info("VideoWriter released: %s", self.output_path)
            self._writer = None

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


def iter_video_frames(video_path: str) -> Iterator[np.ndarray]:
    """Yield frames (BGR numpy arrays) from a video file, one at a time.

    Used by the offline dataset pipeline when extracting pose sequences
    from recorded exercise videos rather than a live webcam feed.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()
