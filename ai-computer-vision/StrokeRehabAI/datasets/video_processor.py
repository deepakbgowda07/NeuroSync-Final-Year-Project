"""
video_processor.py
====================
Video-level preprocessing shared by every dataset converter: frame
extraction, frame-rate sampling, resolution normalization, and
integrity checks (missing frames, corrupted files, duplicate frames).

This sits below `datasets/dataset_converter.py` in the pipeline: the
converter decides *which* videos/samples to process for a given
dataset's layout; this module decides *how* to turn one video file
into a clean, uniform sequence of frames.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FrameIntegrityReport:
    """Summarizes integrity issues found while reading a single video."""

    total_frames_read: int = 0
    corrupted_frame_indices: List[int] = field(default_factory=list)
    duplicate_frame_indices: List[int] = field(default_factory=list)
    is_valid: bool = True
    reason: Optional[str] = None


class VideoProcessor:
    """Reads, samples, and normalizes video frames for the dataset pipeline."""

    def __init__(
        self,
        target_resolution: Tuple[int, int] = (640, 480),
        target_fps: float = 30.0,
        frame_sample_rate: int = 1,
        min_valid_frames: int = 15,
        max_frame_gap_ratio: float = 0.2,
        duplicate_hash_size: int = 8,
    ):
        self.target_width, self.target_height = target_resolution
        self.target_fps = target_fps
        self.frame_sample_rate = max(1, frame_sample_rate)
        self.min_valid_frames = min_valid_frames
        self.max_frame_gap_ratio = max_frame_gap_ratio
        self.duplicate_hash_size = duplicate_hash_size

    # ------------------------------------------------------------------
    # Frame extraction
    # ------------------------------------------------------------------

    def extract_frames(self, video_path: str) -> Tuple[List[np.ndarray], FrameIntegrityReport]:
        """Read a video file, applying frame sampling + resolution/FPS
        normalization, and run integrity checks along the way.

        Returns (frames, integrity_report). `frames` is an empty list if
        the video could not be opened at all.
        """
        import cv2

        path = Path(video_path)
        report = FrameIntegrityReport()

        if not path.exists():
            report.is_valid = False
            report.reason = f"File does not exist: {path}"
            logger.error(report.reason)
            return [], report

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            report.is_valid = False
            report.reason = f"Could not open video (possibly corrupted): {path}"
            logger.error(report.reason)
            return [], report

        source_fps = cap.get(cv2.CAP_PROP_FPS) or self.target_fps
        fps_stride = max(1, round(source_fps / self.target_fps)) if self.target_fps else 1
        combined_stride = fps_stride * self.frame_sample_rate

        frames: List[np.ndarray] = []
        frame_hashes: List[str] = []
        raw_index = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            report.total_frames_read += 1

            if raw_index % combined_stride == 0:
                if frame is None or frame.size == 0:
                    report.corrupted_frame_indices.append(raw_index)
                else:
                    normalized = self._normalize_frame(frame)
                    frame_hash = self._perceptual_hash(normalized)
                    if frame_hash in frame_hashes:
                        report.duplicate_frame_indices.append(raw_index)
                    frame_hashes.append(frame_hash)
                    frames.append(normalized)

            raw_index += 1

        cap.release()

        self._finalize_report(report, len(frames))
        return frames, report

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _normalize_frame(self, frame: np.ndarray):
        """Resize to the configured target resolution (letterboxed to
        preserve aspect ratio, matching camera/camera_utils.resize_frame)."""
        from camera.camera_utils import resize_frame

        return resize_frame(frame, (self.target_width, self.target_height))

    def _perceptual_hash(self, frame: np.ndarray) -> str:
        """Cheap perceptual hash (downsample + average-threshold) used for
        near-duplicate frame detection — robust to minor compression noise,
        unlike a raw byte hash."""
        import cv2

        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (self.duplicate_hash_size, self.duplicate_hash_size))
        avg = small.mean()
        bits = (small > avg).flatten()
        return hashlib.md5(np.packbits(bits).tobytes()).hexdigest()

    # ------------------------------------------------------------------
    # Integrity
    # ------------------------------------------------------------------

    def _finalize_report(self, report: FrameIntegrityReport, num_extracted_frames: int) -> None:
        if num_extracted_frames < self.min_valid_frames:
            report.is_valid = False
            report.reason = (
                f"Too few valid frames extracted ({num_extracted_frames} < "
                f"{self.min_valid_frames})."
            )
            return

        gap_count = len(report.corrupted_frame_indices)
        gap_ratio = gap_count / max(1, report.total_frames_read)
        if gap_ratio > self.max_frame_gap_ratio:
            report.is_valid = False
            report.reason = (
                f"Too many corrupted/missing frames ({gap_ratio:.1%} > "
                f"{self.max_frame_gap_ratio:.1%})."
            )
            return

        report.is_valid = True

    def detect_missing_frames(self, expected_count: int, actual_count: int) -> int:
        """Simple helper: how many frames are missing relative to an
        expected count (e.g. from video metadata / duration * fps)."""
        return max(0, expected_count - actual_count)
