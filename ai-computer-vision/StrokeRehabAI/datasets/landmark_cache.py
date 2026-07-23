"""
landmark_cache.py
===================
Runs MediaPipe BlazePose (via mediapipe_pipeline.PoseEstimator) over an
extracted frame sequence, and caches the resulting landmarks to disk so
repeated dataset-conversion or training-data-prep runs don't re-run
pose estimation on unchanged videos.

Cache layout (under configs/datasets.yaml -> landmark_extraction.cache_dir):

    <cache_dir>/<sample_id>.npz    -- primary cache (landmarks, visibility, meta)
    <cache_dir>/<sample_id>.csv    -- optional flat export (one row per frame*joint)
    <cache_dir>/<sample_id>.json   -- optional export (nested per-frame dict)
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from mediapipe_pipeline.landmark_extractor import LANDMARK_NAMES
from mediapipe_pipeline.pose_estimator import PoseEstimator
from utils.file_io import ensure_dir
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LandmarkExtractionResult:
    sample_id: str
    landmarks_xyz: np.ndarray        # (T, 33, 3)
    visibility: np.ndarray           # (T, 33)
    detected_frame_mask: np.ndarray  # (T,) bool -- False where pose detection failed
    from_cache: bool


class LandmarkCache:
    """Extracts (or loads cached) MediaPipe landmark sequences for dataset samples."""

    def __init__(
        self,
        cache_dir: str = "data/unified/landmarks",
        cache_format: str = "npz",
        overwrite_cache: bool = False,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.cache_dir = ensure_dir(cache_dir)
        self.cache_format = cache_format
        self.overwrite_cache = overwrite_cache
        self._estimator = PoseEstimator(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def _cache_path(self, sample_id: str, extension: str = "npz") -> Path:
        return self.cache_dir / f"{sample_id}.{extension}"

    def has_cache(self, sample_id: str) -> bool:
        return self._cache_path(sample_id, "npz").exists()

    def extract(self, sample_id: str, frames: List[np.ndarray]) -> LandmarkExtractionResult:
        """Extract landmarks for `frames`, or load them from cache if
        present and `overwrite_cache` is False. Frames with no detected
        pose are held at the previous frame's position (gap-filled) so
        downstream fixed-length windows never contain NaNs, and are
        flagged in `detected_frame_mask`.
        """
        cache_path = self._cache_path(sample_id, "npz")
        if cache_path.exists() and not self.overwrite_cache:
            return self._load_cached(sample_id, cache_path)

        if not frames:
            raise ValueError(f"No frames provided for sample '{sample_id}'.")

        num_frames = len(frames)
        landmarks = np.zeros((num_frames, PoseEstimator.NUM_LANDMARKS, 3), dtype=np.float64)
        visibility = np.zeros((num_frames, PoseEstimator.NUM_LANDMARKS), dtype=np.float64)
        detected_mask = np.zeros(num_frames, dtype=bool)

        with self._estimator:
            last_valid = None
            for i, frame in enumerate(frames):
                result = self._estimator.process(frame)
                if result.detected:
                    landmarks[i] = result.landmarks_xyz
                    visibility[i] = result.landmarks_visibility
                    detected_mask[i] = True
                    last_valid = (result.landmarks_xyz, result.landmarks_visibility)
                elif last_valid is not None:
                    landmarks[i], visibility[i] = last_valid
                    logger.debug("Sample %s frame %d: no pose detected, holding last frame.", sample_id, i)

        detection_rate = detected_mask.mean() if num_frames else 0.0
        logger.info(
            "Extracted landmarks for '%s': %d frames, %.1f%% detection rate.",
            sample_id, num_frames, detection_rate * 100,
        )

        result = LandmarkExtractionResult(
            sample_id=sample_id,
            landmarks_xyz=landmarks,
            visibility=visibility,
            detected_frame_mask=detected_mask,
            from_cache=False,
        )
        self._save_cache(result)
        return result

    def _save_cache(self, result: LandmarkExtractionResult) -> None:
        np.savez_compressed(
            self._cache_path(result.sample_id, "npz"),
            landmarks=result.landmarks_xyz,
            visibility=result.visibility,
            detected_frame_mask=result.detected_frame_mask,
        )

        if self.cache_format in ("csv", "all"):
            self._export_csv(result)
        if self.cache_format in ("json", "all"):
            self._export_json(result)

    def _load_cached(self, sample_id: str, cache_path: Path) -> LandmarkExtractionResult:
        with np.load(cache_path) as data:
            result = LandmarkExtractionResult(
                sample_id=sample_id,
                landmarks_xyz=data["landmarks"],
                visibility=data["visibility"],
                detected_frame_mask=data["detected_frame_mask"],
                from_cache=True,
            )
        logger.debug("Loaded cached landmarks for '%s' from %s", sample_id, cache_path)
        return result

    def _export_csv(self, result: LandmarkExtractionResult) -> None:
        csv_path = self._cache_path(result.sample_id, "csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["frame_index", "joint_name", "x", "y", "z", "visibility", "detected"])
            for t in range(result.landmarks_xyz.shape[0]):
                for j, name in enumerate(LANDMARK_NAMES):
                    x, y, z = result.landmarks_xyz[t, j]
                    writer.writerow([t, name, x, y, z, result.visibility[t, j], bool(result.detected_frame_mask[t])])

    def _export_json(self, result: LandmarkExtractionResult) -> None:
        json_path = self._cache_path(result.sample_id, "json")
        payload = {
            "sample_id": result.sample_id,
            "num_frames": int(result.landmarks_xyz.shape[0]),
            "detection_rate": float(result.detected_frame_mask.mean()),
            "frames": [
                {
                    "frame_index": t,
                    "detected": bool(result.detected_frame_mask[t]),
                    "landmarks": {
                        name: {
                            "x": float(result.landmarks_xyz[t, j, 0]),
                            "y": float(result.landmarks_xyz[t, j, 1]),
                            "z": float(result.landmarks_xyz[t, j, 2]),
                            "visibility": float(result.visibility[t, j]),
                        }
                        for j, name in enumerate(LANDMARK_NAMES)
                    },
                }
                for t in range(result.landmarks_xyz.shape[0])
            ],
        }
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
