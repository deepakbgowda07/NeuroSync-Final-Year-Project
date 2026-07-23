"""
sample_converter.py
=====================
The shared core of dataset conversion: turns one raw video file (plus
its known subject/exercise/label metadata) into one unified sample —
running it through video_processor -> landmark_cache -> feature
extraction -> label_processor -> a single .npz on disk.

Every dataset-specific converter in `datasets/dataset_converter.py`
(UI-PRMD, KIMORE, IntelliRehabDS, Stroke Rehab Mendeley, and a fully
generic "custom" layout) calls into `convert_video_to_sample()` so the
unified-format logic is written exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from datasets.label_processor import LabelProcessor, UnifiedLabel
from datasets.landmark_cache import LandmarkCache
from datasets.preprocessing_report import PreprocessingReportBuilder
from datasets.video_processor import VideoProcessor
from feature_extraction.feature_pipeline import FeatureExtractionPipeline
from utils.file_io import ensure_dir
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ConversionResult:
    success: bool
    sample_id: str
    output_path: Optional[Path] = None
    num_frames: int = 0
    reason: Optional[str] = None


def convert_video_to_sample(
    video_path: str,
    sample_id: str,
    subject_id: str,
    exercise_name: str,
    source_dataset: str,
    output_dir: str,
    video_processor: VideoProcessor,
    landmark_cache: LandmarkCache,
    label_processor: LabelProcessor,
    feature_pipeline: Optional[FeatureExtractionPipeline] = None,
    raw_clinician_score: Optional[float] = None,
    quality_class: Optional[int] = None,
    report_builder: Optional[PreprocessingReportBuilder] = None,
    extra_metadata: Optional[Dict] = None,
) -> ConversionResult:
    """Convert one raw video into a unified `.npz` sample.

    Pipeline: VideoProcessor (frame extraction + integrity checks) ->
    LandmarkCache (MediaPipe extraction, cached) -> optional
    FeatureExtractionPipeline -> LabelProcessor -> save to `output_dir`.
    """
    if report_builder:
        report_builder.record_video_found()

    video_file = Path(video_path)
    if not video_file.exists():
        reason = f"Video file not found: {video_path}"
        if report_builder:
            report_builder.record_missing_file(str(video_path))
        return ConversionResult(success=False, sample_id=sample_id, reason=reason)

    frames, integrity_report = video_processor.extract_frames(str(video_file))

    if not integrity_report.is_valid:
        if report_builder:
            report_builder.record_corrupted_file(str(video_path), integrity_report.reason or "")
        return ConversionResult(success=False, sample_id=sample_id, reason=integrity_report.reason)

    try:
        landmark_result = landmark_cache.extract(sample_id, frames)
    except Exception as exc:  # noqa: BLE001 - surfaced via report, not a crash
        reason = f"Landmark extraction failed: {exc}"
        logger.error(reason)
        if report_builder:
            report_builder.record_rejected_sample(sample_id, reason)
        return ConversionResult(success=False, sample_id=sample_id, reason=reason)

    label: UnifiedLabel = label_processor.build_label(
        exercise_name=exercise_name,
        quality_class=quality_class,
        raw_clinician_score=raw_clinician_score,
    )

    features = None
    if feature_pipeline is not None:
        try:
            features = feature_pipeline.run(landmark_result.landmarks_xyz)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Feature extraction failed for '%s': %s", sample_id, exc)

    metadata = {
        "sample_id": sample_id,
        "subject_id": subject_id,
        "exercise_type": exercise_name,
        "source_dataset": source_dataset,
        "source_video_path": str(video_path),
        "num_frames": int(landmark_result.landmarks_xyz.shape[0]),
        "pose_detection_rate": float(landmark_result.detected_frame_mask.mean()),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    output_path = ensure_dir(output_dir) / f"{sample_id}.npz"
    np.savez_compressed(
        output_path,
        landmarks=landmark_result.landmarks_xyz,
        visibility=landmark_result.visibility,
        label=label.quality_class,
        label_dict=np.array(str(label.to_dict())),
        metadata=metadata,
        features=features if features is not None else {},
    )

    if report_builder:
        report_builder.record_conversion_success(
            subject_id=subject_id,
            exercise_name=exercise_name,
            num_frames=int(landmark_result.landmarks_xyz.shape[0]),
            label=str(label.quality_class),
        )

    logger.info("Converted sample '%s' -> %s", sample_id, output_path)
    return ConversionResult(
        success=True, sample_id=sample_id, output_path=output_path,
        num_frames=int(landmark_result.landmarks_xyz.shape[0]),
    )
