"""
dataset_converter.py
=====================
Converts each supported dataset's raw layout into the unified internal
representation: one `.npz` per exercise sample containing
    landmarks: (T, 33, 3) float array
    label: unified quality class (int)
    metadata: dict (subject_id, exercise_type, source_dataset, ...)
    features: dict (see feature_extraction/feature_pipeline.py), if enabled

All dataset-specific converters below share one core routine —
`datasets.sample_converter.convert_video_to_sample()` — which does the
actual video -> landmarks -> features -> label -> npz work. This file's
job is purely *discovery*: for each dataset, find video files + their
subject/exercise/label metadata on disk and hand them off.

IMPORTANT — regarding the four named public datasets (UI-PRMD, KIMORE,
IntelliRehabDS, Stroke Rehab Mendeley): none are bundled with this
project (see docs/dataset_guide.md), so these converters could not be
validated against real raw files. Each implements a best-effort,
defensively-coded reader for that dataset's *documented* layout
(video-file discovery + flexible annotation-file parsing covering
common column-naming conventions), and logs/records anything it can't
confidently parse via PreprocessingReportBuilder rather than silently
guessing. Once real raw data is available, inspect a handful of files
and adjust the column/path assumptions marked `TODO` below.

The `convert_custom` path has no such caveat — it defines this
project's own layout and is fully exercised by tests/test_dataset_converter.py.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

from datasets.label_processor import LabelProcessor
from datasets.landmark_cache import LandmarkCache
from datasets.preprocessing_report import PreprocessingReportBuilder
from datasets.sample_converter import ConversionResult, convert_video_to_sample
from datasets.video_processor import VideoProcessor
from feature_extraction.feature_pipeline import FeatureExtractionPipeline
from utils.file_io import ensure_dir
from utils.logger import get_logger

logger = get_logger(__name__)

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
ANNOTATION_EXTENSIONS = (".csv", ".json", ".txt")


class DatasetConverter:
    """Discovers raw dataset files and converts them into unified `.npz` samples."""

    def __init__(
        self,
        output_dir: str = "data/processed",
        video_processor: Optional[VideoProcessor] = None,
        landmark_cache: Optional[LandmarkCache] = None,
        label_processor: Optional[LabelProcessor] = None,
        feature_pipeline: Optional[FeatureExtractionPipeline] = None,
        extract_features: bool = True,
    ):
        self.output_dir = ensure_dir(output_dir)
        self.video_processor = video_processor or VideoProcessor()
        self.landmark_cache = landmark_cache or LandmarkCache()
        self.label_processor = label_processor or LabelProcessor()
        self.feature_pipeline = (feature_pipeline or FeatureExtractionPipeline()) if extract_features else None

    # ------------------------------------------------------------------
    # UI-PRMD
    # ------------------------------------------------------------------

    def convert_ui_prmd(self, raw_dir: Path) -> List[ConversionResult]:
        """UI-PRMD's public release is primarily Kinect/Vicon joint-angle
        and position *files*, with correlated video for a subset of
        recordings. This converter processes whatever video files exist
        under `raw_dir`, inferring subject/exercise from the standard
        UI-PRMD naming convention: `m<exercise>_s<subject>_e<episode>.<ext>`
        (e.g. `m01_s02_e03.avi`). TODO: confirm this naming convention
        against the actual downloaded release, and add a joint-angle-file
        reader (bypassing MediaPipe entirely) if you'd rather use
        UI-PRMD's native Kinect/Vicon skeletal data directly.
        """
        return self._convert_by_filename_pattern(
            raw_dir, source_dataset="UI-PRMD",
            parse_filename=self._parse_ui_prmd_filename,
        )

    @staticmethod
    def _parse_ui_prmd_filename(path: Path) -> Optional[Dict]:
        # Expected: m<exercise_id>_s<subject_id>_e<episode_id>.<ext>
        stem = path.stem.lower()
        parts = stem.split("_")
        if len(parts) < 2 or not parts[0].startswith("m"):
            return None
        exercise_id = parts[0]
        subject_id = next((p for p in parts if p.startswith("s")), "unknown_subject")
        return {"exercise_name": f"exercise_{exercise_id}", "subject_id": subject_id}

    # ------------------------------------------------------------------
    # KIMORE
    # ------------------------------------------------------------------

    def convert_kimore(self, raw_dir: Path) -> List[ConversionResult]:
        """KIMORE's release groups recordings by group (e.g. "Expert",
        "NotExpert", pathology groups) then by exercise (Es1..Es5), one
        subject folder per participant, with clinical scores in an
        accompanying label spreadsheet. TODO: confirm this against the
        actual downloaded release, and point `_load_kimore_scores` at the
        real label file name/columns.
        """
        scores = self._load_flexible_annotations(
            raw_dir, id_keys=("subject_id", "subject", "id"), score_keys=("clinical_score", "score", "totalscore")
        )
        return self._convert_by_directory_pattern(
            raw_dir, source_dataset="KIMORE",
            scores_by_subject=scores, clinician_score_range=(0, 100),
        )

    # ------------------------------------------------------------------
    # IntelliRehabDS
    # ------------------------------------------------------------------

    def convert_intellirehabds(self, raw_dir: Path) -> List[ConversionResult]:
        """IntelliRehabDS (Zenodo) provides per-repetition correctness
        labels alongside Kinect recordings, organized by subject and
        exercise. TODO: confirm the label file format against the actual
        Zenodo release; this converter currently derives correctness
        from any parseable 0/1 or True/False label column it can find.
        """
        scores = self._load_flexible_annotations(
            raw_dir, id_keys=("subject_id", "subject", "id"),
            score_keys=("correctness", "is_correct", "label", "quality"),
        )
        return self._convert_by_directory_pattern(
            raw_dir, source_dataset="IntelliRehabDS",
            scores_by_subject=scores, clinician_score_range=(0, 1),
        )

    # ------------------------------------------------------------------
    # Stroke Rehabilitation Exercise Dataset (Mendeley / NIAID host)
    # ------------------------------------------------------------------

    def convert_stroke_rehab_mendeley(self, raw_dir: Path) -> List[ConversionResult]:
        """Stroke Rehabilitation Exercise Dataset, hosted via the NIAID
        data portal (originally Mendeley). TODO: confirm the exact file
        layout/labels against the actual downloaded release — this
        converter uses the same flexible subject/exercise-folder +
        annotation-file discovery as the other adapters.
        """
        scores = self._load_flexible_annotations(
            raw_dir, id_keys=("subject_id", "patient_id", "id"),
            score_keys=("clinician_score", "score", "rating"),
        )
        return self._convert_by_directory_pattern(
            raw_dir, source_dataset="StrokeRehab_Mendeley",
            scores_by_subject=scores, clinician_score_range=(0, 10),
        )

    # ------------------------------------------------------------------
    # Fully generic / custom layout (this project's own recordings)
    # ------------------------------------------------------------------

    def convert_custom(self, raw_dir: Path, labels_csv: Optional[Path] = None) -> List[ConversionResult]:
        """Converts a fully custom, self-recorded dataset with the layout:

            raw_dir/<subject_id>/<exercise_name>/<trial_name>.mp4

        plus an optional `labels_csv` with columns:
            subject_id, exercise_name, trial_name, clinician_score

        This is this project's own documented format (not a public
        dataset), fully exercised by tests/test_dataset_converter.py.
        """
        label_lookup: Dict[str, float] = {}
        if labels_csv and Path(labels_csv).exists():
            with open(labels_csv, "r", newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    key = f"{row['subject_id']}/{row['exercise_name']}/{row['trial_name']}"
                    label_lookup[key] = float(row.get("clinician_score", 0) or 0)

        self.label_processor.clinician_score_range = self.label_processor.clinician_score_range or (0, 10)

        results = []
        for subject_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
            for exercise_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
                for video_path in sorted(exercise_dir.iterdir()):
                    if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
                        continue
                    key = f"{subject_dir.name}/{exercise_dir.name}/{video_path.stem}"
                    sample_id = key.replace("/", "__")
                    results.append(
                        convert_video_to_sample(
                            video_path=str(video_path),
                            sample_id=sample_id,
                            subject_id=subject_dir.name,
                            exercise_name=exercise_dir.name,
                            source_dataset="CUSTOM",
                            output_dir=str(self.output_dir),
                            video_processor=self.video_processor,
                            landmark_cache=self.landmark_cache,
                            label_processor=self.label_processor,
                            feature_pipeline=self.feature_pipeline,
                            raw_clinician_score=label_lookup.get(key),
                        )
                    )
        return results

    # ------------------------------------------------------------------
    # Shared discovery helpers
    # ------------------------------------------------------------------

    def _convert_by_filename_pattern(self, raw_dir: Path, source_dataset: str, parse_filename) -> List[ConversionResult]:
        report_builder = PreprocessingReportBuilder(source_dataset)
        results = []

        if not raw_dir.exists():
            logger.error("Raw dataset directory not found: %s", raw_dir)
            return results

        video_files = [p for p in raw_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS]
        for video_path in sorted(video_files):
            parsed = parse_filename(video_path)
            if parsed is None:
                report_builder.record_rejected_sample(video_path.stem, "Could not parse filename pattern.")
                continue

            results.append(
                convert_video_to_sample(
                    video_path=str(video_path),
                    sample_id=video_path.stem,
                    subject_id=parsed["subject_id"],
                    exercise_name=parsed["exercise_name"],
                    source_dataset=source_dataset,
                    output_dir=str(self.output_dir),
                    video_processor=self.video_processor,
                    landmark_cache=self.landmark_cache,
                    label_processor=self.label_processor,
                    feature_pipeline=self.feature_pipeline,
                    report_builder=report_builder,
                )
            )

        report_builder.save(f"outputs/evaluation_reports/{source_dataset.lower()}_preprocessing_report.json")
        return results

    def _convert_by_directory_pattern(
        self, raw_dir: Path, source_dataset: str, scores_by_subject: Dict[str, float],
        clinician_score_range: tuple,
    ) -> List[ConversionResult]:
        """Generic subject_dir/exercise_dir/*.video discovery, used by the
        KIMORE / IntelliRehabDS / Mendeley adapters — all three
        (per their documentation) organize recordings per-subject,
        per-exercise, which is a reasonable common denominator absent
        access to the actual raw files to confirm exact naming."""
        report_builder = PreprocessingReportBuilder(source_dataset)
        results = []

        if not raw_dir.exists():
            logger.error("Raw dataset directory not found: %s", raw_dir)
            return results

        self.label_processor.clinician_score_range = clinician_score_range

        for subject_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
            subject_id = subject_dir.name
            score = scores_by_subject.get(subject_id)

            video_files = [p for p in subject_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS]
            if not video_files:
                report_builder.record_missing_file(f"No video files under {subject_dir}")
                continue

            for video_path in sorted(video_files):
                exercise_name = video_path.parent.name if video_path.parent != subject_dir else video_path.stem
                results.append(
                    convert_video_to_sample(
                        video_path=str(video_path),
                        sample_id=f"{subject_id}__{video_path.stem}",
                        subject_id=subject_id,
                        exercise_name=exercise_name,
                        source_dataset=source_dataset,
                        output_dir=str(self.output_dir),
                        video_processor=self.video_processor,
                        landmark_cache=self.landmark_cache,
                        label_processor=self.label_processor,
                        feature_pipeline=self.feature_pipeline,
                        raw_clinician_score=score,
                        report_builder=report_builder,
                    )
                )

        report_builder.save(f"outputs/evaluation_reports/{source_dataset.lower()}_preprocessing_report.json")
        return results

    @staticmethod
    def _load_flexible_annotations(raw_dir: Path, id_keys: tuple, score_keys: tuple) -> Dict[str, float]:
        """Scan `raw_dir` for a CSV/JSON annotation file and try common
        column-name variants for subject id and score, rather than
        assuming one exact schema. Returns {subject_id: score}; empty
        dict (with a logged warning) if nothing parseable is found.
        """
        candidates = [p for p in raw_dir.rglob("*") if p.suffix.lower() in ANNOTATION_EXTENSIONS]
        for path in candidates:
            try:
                if path.suffix.lower() == ".csv":
                    scores = DatasetConverter._parse_csv_annotations(path, id_keys, score_keys)
                elif path.suffix.lower() == ".json":
                    scores = DatasetConverter._parse_json_annotations(path, id_keys, score_keys)
                else:
                    continue
                if scores:
                    logger.info("Loaded %d annotation entries from %s", len(scores), path)
                    return scores
            except Exception as exc:  # noqa: BLE001 - best-effort parsing across unknown files
                logger.debug("Could not parse %s as annotations: %s", path, exc)

        logger.warning(
            "No parseable annotation file found under %s (looked for columns %s / %s). "
            "Samples will be converted without a clinician score.",
            raw_dir, id_keys, score_keys,
        )
        return {}

    @staticmethod
    def _parse_csv_annotations(path: Path, id_keys: tuple, score_keys: tuple) -> Dict[str, float]:
        scores = {}
        with open(path, "r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames:
                return {}
            lower_fields = {f.lower(): f for f in reader.fieldnames}
            id_field = next((lower_fields[k] for k in id_keys if k in lower_fields), None)
            score_field = next((lower_fields[k] for k in score_keys if k in lower_fields), None)
            if not id_field or not score_field:
                return {}
            for row in reader:
                try:
                    scores[str(row[id_field])] = float(row[score_field])
                except (ValueError, TypeError):
                    continue
        return scores

    @staticmethod
    def _parse_json_annotations(path: Path, id_keys: tuple, score_keys: tuple) -> Dict[str, float]:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            return {}
        scores = {}
        for entry in data:
            if not isinstance(entry, dict):
                continue
            lower_entry = {k.lower(): v for k, v in entry.items()}
            subject_id = next((lower_entry[k] for k in id_keys if k in lower_entry), None)
            score = next((lower_entry[k] for k in score_keys if k in lower_entry), None)
            if subject_id is not None and score is not None:
                try:
                    scores[str(subject_id)] = float(score)
                except (ValueError, TypeError):
                    continue
        return scores
