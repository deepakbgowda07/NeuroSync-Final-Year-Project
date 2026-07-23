"""
preprocessing_report.py
=========================
Generates a human- and machine-readable preprocessing report for a
dataset conversion run: how many videos/subjects/exercises/frames were
found, the label distribution, and any missing files or annotations
encountered along the way.

Distinct from `datasets/dataset_statistics.py` (which summarizes the
final *converted* .npz corpus) — this module reports on the *raw ->
converted* process itself, including everything that failed or was
skipped, which is what a methods-section "Dataset" write-up (and basic
debugging) actually needs.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List

from utils.json_export import write_json
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PreprocessingReport:
    dataset_name: str
    num_videos_found: int = 0
    num_videos_converted: int = 0
    num_subjects: int = 0
    num_exercises: int = 0
    total_frames: int = 0
    label_distribution: Dict[str, int] = field(default_factory=dict)
    missing_files: List[str] = field(default_factory=list)
    missing_annotations: List[str] = field(default_factory=list)
    corrupted_files: List[str] = field(default_factory=list)
    duplicate_files: List[str] = field(default_factory=list)
    rejected_samples: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "dataset_name": self.dataset_name,
            "num_videos_found": self.num_videos_found,
            "num_videos_converted": self.num_videos_converted,
            "conversion_success_rate": (
                self.num_videos_converted / self.num_videos_found if self.num_videos_found else 0.0
            ),
            "num_subjects": self.num_subjects,
            "num_exercises": self.num_exercises,
            "total_frames": self.total_frames,
            "label_distribution": self.label_distribution,
            "missing_files": self.missing_files,
            "missing_annotations": self.missing_annotations,
            "corrupted_files": self.corrupted_files,
            "duplicate_files": self.duplicate_files,
            "rejected_samples": self.rejected_samples,
        }

    def print_summary(self) -> None:
        print(f"\n=== Preprocessing Report: {self.dataset_name} ===")
        print(f"Videos found         : {self.num_videos_found}")
        print(f"Videos converted      : {self.num_videos_converted}")
        print(f"Subjects              : {self.num_subjects}")
        print(f"Exercises             : {self.num_exercises}")
        print(f"Total frames          : {self.total_frames}")
        print(f"Missing files         : {len(self.missing_files)}")
        print(f"Missing annotations   : {len(self.missing_annotations)}")
        print(f"Corrupted files       : {len(self.corrupted_files)}")
        print(f"Duplicate files       : {len(self.duplicate_files)}")
        print(f"Label distribution    : {self.label_distribution}\n")


class PreprocessingReportBuilder:
    """Accumulates events during a dataset conversion run and produces a
    final PreprocessingReport."""

    def __init__(self, dataset_name: str):
        self.report = PreprocessingReport(dataset_name=dataset_name)
        self._subjects: set = set()
        self._exercises: set = set()
        self._label_counter: Counter = Counter()

    def record_video_found(self) -> None:
        self.report.num_videos_found += 1

    def record_conversion_success(self, subject_id: str, exercise_name: str, num_frames: int, label: str) -> None:
        self.report.num_videos_converted += 1
        self.report.total_frames += num_frames
        self._subjects.add(subject_id)
        self._exercises.add(exercise_name)
        self._label_counter[str(label)] += 1

    def record_missing_file(self, path: str) -> None:
        self.report.missing_files.append(path)
        logger.warning("Missing file: %s", path)

    def record_missing_annotation(self, path: str) -> None:
        self.report.missing_annotations.append(path)
        logger.warning("Missing annotation: %s", path)

    def record_corrupted_file(self, path: str, reason: str = "") -> None:
        self.report.corrupted_files.append(path)
        logger.warning("Corrupted file: %s (%s)", path, reason)

    def record_duplicate_file(self, path: str) -> None:
        self.report.duplicate_files.append(path)
        logger.info("Duplicate file skipped: %s", path)

    def record_rejected_sample(self, sample_id: str, reason: str) -> None:
        self.report.rejected_samples.append({"sample_id": sample_id, "reason": reason})
        logger.warning("Rejected sample '%s': %s", sample_id, reason)

    def finalize(self) -> PreprocessingReport:
        self.report.num_subjects = len(self._subjects)
        self.report.num_exercises = len(self._exercises)
        self.report.label_distribution = dict(self._label_counter)
        return self.report

    def save(self, output_path: str = "outputs/evaluation_reports/preprocessing_report.json") -> PreprocessingReport:
        report = self.finalize()
        write_json(report.to_dict(), output_path)
        report.print_summary()
        logger.info("Preprocessing report saved to %s", output_path)
        return report
