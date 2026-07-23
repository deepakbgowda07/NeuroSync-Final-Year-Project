"""
dataset_validator.py
=====================
Deeper structural validation once a dataset is present: verifies file
formats parse correctly, landmark/label dimensionality matches
expectations, and reports issues before a training run wastes time on
malformed data.

TODO (next development phase): implement per-dataset parsers.
UI-PRMD, KIMORE, and IntelliRehabDS each ship skeletal data in
different raw formats (.txt / .csv / .mat variants); the parsing
adapters below are stubs pending real data access.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationIssue:
    severity: str  # "error" | "warning"
    message: str


@dataclass
class ValidationReport:
    dataset_name: str
    total_files_checked: int = 0
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)


class DatasetValidator:
    """Runs format/consistency checks against a dataset directory."""

    def __init__(self, dataset_name: str, expected_landmark_dim: int = 99):
        self.dataset_name = dataset_name
        self.expected_landmark_dim = expected_landmark_dim

    def validate(self, dataset_dir: Path) -> ValidationReport:
        report = ValidationReport(dataset_name=self.dataset_name)

        if not dataset_dir.exists():
            report.issues.append(ValidationIssue("error", f"Directory does not exist: {dataset_dir}"))
            return report

        # TODO: dispatch to a dataset-specific parser/validator here based
        # on self.dataset_name, once the raw formats are available for
        # inspection (UI-PRMD .txt, KIMORE .csv, IntelliRehabDS .mat, etc.)
        candidate_files = list(dataset_dir.rglob("*.*"))
        report.total_files_checked = len(candidate_files)

        if report.total_files_checked == 0:
            report.issues.append(ValidationIssue("warning", "No files found to validate."))

        logger.info(
            "Validated %s: %d files checked, %d issues found.",
            self.dataset_name,
            report.total_files_checked,
            len(report.issues),
        )
        return report
