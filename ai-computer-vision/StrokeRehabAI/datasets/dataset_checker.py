"""
dataset_checker.py
===================
Checks whether a configured dataset's expected local directory exists
and contains a plausible amount of data, without doing deep format
validation (that's dataset_validator.py's job).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

from configs.config_loader import load_config
from utils.file_io import resolve_path
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetCheckResult:
    dataset_name: str
    local_path: Path
    exists: bool
    file_count: int
    is_probably_ready: bool


def check_dataset(dataset_name: str) -> DatasetCheckResult:
    cfg = load_config()
    entry = cfg.datasets.sources.get(dataset_name)
    if entry is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    local_path = resolve_path(entry["local_path"])
    exists = local_path.exists()
    file_count = sum(1 for _ in local_path.rglob("*")) if exists else 0

    result = DatasetCheckResult(
        dataset_name=dataset_name,
        local_path=local_path,
        exists=exists,
        file_count=file_count,
        is_probably_ready=exists and file_count > 0,
    )

    status = "READY" if result.is_probably_ready else "NOT READY"
    logger.info("[%s] %s -- %d files found at %s", status, dataset_name, file_count, local_path)
    return result


class DatasetChecker:
    """Convenience wrapper to check every configured dataset at once."""

    def check_all(self) -> List[DatasetCheckResult]:
        cfg = load_config()
        return [check_dataset(name) for name in cfg.datasets.sources.keys()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, help="Check a single dataset; omit to check all.")
    args = parser.parse_args()

    if args.dataset:
        check_dataset(args.dataset)
    else:
        DatasetChecker().check_all()
