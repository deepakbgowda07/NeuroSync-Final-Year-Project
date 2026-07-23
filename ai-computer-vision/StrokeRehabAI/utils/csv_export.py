"""
csv_export.py
=============
CSV export helpers for exercise session logs, joint-angle time series,
and evaluation results — used by both the dashboard and reports module.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Sequence, Union

from utils.file_io import ensure_dir
from utils.logger import get_logger

logger = get_logger(__name__)


def write_dicts_to_csv(rows: Sequence[Dict], output_path: Union[str, Path], fieldnames: List[str] = None) -> Path:
    """Write a list of flat dicts to a CSV file, creating parent dirs as needed."""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    if not rows:
        logger.warning("write_dicts_to_csv called with no rows: %s", output_path)
        fieldnames = fieldnames or []
    else:
        fieldnames = fieldnames or list(rows[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Wrote %d rows to %s", len(rows), output_path)
    return output_path


def read_csv_as_dicts(input_path: Union[str, Path]) -> List[Dict]:
    """Read a CSV file into a list of dicts (one dict per row)."""
    with open(input_path, "r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))
