"""
json_export.py
===============
JSON export/import helpers for structured artifacts: model metadata,
evaluation reports, session summaries, and config snapshots.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

from utils.file_io import ensure_dir
from utils.logger import get_logger

logger = get_logger(__name__)


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar/array types gracefully."""

    def default(self, obj: Any) -> Any:
        try:
            import numpy as np

            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        return super().default(obj)


def write_json(data: Any, output_path: Union[str, Path], indent: int = 2) -> Path:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, cls=NumpyJSONEncoder)
    logger.debug("Wrote JSON to %s", output_path)
    return output_path


def read_json(input_path: Union[str, Path]) -> Any:
    with open(input_path, "r", encoding="utf-8") as fh:
        return json.load(fh)
