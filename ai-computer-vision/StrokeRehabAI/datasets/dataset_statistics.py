"""
dataset_statistics.py
======================
Summary statistics over a processed dataset directory: sample counts,
sequence-length distribution, class balance, and per-joint coordinate
ranges. Useful both as a sanity check before training and as a
reproducible "Dataset" section for a future paper.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict

import numpy as np

from utils.file_io import list_files
from utils.json_export import write_json
from utils.logger import get_logger

logger = get_logger(__name__)


class DatasetStatistics:
    """Computes and reports summary statistics for a processed dataset."""

    def __init__(self, processed_dir: str = "data/processed"):
        self.processed_dir = Path(processed_dir)

    def compute(self) -> Dict:
        npz_files = list(self.processed_dir.rglob("*.npz"))
        if not npz_files:
            logger.warning("No processed .npz samples found in %s", self.processed_dir)
            return {"num_samples": 0}

        seq_lengths = []
        label_counts: Counter = Counter()

        for f in npz_files:
            with np.load(f, allow_pickle=True) as data:
                seq_lengths.append(len(data["landmarks"]))
                label_counts[str(data["label"])] += 1

        stats = {
            "num_samples": len(npz_files),
            "sequence_length_mean": float(np.mean(seq_lengths)),
            "sequence_length_min": int(np.min(seq_lengths)),
            "sequence_length_max": int(np.max(seq_lengths)),
            "label_distribution": dict(label_counts),
        }
        logger.info("Dataset statistics: %s", stats)
        return stats

    def save_report(self, output_path: str = "outputs/evaluation_reports/dataset_statistics.json") -> None:
        stats = self.compute()
        write_json(stats, output_path)


if __name__ == "__main__":
    DatasetStatistics().save_report()
