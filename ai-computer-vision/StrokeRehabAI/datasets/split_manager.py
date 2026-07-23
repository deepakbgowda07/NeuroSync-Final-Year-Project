"""
split_manager.py
==================
Deterministic train/validation/test splitting over dataset samples,
supporting either sample-level or subject-level splitting.

Subject-level splitting (the default, `split_by: "subject"` in
configs/datasets.yaml) is important for clinical data: it guarantees no
patient appears in more than one split, which sample-level random
splitting cannot — that leakage inflates validation metrics because the
model partially memorizes a specific patient's movement signature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from utils.json_export import write_json
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SplitResult:
    train_ids: List[str] = field(default_factory=list)
    val_ids: List[str] = field(default_factory=list)
    test_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, List[str]]:
        return {"train": self.train_ids, "val": self.val_ids, "test": self.test_ids}


class SplitManager:
    """Builds reproducible train/val/test splits keyed by sample_id or subject_id."""

    def __init__(self, train_split: float = 0.7, val_split: float = 0.15, test_split: float = 0.15, seed: int = 42):
        total = train_split + val_split + test_split
        if not np.isclose(total, 1.0, atol=1e-3):
            raise ValueError(f"Splits must sum to 1.0, got {total:.3f}")
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed

    def split_by_sample(self, sample_ids: Sequence[str]) -> SplitResult:
        """Randomly partitions individual samples. Use only when samples
        are already guaranteed independent (e.g. one video per subject)."""
        ids = list(sample_ids)
        rng = np.random.default_rng(self.seed)
        rng.shuffle(ids)

        n_train = int(len(ids) * self.train_split)
        n_val = int(len(ids) * self.val_split)

        result = SplitResult(
            train_ids=ids[:n_train],
            val_ids=ids[n_train:n_train + n_val],
            test_ids=ids[n_train + n_val:],
        )
        self._log_summary(result)
        return result

    def split_by_subject(self, sample_id_to_subject: Dict[str, str]) -> SplitResult:
        """Partition by subject_id first, then assign every sample
        belonging to a subject to that subject's split — preventing any
        patient from appearing in more than one split."""
        subjects = sorted(set(sample_id_to_subject.values()))
        rng = np.random.default_rng(self.seed)
        rng.shuffle(subjects)

        n_train = max(1, int(len(subjects) * self.train_split)) if subjects else 0
        n_val = max(0, int(len(subjects) * self.val_split))

        train_subjects = set(subjects[:n_train])
        val_subjects = set(subjects[n_train:n_train + n_val])
        test_subjects = set(subjects[n_train + n_val:])

        result = SplitResult(
            train_ids=[sid for sid, subj in sample_id_to_subject.items() if subj in train_subjects],
            val_ids=[sid for sid, subj in sample_id_to_subject.items() if subj in val_subjects],
            test_ids=[sid for sid, subj in sample_id_to_subject.items() if subj in test_subjects],
        )
        logger.info(
            "Subject-level split: %d train subjects, %d val subjects, %d test subjects.",
            len(train_subjects), len(val_subjects), len(test_subjects),
        )
        self._log_summary(result)
        return result

    def _log_summary(self, result: SplitResult) -> None:
        logger.info(
            "Split summary: train=%d, val=%d, test=%d samples.",
            len(result.train_ids), len(result.val_ids), len(result.test_ids),
        )

    def save(self, result: SplitResult, output_path: str) -> None:
        write_json(result.to_dict(), output_path)
        logger.info("Saved split assignment to %s", output_path)
