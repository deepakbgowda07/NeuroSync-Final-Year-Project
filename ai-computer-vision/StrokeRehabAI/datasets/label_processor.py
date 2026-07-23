"""
label_processor.py
====================
Unifies the different label types that appear across supported
datasets into a consistent internal representation:

    - Exercise label       (categorical: which exercise is being performed)
    - Quality label         (categorical: correctness/severity class — the
                              default target for models/model_factory.py's
                              `output_head: "classification"`)
    - Regression target     (continuous: e.g. a normalized 0-1 quality score,
                              for `output_head: "regression"`)
    - Clinician score        (raw clinical rating, dataset-specific scale —
                              e.g. KIMORE's 0-100 clinical score)
    - Binary correct/incorrect (derived from quality/clinician score via a
                              configurable threshold)

Each dataset converter maps its native labels into this scheme when
producing unified .npz samples (see datasets/dataset_converter.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class UnifiedLabel:
    exercise_name: str
    quality_class: int                    # categorical correctness/severity class
    regression_target: Optional[float]    # normalized [0, 1] continuous quality score
    clinician_score: Optional[float]      # raw clinical score, dataset-native scale
    binary_correct: Optional[int]         # 1 = correct, 0 = incorrect, None = unknown

    def to_dict(self) -> Dict:
        return {
            "exercise_name": self.exercise_name,
            "quality_class": self.quality_class,
            "regression_target": self.regression_target,
            "clinician_score": self.clinician_score,
            "binary_correct": self.binary_correct,
        }


class LabelEncoder:
    """Maps exercise name strings to stable integer indices, fitted once
    per dataset (or loaded from a saved mapping to keep train/val/test
    consistent)."""

    def __init__(self):
        self._name_to_index: Dict[str, int] = {}

    def fit(self, exercise_names: List[str]) -> "LabelEncoder":
        for name in sorted(set(exercise_names)):
            self._name_to_index.setdefault(name, len(self._name_to_index))
        return self

    def encode(self, exercise_name: str) -> int:
        if exercise_name not in self._name_to_index:
            self._name_to_index[exercise_name] = len(self._name_to_index)
            logger.debug("LabelEncoder: added new exercise class '%s'", exercise_name)
        return self._name_to_index[exercise_name]

    def decode(self, index: int) -> str:
        for name, idx in self._name_to_index.items():
            if idx == index:
                return name
        raise KeyError(f"No exercise name registered for index {index}")

    @property
    def mapping(self) -> Dict[str, int]:
        return dict(self._name_to_index)


class LabelProcessor:
    """Normalizes dataset-native labels/scores into a UnifiedLabel."""

    def __init__(
        self,
        exercise_encoder: Optional[LabelEncoder] = None,
        clinician_score_range: Optional[tuple] = None,
        binary_threshold: float = 0.6,
    ):
        self.exercise_encoder = exercise_encoder or LabelEncoder()
        self.clinician_score_range = clinician_score_range  # e.g. (0, 100) for KIMORE
        self.binary_threshold = binary_threshold

    def normalize_clinician_score(self, raw_score: float) -> float:
        """Scale a dataset-native clinician score into [0, 1]."""
        if self.clinician_score_range is None:
            raise ValueError("clinician_score_range must be set to normalize clinician scores.")
        low, high = self.clinician_score_range
        if high == low:
            return 0.0
        return float(np.clip((raw_score - low) / (high - low), 0.0, 1.0))

    def derive_binary_correctness(self, normalized_score: float) -> int:
        """Threshold a normalized [0, 1] quality score into correct(1)/incorrect(0)."""
        return int(normalized_score >= self.binary_threshold)

    def build_label(
        self,
        exercise_name: str,
        quality_class: Optional[int] = None,
        raw_clinician_score: Optional[float] = None,
        regression_target: Optional[float] = None,
    ) -> UnifiedLabel:
        """Assemble a UnifiedLabel from whatever native label fields a
        given dataset sample provides. Missing fields are derived where
        possible (e.g. binary correctness from a normalized clinician
        score) and left as None otherwise.
        """
        exercise_index = self.exercise_encoder.encode(exercise_name)

        normalized_score = regression_target
        if normalized_score is None and raw_clinician_score is not None and self.clinician_score_range is not None:
            normalized_score = self.normalize_clinician_score(raw_clinician_score)

        binary_correct = None
        if normalized_score is not None:
            binary_correct = self.derive_binary_correctness(normalized_score)

        resolved_quality_class = quality_class
        if resolved_quality_class is None and binary_correct is not None:
            resolved_quality_class = binary_correct

        return UnifiedLabel(
            exercise_name=exercise_name,
            quality_class=int(resolved_quality_class) if resolved_quality_class is not None else -1,
            regression_target=normalized_score,
            clinician_score=raw_clinician_score,
            binary_correct=binary_correct,
        )
