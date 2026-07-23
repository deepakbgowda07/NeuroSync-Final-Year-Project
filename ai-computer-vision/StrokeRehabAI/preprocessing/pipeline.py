"""
pipeline.py
===========
End-to-end preprocessing orchestration: raw landmarks -> normalized ->
(optionally augmented) -> windowed sequences ready for the model.
Used identically by both the training data pipeline and the live
inference pipeline, so behavior stays consistent between train/serve.

Supports two output representations, matching
models.model_factory.data_representation_for():

    "flat"  -> (sequence_length, num_joints * in_channels) windows,
               consumed by the LSTM baseline.
    "graph" -> (sequence_length, num_joints, in_channels) windows,
               consumed by the ST-GCN model (which further permutes to
               (in_channels, sequence_length, num_joints) at the model
               boundary — see inference/predictor.py and
               training/dataset_loader.py).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from preprocessing.augmentations import SequenceAugmenter
from preprocessing.normalizer import PoseNormalizer
from preprocessing.sequence_builder import SequenceBuilder
from utils.logger import get_logger

logger = get_logger(__name__)


class PreprocessingPipeline:
    """Composable preprocessing pipeline shared across training and inference."""

    def __init__(
        self,
        sequence_length: int = 90,
        stride: int = 15,
        feature_dim: int = 99,
        apply_augmentation: bool = False,
        augmenter: Optional[SequenceAugmenter] = None,
        data_representation: str = "flat",
    ):
        if data_representation not in ("flat", "graph"):
            raise ValueError(f"Unknown data_representation: {data_representation}")

        self.normalizer = PoseNormalizer()
        self.sequence_builder = SequenceBuilder(sequence_length=sequence_length, feature_dim=feature_dim, stride=stride)
        self.data_representation = data_representation
        if apply_augmentation:
            self.augmenter = augmenter or SequenceAugmenter()
        else:
            self.augmenter = None

    def reset(self) -> None:
        self.sequence_builder.reset()

    def process_frame(self, raw_landmarks_xyz: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Feed one frame's raw (33, 3) landmarks through normalization and
        the sliding-window buffer. Returns a completed (flat) window when
        ready. Live inference always uses the flat representation
        internally; inference/predictor.py reshapes to graph layout if
        the active model needs it."""
        if raw_landmarks_xyz is None:
            return self.sequence_builder.add_frame(None)

        normalized = self.normalizer.normalize_frame(raw_landmarks_xyz)
        flat = normalized.flatten()
        return self.sequence_builder.add_frame(flat)

    def process_offline_sequence(self, raw_sequence_xyz: np.ndarray) -> list:
        """Batch path used by the dataset pipeline: normalize a full
        recorded sequence, optionally augment it, then window it into
        either flat or graph-shaped windows depending on
        `data_representation`."""
        normalized = self.normalizer.normalize_sequence(raw_sequence_xyz)
        if self.augmenter is not None:
            normalized = self.augmenter(normalized)

        if self.data_representation == "graph":
            return SequenceBuilder.windows_from_full_sequence_graph(
                normalized, self.sequence_builder.sequence_length, self.sequence_builder.stride
            )

        flat = normalized.reshape(normalized.shape[0], -1)
        return SequenceBuilder.windows_from_full_sequence(
            flat, self.sequence_builder.sequence_length, self.sequence_builder.stride
        )
