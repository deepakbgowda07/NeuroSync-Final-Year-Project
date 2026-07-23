"""
rehab_dataset.py
=================
PyTorch Dataset implementation over the unified processed-data format
(.npz files produced by dataset_converter.py). This is what
training/dataset_loader.py wraps in a DataLoader.

Supports both data representations the model factory understands (see
models.model_factory.data_representation_for):
    "flat"  -> (sequence_length, num_joints * in_channels), for the LSTM baseline
    "graph" -> (in_channels, sequence_length, num_joints), for ST-GCN (default)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from datasets.split_manager import SplitManager
from preprocessing.augmentations import SequenceAugmenter
from preprocessing.pipeline import PreprocessingPipeline
from utils.logger import get_logger

logger = get_logger(__name__)


class RehabExerciseDataset:
    """torch.utils.data.Dataset-compatible dataset over processed .npz samples.

    Deliberately avoids importing torch at module scope so this class
    can be introspected/tested without a torch installation; the
    __getitem__ return values are converted to tensors lazily.
    """

    def __init__(
        self,
        processed_dir: str = "data/processed",
        sequence_length: int = 90,
        stride: int = 15,
        apply_augmentation: bool = False,
        augmenter: Optional[SequenceAugmenter] = None,
        data_representation: str = "graph",
        in_channels: int = 3,
        file_list: Optional[List[Path]] = None,
    ):
        self.processed_dir = Path(processed_dir)
        self.sequence_length = sequence_length
        self.stride = stride
        self.data_representation = data_representation
        self.in_channels = in_channels
        self.pipeline = PreprocessingPipeline(
            sequence_length=sequence_length,
            stride=stride,
            apply_augmentation=apply_augmentation,
            augmenter=augmenter,
            data_representation=data_representation,
        )
        self.files = file_list if file_list is not None else sorted(self.processed_dir.rglob("*.npz"))

        if not self.files:
            logger.warning("RehabExerciseDataset found no .npz files in %s", self.processed_dir)

        self._index: List[Tuple[int, np.ndarray, int]] = []
        self._file_subject_ids: List[str] = []
        self._build_index()

    @staticmethod
    def discover_subjects(processed_dir: str) -> Dict[Path, str]:
        """Cheaply map every `.npz` file under `processed_dir` to its
        subject_id by reading only the `metadata` array (never
        `landmarks`) — used to split at the file level *before* any
        dataset object (and its augmentation-dependent window count) is
        built, avoiding index mismatches between differently-augmented
        dataset instances built from the same file set.
        """
        mapping: Dict[Path, str] = {}
        for f in sorted(Path(processed_dir).rglob("*.npz")):
            with np.load(f, allow_pickle=True) as data:
                metadata = data["metadata"].item() if "metadata" in data else {}
            mapping[f] = str(metadata.get("subject_id", f.stem))
        return mapping

    def _build_index(self) -> None:
        """Pre-compute all sliding windows across all sample files so
        __getitem__ has O(1) random access (standard PyTorch Dataset
        expectation)."""
        for file_idx, f in enumerate(self.files):
            with np.load(f, allow_pickle=True) as data:
                landmarks = data["landmarks"]  # (T, 33, 3)
                label = data["label"]
                metadata = data["metadata"].item() if "metadata" in data else {}

            self._file_subject_ids.append(str(metadata.get("subject_id", f"unknown_{file_idx}")))

            windows = self.pipeline.process_offline_sequence(landmarks)
            for window in windows:
                if self.data_representation == "graph":
                    # window: (T, V, C) -> keep channel-first for the model at __getitem__ time
                    window = window[..., : self.in_channels]
                self._index.append((file_idx, window, label))

        logger.info(
            "RehabExerciseDataset indexed %d windows from %d files (representation=%s).",
            len(self._index), len(self.files), self.data_representation,
        )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        import torch

        _, window, label = self._index[idx]

        if self.data_representation == "graph":
            # (T, V, C) -> (C, T, V), what models.stgcn_model.STGCN expects.
            window = np.transpose(window, (2, 0, 1))

        window_tensor = torch.tensor(window, dtype=torch.float32)
        label_tensor = torch.tensor(int(label), dtype=torch.long)
        return window_tensor, label_tensor

    def sample_id_to_subject_map(self) -> Dict[str, str]:
        """Maps each window's index (as a string id) to its source
        file's subject_id — the input SplitManager.split_by_subject expects."""
        return {str(i): self._file_subject_ids[file_idx] for i, (file_idx, _, _) in enumerate(self._index)}

    def split(self, train_ratio: float, val_ratio: float, seed: int = 42, split_by: str = "subject"):
        """Return (train_indices, val_indices, test_indices) using a
        deterministic split. `split_by="subject"` (default) guarantees no
        patient's windows appear in more than one split, preventing data
        leakage via overlapping windows from the same recording;
        `split_by="sample"` splits by source file only (still leak-free
        across files, but does not group multiple files per subject)."""
        test_ratio = max(0.0, 1.0 - train_ratio - val_ratio)
        manager = SplitManager(train_ratio, val_ratio, test_ratio, seed=seed)

        if split_by == "subject":
            id_to_subject = self.sample_id_to_subject_map()
            result = manager.split_by_subject(id_to_subject)
        else:
            result = manager.split_by_sample([str(i) for i in range(len(self._index))])

        return (
            [int(i) for i in result.train_ids],
            [int(i) for i in result.val_ids],
            [int(i) for i in result.test_ids],
        )
