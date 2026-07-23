"""
dataset_loader.py
==================
Wraps datasets.rehab_dataset.RehabExerciseDataset in PyTorch DataLoaders
with train/val/test splits and config-driven batch size / worker count.

Automatically selects the flat (LSTM) or graph (ST-GCN) data
representation based on the active model architecture (see
models.model_factory.data_representation_for), so switching
`model.architecture` in configs/model.yaml doesn't require touching
training code.

Splitting happens at the *file* level (via
RehabExerciseDataset.discover_subjects + SplitManager) before any
dataset object is built. This matters because the training split uses
augmentation (which can change a sample's window count via temporal
crop/drop/stretch) while val/test do not — building one shared
"full_dataset" and reusing its window indices across differently
augmented dataset instances would silently misalign windows to labels.
Instead, three independent RehabExerciseDataset instances are built,
each restricted to its own file subset via `file_list`.
"""

from __future__ import annotations

from typing import Tuple

from datasets.rehab_dataset import RehabExerciseDataset
from datasets.split_manager import SplitManager
from models.model_factory import data_representation_for
from training.augmentation import build_augmenter
from utils.logger import get_logger

logger = get_logger(__name__)


def build_dataloaders(training_cfg, datasets_cfg, model_cfg, augmentation_cfg=None) -> Tuple:
    """Return (train_loader, val_loader, test_loader) built from config."""
    from torch.utils.data import DataLoader

    representation = data_representation_for(model_cfg)
    in_channels = model_cfg.stgcn.in_channels if representation == "graph" else 3

    split_seed = datasets_cfg.split_seed if hasattr(datasets_cfg, "split_seed") else training_cfg.seed
    split_by = datasets_cfg.split_by if hasattr(datasets_cfg, "split_by") else "subject"

    file_to_subject = RehabExerciseDataset.discover_subjects(datasets_cfg.processed_dir)
    if not file_to_subject:
        logger.warning(
            "No processed .npz samples found in %s — dataloaders will be empty. "
            "Run datasets/dataset_converter.py first (see docs/dataset_guide.md).",
            datasets_cfg.processed_dir,
        )

    manager = SplitManager(datasets_cfg.train_split, datasets_cfg.val_split, datasets_cfg.test_split, seed=split_seed)
    path_by_str = {str(p): p for p in file_to_subject}

    if split_by == "subject":
        split_result = manager.split_by_subject({str(p): subj for p, subj in file_to_subject.items()})
    else:
        split_result = manager.split_by_sample([str(p) for p in file_to_subject])

    train_files = [path_by_str[s] for s in split_result.train_ids]
    val_files = [path_by_str[s] for s in split_result.val_ids]
    test_files = [path_by_str[s] for s in split_result.test_ids]

    for split_name, files in (("train", train_files), ("val", val_files), ("test", test_files)):
        if not files:
            logger.warning(
                "The '%s' split is empty — likely too few subjects/samples for the "
                "configured split ratios (train=%.2f, val=%.2f, test=%.2f). Metrics "
                "for this split will be reported as None until more data is added.",
                split_name, datasets_cfg.train_split, datasets_cfg.val_split, datasets_cfg.test_split,
            )

    common_kwargs = dict(
        processed_dir=datasets_cfg.processed_dir,
        sequence_length=datasets_cfg.sequence_length,
        stride=datasets_cfg.stride,
        data_representation=representation,
        in_channels=in_channels,
    )

    train_dataset = RehabExerciseDataset(
        file_list=train_files, apply_augmentation=True, augmenter=build_augmenter(augmentation_cfg), **common_kwargs
    )
    val_dataset = RehabExerciseDataset(file_list=val_files, apply_augmentation=False, **common_kwargs)
    test_dataset = RehabExerciseDataset(file_list=test_files, apply_augmentation=False, **common_kwargs)

    num_workers = training_cfg.num_workers
    pin_memory = bool(getattr(training_cfg, "pin_memory", True))
    persistent_workers = num_workers > 0
    prefetch_factor = 4 if num_workers > 0 else None

    loader_kwargs = dict(num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    if prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(train_dataset, batch_size=training_cfg.batch_size, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=training_cfg.batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=training_cfg.batch_size, shuffle=False, **loader_kwargs)

    logger.info(
        "Dataloaders built (representation=%s, split_by=%s): train=%d files/%d windows, "
        "val=%d files/%d windows, test=%d files/%d windows, batch_size=%d, num_workers=%d, pin_memory=%s",
        representation, split_by,
        len(train_files), len(train_dataset), len(val_files), len(val_dataset), len(test_files), len(test_dataset),
        training_cfg.batch_size, num_workers, pin_memory,
    )
    return train_loader, val_loader, test_loader
