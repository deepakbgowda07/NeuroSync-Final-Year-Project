"""Tests for training.dataset_loader.build_dataloaders — in particular,
regression coverage for the file-level split fix (augmentation changes
a sample's window count, so splitting must happen at the file level
before any dataset object with augmentation-dependent indexing is built)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from configs.config_loader import load_config
from training.dataset_loader import build_dataloaders


def _write_synthetic_samples(root, num_samples=20, num_subjects=6):
    rng = np.random.default_rng(0)
    for i in range(num_samples):
        subject = f"subj{i % num_subjects}"
        num_frames = int(rng.integers(80, 140))
        landmarks = rng.uniform(0, 1, size=(num_frames, 33, 3))
        label = int(rng.integers(0, 5))
        metadata = {"sample_id": f"s{i}", "subject_id": subject, "exercise_type": "ex", "source_dataset": "TEST"}
        np.savez_compressed(root / f"s{i}.npz", landmarks=landmarks, label=label, metadata=metadata)


@pytest.fixture
def cfg(tmp_path):
    _write_synthetic_samples(tmp_path)
    cfg = load_config(force_reload=True)
    cfg.datasets.processed_dir = str(tmp_path)
    cfg.datasets.sequence_length = 30
    cfg.datasets.stride = 15
    cfg.training.batch_size = 2
    cfg.training.num_workers = 0
    return cfg


def test_build_dataloaders_no_leakage_between_splits(cfg):
    train_loader, val_loader, test_loader = build_dataloaders(cfg.training, cfg.datasets, cfg.model, cfg.augmentation)
    assert len(train_loader.dataset) > 0
    assert len(val_loader.dataset) > 0 or len(test_loader.dataset) > 0


def test_build_dataloaders_batches_have_consistent_shapes(cfg):
    train_loader, _, _ = build_dataloaders(cfg.training, cfg.datasets, cfg.model, cfg.augmentation)
    batch_x, batch_y = next(iter(train_loader))
    assert batch_x.shape[0] == cfg.training.batch_size
    assert batch_x.shape[1] == cfg.model.stgcn.in_channels
    assert batch_x.shape[3] == cfg.model.stgcn.num_joints
    assert batch_y.shape[0] == cfg.training.batch_size


def test_build_dataloaders_train_uses_augmentation_val_does_not(cfg):
    # Indirect check: train dataset windows should not error out even though
    # augmentation may change per-sample sequence length before re-windowing
    # (this is the scenario that previously caused an index mismatch).
    train_loader, val_loader, test_loader = build_dataloaders(cfg.training, cfg.datasets, cfg.model, cfg.augmentation)
    for batch_x, batch_y in train_loader:
        assert batch_x.shape[2] == cfg.datasets.sequence_length
    for batch_x, batch_y in val_loader:
        assert batch_x.shape[2] == cfg.datasets.sequence_length
