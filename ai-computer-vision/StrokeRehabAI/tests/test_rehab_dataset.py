"""Tests for datasets.rehab_dataset.RehabExerciseDataset."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from datasets.rehab_dataset import RehabExerciseDataset


def _write_synthetic_samples(root, num_samples=12, num_subjects=4):
    rng = np.random.default_rng(0)
    for i in range(num_samples):
        subject = f"subj{i % num_subjects}"
        num_frames = int(rng.integers(60, 100))
        landmarks = rng.uniform(0, 1, size=(num_frames, 33, 3))
        label = int(rng.integers(0, 5))
        metadata = {"sample_id": f"s{i}", "subject_id": subject, "exercise_type": "ex", "source_dataset": "TEST"}
        np.savez_compressed(root / f"s{i}.npz", landmarks=landmarks, label=label, metadata=metadata)


@pytest.fixture
def processed_dir(tmp_path):
    _write_synthetic_samples(tmp_path)
    return tmp_path


def test_graph_representation_shape(processed_dir):
    ds = RehabExerciseDataset(
        processed_dir=str(processed_dir), sequence_length=30, stride=15,
        data_representation="graph", in_channels=3,
    )
    assert len(ds) > 0
    x, y = ds[0]
    assert x.shape == (3, 30, 33)
    assert y.dtype == torch.long


def test_flat_representation_shape(processed_dir):
    ds = RehabExerciseDataset(
        processed_dir=str(processed_dir), sequence_length=30, stride=15,
        data_representation="flat",
    )
    x, y = ds[0]
    assert x.shape == (30, 99)


def test_discover_subjects_maps_every_file(processed_dir):
    mapping = RehabExerciseDataset.discover_subjects(str(processed_dir))
    assert len(mapping) == 12
    assert all(v.startswith("subj") for v in mapping.values())


def test_split_by_subject_has_no_overlap(processed_dir):
    ds = RehabExerciseDataset(processed_dir=str(processed_dir), sequence_length=30, stride=15)
    train_idx, val_idx, test_idx = ds.split(0.5, 0.25, seed=1, split_by="subject")
    assert set(train_idx).isdisjoint(val_idx)
    assert set(train_idx).isdisjoint(test_idx)
    assert set(val_idx).isdisjoint(test_idx)
