"""Tests for datasets.split_manager."""

import pytest

from datasets.split_manager import SplitManager


def test_split_ratios_must_sum_to_one():
    with pytest.raises(ValueError):
        SplitManager(0.5, 0.3, 0.3)


def test_split_by_sample_partitions_all_ids():
    manager = SplitManager(0.6, 0.2, 0.2, seed=1)
    ids = [f"sample_{i}" for i in range(20)]
    result = manager.split_by_sample(ids)

    all_assigned = set(result.train_ids) | set(result.val_ids) | set(result.test_ids)
    assert all_assigned == set(ids)
    assert len(set(result.train_ids) & set(result.val_ids)) == 0
    assert len(set(result.train_ids) & set(result.test_ids)) == 0


def test_split_by_sample_is_deterministic_given_seed():
    ids = [f"sample_{i}" for i in range(20)]
    result_a = SplitManager(0.6, 0.2, 0.2, seed=7).split_by_sample(ids)
    result_b = SplitManager(0.6, 0.2, 0.2, seed=7).split_by_sample(ids)
    assert result_a.train_ids == result_b.train_ids


def test_split_by_subject_keeps_all_samples_of_a_subject_together():
    id_to_subject = {}
    for subject_idx in range(10):
        for sample_idx in range(3):
            id_to_subject[f"s{subject_idx}_sample{sample_idx}"] = f"subject_{subject_idx}"

    manager = SplitManager(0.7, 0.15, 0.15, seed=42)
    result = manager.split_by_subject(id_to_subject)

    def subjects_in(ids):
        return {id_to_subject[i] for i in ids}

    train_subjects = subjects_in(result.train_ids)
    val_subjects = subjects_in(result.val_ids)
    test_subjects = subjects_in(result.test_ids)

    assert train_subjects.isdisjoint(val_subjects)
    assert train_subjects.isdisjoint(test_subjects)
    assert val_subjects.isdisjoint(test_subjects)
