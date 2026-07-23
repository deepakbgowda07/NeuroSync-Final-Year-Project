"""Tests for preprocessing.augmentations."""

import numpy as np

from preprocessing.augmentations import AugmentationConfig, MIRROR_INDEX_MAP, SequenceAugmenter


def test_mirror_index_map_swaps_left_and_right():
    from mediapipe_pipeline.landmark_extractor import LANDMARK_NAMES

    left_shoulder_idx = LANDMARK_NAMES.index("left_shoulder")
    right_shoulder_idx = LANDMARK_NAMES.index("right_shoulder")
    assert MIRROR_INDEX_MAP[left_shoulder_idx] == right_shoulder_idx
    assert MIRROR_INDEX_MAP[right_shoulder_idx] == left_shoulder_idx


def test_mirror_index_map_leaves_midline_joints_unchanged():
    from mediapipe_pipeline.landmark_extractor import LANDMARK_NAMES

    nose_idx = LANDMARK_NAMES.index("nose")
    assert MIRROR_INDEX_MAP[nose_idx] == nose_idx


def test_augmenter_preserves_joint_and_channel_count(synthetic_landmark_sequence):
    config = AugmentationConfig(temporal_crop_prob=0.0, random_frame_drop_prob=0.0, temporal_stretch_prob=0.0)
    augmenter = SequenceAugmenter(config=config, rng=np.random.default_rng(0))
    result = augmenter(synthetic_landmark_sequence)
    assert result.shape[1:] == synthetic_landmark_sequence.shape[1:]


def test_temporal_crop_reduces_or_keeps_length():
    config = AugmentationConfig(
        temporal_crop_prob=1.0, temporal_crop_min_ratio=0.5,
        random_frame_drop_prob=0.0, temporal_stretch_prob=0.0,
        rotation_deg_std=0.0, noise_std=0.0, scale_jitter_std=0.0,
    )
    augmenter = SequenceAugmenter(config=config, rng=np.random.default_rng(1))
    seq = np.random.default_rng(1).uniform(0, 1, size=(100, 33, 3))
    result = augmenter(seq)
    assert 50 <= len(result) <= 100


def test_temporal_stretch_changes_length_within_range():
    config = AugmentationConfig(
        temporal_stretch_prob=1.0, temporal_stretch_range=(0.5, 0.5),
        temporal_crop_prob=0.0, random_frame_drop_prob=0.0,
        rotation_deg_std=0.0, noise_std=0.0, scale_jitter_std=0.0,
    )
    augmenter = SequenceAugmenter(config=config, rng=np.random.default_rng(2))
    seq = np.random.default_rng(2).uniform(0, 1, size=(100, 33, 3))
    result = augmenter(seq)
    assert abs(len(result) - 50) <= 1


def test_mirror_negates_x_axis_when_forced():
    config = AugmentationConfig(
        mirror_prob=1.0, valid_for_mirroring=True,
        temporal_crop_prob=0.0, random_frame_drop_prob=0.0, temporal_stretch_prob=0.0,
        rotation_deg_std=0.0, noise_std=0.0, scale_jitter_std=0.0,
    )
    augmenter = SequenceAugmenter(config=config, rng=np.random.default_rng(3))
    seq = np.ones((5, 33, 3)) * 0.5
    result = augmenter(seq)
    assert np.allclose(result[..., 0], -0.5)
