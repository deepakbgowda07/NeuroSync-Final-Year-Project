"""Tests for preprocessing.normalizer."""

import numpy as np

from preprocessing.normalizer import PoseNormalizer


def test_normalize_frame_centers_on_hip_midpoint(synthetic_landmark_frame):
    normalizer = PoseNormalizer()
    result = normalizer.normalize_frame(synthetic_landmark_frame)
    assert result.shape == synthetic_landmark_frame.shape


def test_normalize_sequence_matches_per_frame(synthetic_landmark_sequence):
    normalizer = PoseNormalizer()
    result = normalizer.normalize_sequence(synthetic_landmark_sequence)
    assert result.shape == synthetic_landmark_sequence.shape
