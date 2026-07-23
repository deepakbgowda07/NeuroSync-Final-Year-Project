"""Tests for preprocessing.sequence_builder."""

import numpy as np

from preprocessing.sequence_builder import SequenceBuilder


def test_add_frame_returns_none_until_buffer_full():
    builder = SequenceBuilder(sequence_length=5, feature_dim=3, stride=1)
    for _ in range(4):
        assert builder.add_frame(np.zeros(3)) is None


def test_add_frame_returns_window_when_full_and_stride_met():
    builder = SequenceBuilder(sequence_length=5, feature_dim=3, stride=1)
    window = None
    for _ in range(5):
        window = builder.add_frame(np.zeros(3))
    assert window is not None
    assert window.shape == (5, 3)


def test_add_frame_handles_missing_detection_by_holding_last():
    builder = SequenceBuilder(sequence_length=3, feature_dim=2, stride=1)
    builder.add_frame(np.array([1.0, 2.0]))
    window = None
    for _ in range(2):
        window = builder.add_frame(None)
    assert window is not None
    np.testing.assert_array_equal(window[-1], [1.0, 2.0])


def test_windows_from_full_sequence():
    full_seq = np.arange(20 * 3).reshape(20, 3).astype(float)
    windows = SequenceBuilder.windows_from_full_sequence(full_seq, sequence_length=5, stride=5)
    assert len(windows) == 4
    assert all(w.shape == (5, 3) for w in windows)
