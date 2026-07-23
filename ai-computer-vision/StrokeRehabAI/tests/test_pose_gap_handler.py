"""Tests for mediapipe_pipeline.pose_smoothing.PoseGapHandler (temporary landmark loss handling)."""

import numpy as np

from mediapipe_pipeline.pose_smoothing import PoseGapHandler


def test_holds_last_known_pose_during_brief_gap():
    handler = PoseGapHandler(max_hold_frames=5)
    valid = np.ones((33, 3))
    result, is_held = handler.update(valid)
    assert not is_held
    np.testing.assert_array_equal(result, valid)

    for _ in range(3):
        result, is_held = handler.update(None)
        assert is_held
        np.testing.assert_array_equal(result, valid)


def test_declares_pose_lost_after_max_hold_frames():
    handler = PoseGapHandler(max_hold_frames=3)
    handler.update(np.ones((33, 3)))
    for _ in range(3):
        handler.update(None)
    result, is_held = handler.update(None)
    assert result is None
    assert handler.is_lost


def test_recovers_after_new_valid_detection():
    handler = PoseGapHandler(max_hold_frames=2)
    handler.update(np.ones((33, 3)))
    handler.update(None)
    handler.update(None)
    handler.update(None)  # now lost
    assert handler.is_lost

    new_valid = np.full((33, 3), 2.0)
    result, is_held = handler.update(new_valid)
    assert not is_held
    assert not handler.is_lost
    np.testing.assert_array_equal(result, new_valid)


def test_reset_clears_state():
    handler = PoseGapHandler(max_hold_frames=2)
    handler.update(np.ones((33, 3)))
    handler.reset()
    result, is_held = handler.update(None)
    assert result is None
    assert not is_held
