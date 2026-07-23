"""Tests for visualization.skeleton_renderer.SkeletonRenderer (green/red joint coloring)."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from configs.config_loader import load_config
from inference.error_detector import DetectedError, ErrorType
from utils.joint_angles import MEDIAPIPE_LANDMARK_INDEX
from visualization.skeleton_renderer import SkeletonRenderer, _ERROR_TO_JOINTS


@pytest.fixture
def renderer():
    cfg = load_config(force_reload=True)
    return SkeletonRenderer(cfg.visualization)


def _synthetic_landmarks():
    rng = np.random.default_rng(0)
    lm = rng.uniform(0.1, 0.9, size=(33, 3))
    lm[:, 2] = 0.0
    return lm


def test_flagged_joints_empty_with_no_errors(renderer):
    assert renderer._flagged_joints(None) == set()
    assert renderer._flagged_joints([]) == set()


def test_flagged_joints_maps_shoulder_hiking_to_shoulders(renderer):
    errors = [DetectedError(ErrorType.SHOULDER_HIKING, "minor", {})]
    flagged = renderer._flagged_joints(errors)
    assert flagged == _ERROR_TO_JOINTS[ErrorType.SHOULDER_HIKING]


def test_draw_produces_a_modified_frame_without_crashing(renderer):
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    lm = _synthetic_landmarks()
    result = renderer.draw(frame.copy(), lm, errors=[DetectedError(ErrorType.TRUNK_COMPENSATION, "moderate", {})])
    assert result.shape == frame.shape
    assert not np.array_equal(result, frame)  # something was drawn


def test_draw_with_no_errors_uses_correct_color(renderer):
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    lm = _synthetic_landmarks()
    result = renderer.draw(frame.copy(), lm, errors=None)
    # correct_color joints should appear somewhere in the frame
    assert tuple(renderer.correct_color) in [tuple(px) for px in result.reshape(-1, 3)]
