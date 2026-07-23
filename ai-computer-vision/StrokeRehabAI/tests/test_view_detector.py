"""Tests for mediapipe_pipeline.view_detector."""

import numpy as np

from mediapipe_pipeline.view_detector import CameraView, ViewDetector
from utils.joint_angles import MEDIAPIPE_LANDMARK_INDEX


def _landmarks_with_shoulders(left_shoulder, right_shoulder):
    lm = np.zeros((33, 3))
    lm[MEDIAPIPE_LANDMARK_INDEX["left_shoulder"]] = left_shoulder
    lm[MEDIAPIPE_LANDMARK_INDEX["right_shoulder"]] = right_shoulder
    return lm


def test_classify_front_view():
    detector = ViewDetector()
    lm = _landmarks_with_shoulders([0.35, 0.3, 0.0], [0.65, 0.3, 0.0])
    result = detector.classify_frame(lm)
    assert result.view == CameraView.FRONT


def test_classify_left_side_view():
    detector = ViewDetector()
    lm = _landmarks_with_shoulders([0.5, 0.3, -0.15], [0.5, 0.3, 0.05])
    result = detector.classify_frame(lm)
    assert result.view == CameraView.LEFT_SIDE


def test_classify_right_side_view():
    detector = ViewDetector()
    lm = _landmarks_with_shoulders([0.5, 0.3, 0.05], [0.5, 0.3, -0.15])
    result = detector.classify_frame(lm)
    assert result.view == CameraView.RIGHT_SIDE


def test_detect_applies_temporal_smoothing():
    detector = ViewDetector(smoothing_window=5)
    front_lm = _landmarks_with_shoulders([0.35, 0.3, 0.0], [0.65, 0.3, 0.0])
    side_lm = _landmarks_with_shoulders([0.5, 0.3, -0.15], [0.5, 0.3, 0.05])

    for _ in range(4):
        detector.detect(front_lm)
    result = detector.detect(side_lm)  # a single noisy side-view frame shouldn't flip majority
    assert result.view == CameraView.FRONT


def test_matches_required_view():
    front_result = ViewDetector().classify_frame(_landmarks_with_shoulders([0.35, 0.3, 0.0], [0.65, 0.3, 0.0]))
    assert ViewDetector.matches_required_view(front_result, "front")
    assert not ViewDetector.matches_required_view(front_result, "side")
    assert ViewDetector.matches_required_view(front_result, "any")
