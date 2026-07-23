"""Tests for inference.calibration.CalibrationSession."""

import numpy as np
import pytest

from inference.calibration import CalibrationSession
from utils.joint_angles import MEDIAPIPE_LANDMARK_INDEX


def _neutral_landmarks():
    lm = np.zeros((33, 3))
    lm[MEDIAPIPE_LANDMARK_INDEX["left_shoulder"]] = [0.35, 0.3, 0]
    lm[MEDIAPIPE_LANDMARK_INDEX["right_shoulder"]] = [0.65, 0.3, 0]
    lm[MEDIAPIPE_LANDMARK_INDEX["left_elbow"]] = [0.3, 0.5, 0]
    lm[MEDIAPIPE_LANDMARK_INDEX["right_elbow"]] = [0.7, 0.5, 0]
    lm[MEDIAPIPE_LANDMARK_INDEX["left_wrist"]] = [0.3, 0.7, 0]
    lm[MEDIAPIPE_LANDMARK_INDEX["right_wrist"]] = [0.7, 0.7, 0]
    lm[MEDIAPIPE_LANDMARK_INDEX["left_hip"]] = [0.4, 0.65, 0]
    lm[MEDIAPIPE_LANDMARK_INDEX["right_hip"]] = [0.6, 0.65, 0]
    return lm


def test_calibration_completes_and_produces_profile():
    rng = np.random.default_rng(1)
    base = _neutral_landmarks()
    session = CalibrationSession(duration_seconds=0.01, min_frames_required=10, stability_std_threshold_deg=5.0)
    session.start()
    for _ in range(30):
        session.add_frame(base + rng.normal(0, 0.001, size=base.shape))

    import time
    time.sleep(0.02)
    assert session.is_complete

    profile = session.finalize()
    assert profile.shoulder_width > 0
    assert profile.torso_length > 0
    assert profile.num_frames_used == 30


def test_calibration_rejects_too_few_frames():
    session = CalibrationSession(duration_seconds=0.01, min_frames_required=50)
    session.start()
    for _ in range(5):
        session.add_frame(_neutral_landmarks())

    with pytest.raises(RuntimeError, match="only"):
        session.finalize()


def test_calibration_rejects_unstable_pose():
    rng = np.random.default_rng(2)
    base = _neutral_landmarks()
    session = CalibrationSession(duration_seconds=0.01, min_frames_required=10, stability_std_threshold_deg=1.0)
    session.start()
    for i in range(30):
        # Introduce large, systematic elbow-angle drift -> unstable
        wobble = base.copy()
        wobble[MEDIAPIPE_LANDMARK_INDEX["left_wrist"]][0] += 0.05 * np.sin(i)
        session.add_frame(wobble)

    with pytest.raises(RuntimeError, match="not stable"):
        session.finalize()


def test_add_frame_skips_none_landmarks():
    session = CalibrationSession(duration_seconds=0.01, min_frames_required=5)
    session.start()
    accepted = session.add_frame(None)
    assert not accepted


def test_add_frame_before_start_raises():
    session = CalibrationSession()
    with pytest.raises(RuntimeError):
        session.add_frame(_neutral_landmarks())
