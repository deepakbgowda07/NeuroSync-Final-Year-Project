"""Tests for utils.joint_angles."""

import numpy as np

from utils.joint_angles import angle_between_points, compute_all_joint_angles


def test_straight_line_angle_is_180():
    a, b, c = (0, 0, 0), (1, 0, 0), (2, 0, 0)
    assert np.isclose(angle_between_points(a, b, c), 180.0)


def test_right_angle():
    a, b, c = (1, 0, 0), (0, 0, 0), (0, 1, 0)
    assert np.isclose(angle_between_points(a, b, c), 90.0)


def test_degenerate_points_returns_zero():
    a = b = c = (0, 0, 0)
    assert angle_between_points(a, b, c) == 0.0


def test_compute_all_joint_angles_returns_expected_keys(synthetic_landmark_frame):
    angles = compute_all_joint_angles(synthetic_landmark_frame)
    assert "left_elbow_angle" in angles
    assert "right_knee_angle" in angles
    assert "left_shoulder_abduction_angle" in angles
    assert "right_forearm_rotation_proxy" in angles
    assert len(angles) == 14
