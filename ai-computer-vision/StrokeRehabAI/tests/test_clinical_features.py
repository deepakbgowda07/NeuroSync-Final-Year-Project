"""Tests for feature_extraction.clinical_features."""

import numpy as np
import pytest

from feature_extraction.clinical_features import ClinicalFeatureExtractor
from utils.joint_angles import MEDIAPIPE_LANDMARK_INDEX


def _upright_sequence(num_frames=30):
    """A synthetic sequence where shoulders sit directly above hips
    (perfectly upright trunk, trunk_lean should be ~0deg)."""
    seq = np.zeros((num_frames, 33, 3))
    seq[:, MEDIAPIPE_LANDMARK_INDEX["left_hip"]] = [0.4, 0.6, 0]
    seq[:, MEDIAPIPE_LANDMARK_INDEX["right_hip"]] = [0.6, 0.6, 0]
    seq[:, MEDIAPIPE_LANDMARK_INDEX["left_shoulder"]] = [0.4, 0.3, 0]
    seq[:, MEDIAPIPE_LANDMARK_INDEX["right_shoulder"]] = [0.6, 0.3, 0]
    seq[:, MEDIAPIPE_LANDMARK_INDEX["left_wrist"]] = [0.3, 0.35, 0]
    seq[:, MEDIAPIPE_LANDMARK_INDEX["right_wrist"]] = [0.7, 0.35, 0]
    return seq


def test_trunk_lean_near_zero_for_upright_posture():
    extractor = ClinicalFeatureExtractor(fps=30)
    seq = _upright_sequence()
    lean = extractor.trunk_lean_deg(seq)
    assert np.all(lean < 5.0)


def test_exercise_duration_matches_frame_count_and_fps():
    extractor = ClinicalFeatureExtractor(fps=30)
    seq = _upright_sequence(num_frames=90)
    duration = extractor.exercise_duration_seconds(seq)
    assert duration == pytest.approx(3.0)


def test_compensation_indicators_returns_expected_keys():
    extractor = ClinicalFeatureExtractor(fps=30)
    seq = _upright_sequence()
    indicators = extractor.compensation_indicators(seq)
    assert "excessive_trunk_lean_ratio" in indicators
    assert "shoulder_hiking_ratio" in indicators
    assert 0.0 <= indicators["excessive_trunk_lean_ratio"] <= 1.0


def test_movement_smoothness_returns_finite_value():
    extractor = ClinicalFeatureExtractor(fps=30)
    rng = np.random.default_rng(0)
    seq = rng.uniform(0, 1, size=(60, 33, 3))
    smoothness = extractor.movement_smoothness(seq, MEDIAPIPE_LANDMARK_INDEX["left_wrist"])
    assert np.isfinite(smoothness)
