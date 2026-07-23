"""Tests for training.metrics."""

import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")

from training.metrics import compute_classification_metrics, compute_regression_metrics, joint_angle_error


def test_compute_classification_metrics_perfect_predictions():
    preds = np.array([0, 1, 2, 1, 0])
    targets = np.array([0, 1, 2, 1, 0])
    metrics = compute_classification_metrics(preds, targets, num_classes=3)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert len(metrics["confusion_matrix"]) == 3


def test_compute_classification_metrics_with_probabilities_computes_roc_auc():
    preds = np.array([0, 1, 0, 1])
    targets = np.array([0, 1, 0, 1])
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])
    metrics = compute_classification_metrics(preds, targets, num_classes=2, probabilities=probs)
    assert metrics["roc_auc"] is not None
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_compute_regression_metrics():
    preds = np.array([1.0, 2.0, 3.0])
    targets = np.array([1.5, 2.0, 2.5])
    metrics = compute_regression_metrics(preds, targets)
    assert metrics["mae"] == pytest.approx(1 / 3, abs=1e-6)
    assert metrics["rmse"] > 0


def test_joint_angle_error_zero_when_identical():
    angles = np.random.default_rng(0).uniform(0, 180, size=(10, 8))
    result = joint_angle_error(angles, angles)
    assert result["mean_absolute_error_deg"] == 0.0
    assert result["max_absolute_error_deg"] == 0.0


def test_joint_angle_error_shape_mismatch_raises():
    a = np.zeros((10, 8))
    b = np.zeros((10, 9))
    with pytest.raises(ValueError):
        joint_angle_error(a, b)
