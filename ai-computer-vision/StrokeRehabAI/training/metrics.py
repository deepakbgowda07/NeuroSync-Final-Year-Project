"""
metrics.py
==========
Training/validation/test metric computation: classification metrics
(accuracy, precision, recall, F1, ROC AUC, confusion matrix) via
scikit-learn, plus regression metrics (MAE, RMSE) and the
domain-specific joint-angle error metric used to sanity-check that the
model's implicit understanding of movement correlates with measurable
kinematic differences.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


def compute_classification_metrics(
    preds, targets, num_classes: int, probabilities: Optional[np.ndarray] = None
) -> Dict[str, object]:
    """Compute accuracy/precision/recall/F1/confusion-matrix (and ROC AUC
    if per-class probabilities are supplied) for one batch or epoch.

    `preds` and `targets` are 1D integer arrays/tensors of predicted and
    true class labels. `probabilities`, if given, is (N, num_classes) and
    enables one-vs-rest ROC AUC via scikit-learn.
    """
    preds_np = _to_numpy(preds)
    targets_np = _to_numpy(targets)

    metrics: Dict[str, object] = {}

    if len(preds_np) == 0 or len(targets_np) == 0:
        logger.warning(
            "compute_classification_metrics called with an empty batch/split "
            "(likely too few samples for the configured train/val/test ratios) "
            "— returning None-filled metrics rather than crashing."
        )
        return {
            "accuracy": None, "precision": None, "recall": None, "f1_score": None,
            "confusion_matrix": None, "roc_auc": None,
        }

    try:
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            confusion_matrix, roc_auc_score,
        )

        metrics["accuracy"] = float(accuracy_score(targets_np, preds_np))
        metrics["precision"] = float(precision_score(targets_np, preds_np, average="macro", zero_division=0))
        metrics["recall"] = float(recall_score(targets_np, preds_np, average="macro", zero_division=0))
        metrics["f1_score"] = float(f1_score(targets_np, preds_np, average="macro", zero_division=0))
        metrics["confusion_matrix"] = confusion_matrix(
            targets_np, preds_np, labels=list(range(num_classes))
        ).tolist()

        if probabilities is not None:
            try:
                probs_np = _to_numpy(probabilities)
                if num_classes == 2:
                    metrics["roc_auc"] = float(roc_auc_score(targets_np, probs_np[:, 1]))
                else:
                    metrics["roc_auc"] = float(
                        roc_auc_score(targets_np, probs_np, multi_class="ovr", labels=list(range(num_classes)))
                    )
            except ValueError as exc:
                logger.debug("ROC AUC could not be computed (likely missing classes in batch): %s", exc)
                metrics["roc_auc"] = None

    except ImportError:
        logger.warning("scikit-learn not installed; falling back to accuracy-only metrics.")
        accuracy = float((preds_np == targets_np).mean()) if len(preds_np) else 0.0
        metrics = {"accuracy": accuracy, "precision": None, "recall": None, "f1_score": None, "confusion_matrix": None}

    return metrics


def compute_regression_metrics(preds, targets) -> Dict[str, float]:
    """MAE and RMSE for continuous regression targets (e.g. normalized
    quality score, when model.output_head == 'regression')."""
    preds_np = _to_numpy(preds).astype(np.float64)
    targets_np = _to_numpy(targets).astype(np.float64)

    if len(preds_np) == 0:
        return {"mae": 0.0, "rmse": 0.0}

    errors = preds_np - targets_np
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    return {"mae": mae, "rmse": rmse}


def joint_angle_error(predicted_angles_deg: np.ndarray, reference_angles_deg: np.ndarray) -> Dict[str, float]:
    """Mean absolute and max joint-angle error (degrees) between a
    predicted/reconstructed pose sequence and a reference sequence —
    e.g. for auxiliary pose-consistency checks, or comparing a patient's
    measured angles against a clinician-annotated reference trajectory.

    Both inputs are (T, num_angles) arrays over the same angle set (see
    utils.joint_angles.JOINT_ANGLE_DEFINITIONS).
    """
    predicted = np.asarray(predicted_angles_deg, dtype=np.float64)
    reference = np.asarray(reference_angles_deg, dtype=np.float64)

    if predicted.shape != reference.shape:
        raise ValueError(f"Shape mismatch: predicted {predicted.shape} vs reference {reference.shape}")

    abs_error = np.abs(predicted - reference)
    return {
        "mean_absolute_error_deg": float(np.nanmean(abs_error)),
        "max_absolute_error_deg": float(np.nanmax(abs_error)),
        "rmse_deg": float(np.sqrt(np.nanmean(abs_error ** 2))),
    }


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)
