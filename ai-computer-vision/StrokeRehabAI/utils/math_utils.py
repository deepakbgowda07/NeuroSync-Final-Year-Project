"""
math_utils.py
=============
Small numeric helpers shared across preprocessing, evaluation, and
visualization (smoothing, normalization, clamping).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def min_max_normalize(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """Scale values to [0, 1] along the given axis. Safe against zero range."""
    v_min = values.min(axis=axis, keepdims=True)
    v_max = values.max(axis=axis, keepdims=True)
    denom = np.where((v_max - v_min) < 1e-8, 1.0, v_max - v_min)
    return (values - v_min) / denom


def z_score_normalize(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """Standardize values to zero mean / unit variance along an axis."""
    mean = values.mean(axis=axis, keepdims=True)
    std = values.std(axis=axis, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (values - mean) / std


def moving_average(values: Sequence[float], window: int = 5) -> np.ndarray:
    """Simple moving average smoothing, used to reduce landmark jitter."""
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def exponential_moving_average(values: Sequence[float], alpha: float = 0.3) -> np.ndarray:
    """EMA smoothing — lower latency than a sliding window average,
    useful for real-time skeleton jitter reduction."""
    arr = np.asarray(values, dtype=np.float64)
    ema = np.zeros_like(arr)
    if len(arr) == 0:
        return ema
    ema[0] = arr[0]
    for i in range(1, len(arr)):
        ema[i] = alpha * arr[i] + (1 - alpha) * ema[i - 1]
    return ema
