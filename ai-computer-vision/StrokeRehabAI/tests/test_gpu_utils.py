"""Tests for utils.gpu_utils."""

from utils.gpu_utils import estimate_safe_batch_size


def test_estimate_safe_batch_size_low_vram():
    assert estimate_safe_batch_size(4) == 8


def test_estimate_safe_batch_size_rtx_3050():
    assert estimate_safe_batch_size(6) == 16


def test_estimate_safe_batch_size_none_returns_default():
    assert estimate_safe_batch_size(None, default=12) == 12
