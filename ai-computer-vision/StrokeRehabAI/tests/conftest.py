"""
conftest.py
===========
Shared pytest fixtures: a small synthetic landmark sequence used
across preprocessing/feature/model tests without needing a live
camera, MediaPipe, or a real dataset.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def synthetic_landmark_frame() -> np.ndarray:
    """A single (33, 3) frame of plausible normalized landmark coordinates."""
    rng = np.random.default_rng(42)
    return rng.uniform(0.0, 1.0, size=(33, 3))


@pytest.fixture
def synthetic_landmark_sequence() -> np.ndarray:
    """A (90, 33, 3) synthetic sequence, matching the default sequence_length."""
    rng = np.random.default_rng(42)
    return rng.uniform(0.0, 1.0, size=(90, 33, 3))
