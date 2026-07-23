"""
geometry.py
===========
General-purpose 2D/3D geometry helpers used by joint-angle calculation
and pose post-processing (vectors, distances, normalization).
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


Point3D = Tuple[float, float, float]


def to_np(point: Sequence[float]) -> np.ndarray:
    """Convert an (x, y[, z]) sequence to a numpy float array."""
    return np.asarray(point, dtype=np.float64)


def euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Euclidean distance between two points of equal dimensionality."""
    return float(np.linalg.norm(to_np(a) - to_np(b)))


def midpoint(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    """Midpoint between two points."""
    return (to_np(a) + to_np(b)) / 2.0


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Return a unit vector; returns a zero vector unchanged (avoids div/0)."""
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return v
    return v / norm


def vector_between(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    """Vector pointing from point a to point b."""
    return to_np(b) - to_np(a)


def landmarks_to_array(landmarks) -> np.ndarray:
    """Convert a MediaPipe-style landmark list into an (N, 3) numpy array.

    Accepts anything with `.x`, `.y`, `.z` attributes per item (i.e. the
    MediaPipe `NormalizedLandmarkList.landmark` iterable), or a plain
    iterable of (x, y, z) tuples.
    """
    points = []
    for lm in landmarks:
        if hasattr(lm, "x"):
            points.append((lm.x, lm.y, getattr(lm, "z", 0.0)))
        else:
            points.append(tuple(lm))
    return np.asarray(points, dtype=np.float64)


def bounding_box(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (min_xyz, max_xyz) bounding box for a set of 2D/3D points."""
    return points.min(axis=0), points.max(axis=0)
