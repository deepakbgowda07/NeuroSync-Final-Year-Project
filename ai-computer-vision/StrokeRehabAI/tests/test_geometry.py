"""Tests for utils.geometry."""

import numpy as np

from utils.geometry import euclidean_distance, midpoint, normalize_vector


def test_euclidean_distance_simple():
    assert euclidean_distance((0, 0, 0), (3, 4, 0)) == 5.0


def test_midpoint():
    result = midpoint((0, 0), (2, 2))
    np.testing.assert_allclose(result, [1, 1])


def test_normalize_vector_unit_length():
    v = np.array([3.0, 4.0])
    unit = normalize_vector(v)
    assert np.isclose(np.linalg.norm(unit), 1.0)


def test_normalize_vector_zero_safe():
    v = np.zeros(3)
    result = normalize_vector(v)
    np.testing.assert_array_equal(result, v)
