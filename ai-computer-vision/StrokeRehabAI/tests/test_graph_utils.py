"""Tests for models.graph_utils (ST-GCN skeleton graph construction)."""

import numpy as np
import pytest

from models.graph_utils import Graph, NUM_MEDIAPIPE_JOINTS


@pytest.mark.parametrize("strategy,expected_subsets", [("uniform", 1), ("distance", 2), ("spatial", 3)])
def test_graph_subset_count_matches_strategy(strategy, expected_subsets):
    graph = Graph(strategy=strategy)
    assert graph.A.shape == (expected_subsets, NUM_MEDIAPIPE_JOINTS, NUM_MEDIAPIPE_JOINTS)


def test_graph_adjacency_has_no_nan_or_inf():
    graph = Graph(strategy="spatial")
    assert np.all(np.isfinite(graph.A))


def test_graph_adjacency_is_nonnegative():
    graph = Graph(strategy="spatial")
    assert np.all(graph.A >= 0)


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        Graph(strategy="not_a_real_strategy")
