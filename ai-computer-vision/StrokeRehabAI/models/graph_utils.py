"""
graph_utils.py
===============
Skeleton graph construction for the ST-GCN model, built over MediaPipe
Pose's 33-joint layout (see mediapipe_pipeline/landmark_extractor.py's
LANDMARK_NAMES for the canonical index order).

Implements the three partitioning strategies from the original ST-GCN
paper (Yan, Xiong & Lin, 2018 — "Spatial Temporal Graph Convolutional
Networks for Skeleton-Based Action Recognition"):

    - "uniform":  a single adjacency subset (all neighbors treated equally)
    - "distance": two subsets (self + 1-hop neighbors)
    - "spatial":  three subsets (root / centripetal / centrifugal),
                  partitioned relative to the skeleton's gravity center —
                  this is the strategy the ST-GCN paper found most
                  effective and is this project's default.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)

# MediaPipe Pose has 33 landmarks (see mediapipe_pipeline/landmark_extractor.py::LANDMARK_NAMES).
NUM_MEDIAPIPE_JOINTS = 33

# Undirected bone connections across the full 33-point MediaPipe topology
# (face + torso + limbs), matching mediapipe.solutions.pose.POSE_CONNECTIONS.
MEDIAPIPE_POSE_EDGES: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]

# Skeleton "center" used for spatial-configuration partitioning — the
# midpoint between the hips is the natural gravity center for a standing
# or seated rehab exercise.
CENTER_JOINTS: Tuple[int, int] = (23, 24)  # left_hip, right_hip


def _build_hop_distance_matrix(num_node: int, edges: List[Tuple[int, int]], max_hop: int = 1) -> np.ndarray:
    """Compute shortest-path hop distance between every pair of joints
    (up to max_hop; farther pairs are marked as infinite/unreachable)."""
    adjacency = np.zeros((num_node, num_node))
    for i, j in edges:
        adjacency[i, j] = 1
        adjacency[j, i] = 1

    hop_dist = np.full((num_node, num_node), np.inf)
    transfer_mats = [np.linalg.matrix_power(adjacency, d) for d in range(max_hop + 1)]
    arrive_mat = np.stack(transfer_mats) > 0
    for d in range(max_hop, -1, -1):
        hop_dist[arrive_mat[d]] = d
    return hop_dist


def _normalize_adjacency(adjacency: np.ndarray) -> np.ndarray:
    """Symmetric-ish degree normalization: A_norm = D^-1 * A (row-normalized),
    matching the original ST-GCN implementation's `normalize_digraph`."""
    degree = np.sum(adjacency, axis=0)
    degree_inv = np.zeros_like(degree)
    nonzero = degree > 0
    degree_inv[nonzero] = degree[nonzero] ** -1
    degree_matrix = np.diag(degree_inv)
    return adjacency @ degree_matrix


class Graph:
    """Builds the (num_subsets, V, V) adjacency tensor `A` consumed by
    every st_gcn spatial-graph-convolution block."""

    def __init__(
        self,
        num_node: int = NUM_MEDIAPIPE_JOINTS,
        edges: List[Tuple[int, int]] = None,
        strategy: str = "spatial",
        max_hop: int = 1,
        center: Tuple[int, int] = CENTER_JOINTS,
    ):
        self.num_node = num_node
        self.edges = edges if edges is not None else MEDIAPIPE_POSE_EDGES
        self.strategy = strategy
        self.max_hop = max_hop
        self.center = center

        self.hop_dist = _build_hop_distance_matrix(self.num_node, self.edges, max_hop)
        self.A = self._build_adjacency()
        logger.debug(
            "Graph built: strategy=%s, num_node=%d, num_subsets=%d",
            self.strategy, self.num_node, self.A.shape[0],
        )

    def _build_adjacency(self) -> np.ndarray:
        valid_hop = list(range(self.max_hop + 1))
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dist == hop] = 1
        normalized = _normalize_adjacency(adjacency)

        if self.strategy == "uniform":
            return normalized[np.newaxis, ...]

        if self.strategy == "distance":
            subsets = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                subsets[i][self.hop_dist == hop] = normalized[self.hop_dist == hop]
            return subsets

        if self.strategy == "spatial":
            return self._spatial_partition(normalized, valid_hop)

        raise ValueError(f"Unknown graph strategy: {self.strategy}")

    def _spatial_partition(self, normalized: np.ndarray, valid_hop: List[int]) -> np.ndarray:
        """Three-subset spatial-configuration partitioning: for each edge,
        classify the neighbor as the root joint itself, closer to the
        skeleton center (centripetal), or farther from it (centrifugal)."""
        center_hop_dist = self.hop_dist[list(self.center), :].min(axis=0)

        root_subset = np.zeros((self.num_node, self.num_node))
        centripetal_subset = np.zeros((self.num_node, self.num_node))
        centrifugal_subset = np.zeros((self.num_node, self.num_node))

        for i in range(self.num_node):
            for j in range(self.num_node):
                if self.hop_dist[j, i] not in valid_hop:
                    continue
                if center_hop_dist[j] == center_hop_dist[i]:
                    root_subset[j, i] = normalized[j, i]
                elif center_hop_dist[j] > center_hop_dist[i]:
                    centripetal_subset[j, i] = normalized[j, i]
                else:
                    centrifugal_subset[j, i] = normalized[j, i]

        return np.stack([root_subset, centripetal_subset, centrifugal_subset])
