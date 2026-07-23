"""
sequence_builder.py
====================
Assembles per-frame landmark features into fixed-length sliding-window
sequences for the LSTM/GRU model input, handling gap-filling for
frames with no detected pose.

TODO (next development phase): add configurable interpolation
strategies (linear vs. hold-last) for missing-pose gaps, and support
variable-length sequences via padding + attention masks for
transformer-based architectures.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


class SequenceBuilder:
    """Maintains a rolling buffer of per-frame feature vectors and emits
    fixed-length windows for model inference or training sample creation."""

    def __init__(self, sequence_length: int = 90, feature_dim: int = 99, stride: int = 15):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.stride = stride
        self._buffer: Deque[np.ndarray] = deque(maxlen=sequence_length)
        self._frames_since_last_window = 0

    def reset(self) -> None:
        self._buffer.clear()
        self._frames_since_last_window = 0

    def add_frame(self, feature_vector: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Add one frame's feature vector (or None on a missed detection).

        Returns a (sequence_length, feature_dim) window once the buffer
        is full and the stride condition is met, otherwise None.
        """
        if feature_vector is None:
            feature_vector = self._buffer[-1].copy() if self._buffer else np.zeros(self.feature_dim)
            logger.debug("No pose detected this frame; holding last known pose.")

        self._buffer.append(feature_vector)
        self._frames_since_last_window += 1

        is_full = len(self._buffer) == self.sequence_length
        stride_reached = self._frames_since_last_window >= self.stride

        if is_full and stride_reached:
            self._frames_since_last_window = 0
            return np.stack(list(self._buffer))
        return None

    @staticmethod
    def windows_from_full_sequence(
        full_sequence: np.ndarray, sequence_length: int, stride: int
    ) -> List[np.ndarray]:
        """Offline helper: slice a full (N, feature_dim) sequence (e.g. from
        a processed dataset video) into overlapping fixed-length windows."""
        windows = []
        for start in range(0, max(1, len(full_sequence) - sequence_length + 1), stride):
            window = full_sequence[start:start + sequence_length]
            if len(window) == sequence_length:
                windows.append(window)
        return windows

    @staticmethod
    def windows_from_full_sequence_graph(
        full_sequence_xyz: np.ndarray, sequence_length: int, stride: int
    ) -> List[np.ndarray]:
        """Same sliding-window slicing as windows_from_full_sequence, but
        preserves the (T, V, C) joint/channel shape instead of flattening
        — used for the ST-GCN graph-based data representation."""
        windows = []
        for start in range(0, max(1, len(full_sequence_xyz) - sequence_length + 1), stride):
            window = full_sequence_xyz[start:start + sequence_length]
            if len(window) == sequence_length:
                windows.append(window)
        return windows
