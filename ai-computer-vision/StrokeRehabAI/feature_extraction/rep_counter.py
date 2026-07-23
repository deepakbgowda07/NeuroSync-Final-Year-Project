"""
rep_counter.py
================
Counts exercise repetitions from a joint-angle time series using peak
detection — a standard approach for cyclic rehab exercises (e.g.
repeated shoulder flexion, elbow flexion/extension).

Implemented with a dependency-free local-maxima/minima scan rather than
scipy.signal.find_peaks, since this project's requirements.txt does not
currently include scipy; if scipy is available, results should closely
match `scipy.signal.find_peaks` with an equivalent prominence threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from utils.math_utils import moving_average
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RepCountResult:
    rep_count: int
    peak_frame_indices: List[int]
    peak_values: List[float]


class RepCounter:
    """Counts repetitions in a smoothed 1D joint-angle signal via
    local-maxima detection with a minimum prominence and minimum
    inter-peak distance to reject noise-driven false peaks."""

    def __init__(self, smoothing_window: int = 5, min_prominence_deg: float = 10.0, min_distance_frames: int = 10):
        self.smoothing_window = smoothing_window
        self.min_prominence_deg = min_prominence_deg
        self.min_distance_frames = min_distance_frames

    def count(self, angle_sequence_deg: np.ndarray) -> RepCountResult:
        """Count repetitions in a (T,) joint-angle sequence."""
        if len(angle_sequence_deg) < self.smoothing_window + 2:
            return RepCountResult(rep_count=0, peak_frame_indices=[], peak_values=[])

        smoothed = moving_average(angle_sequence_deg, window=self.smoothing_window)
        peaks = self._find_peaks(smoothed)

        logger.debug("RepCounter found %d peaks (rep count).", len(peaks))
        return RepCountResult(
            rep_count=len(peaks),
            peak_frame_indices=peaks,
            peak_values=[float(smoothed[i]) for i in peaks],
        )

    def _find_peaks(self, signal: np.ndarray) -> List[int]:
        candidate_peaks = []
        for i in range(1, len(signal) - 1):
            if signal[i] > signal[i - 1] and signal[i] >= signal[i + 1]:
                candidate_peaks.append(i)

        # Filter by prominence: local peak must exceed nearby valleys by
        # at least min_prominence_deg.
        prominent_peaks = []
        for peak_idx in candidate_peaks:
            left_valley = signal[max(0, peak_idx - self.min_distance_frames):peak_idx]
            right_valley = signal[peak_idx:peak_idx + self.min_distance_frames]
            left_min = left_valley.min() if len(left_valley) else signal[peak_idx]
            right_min = right_valley.min() if len(right_valley) else signal[peak_idx]
            prominence = signal[peak_idx] - max(left_min, right_min)
            if prominence >= self.min_prominence_deg:
                prominent_peaks.append(peak_idx)

        # Enforce minimum distance between accepted peaks (greedy, by value).
        prominent_peaks.sort(key=lambda idx: signal[idx], reverse=True)
        accepted: List[int] = []
        for idx in prominent_peaks:
            if all(abs(idx - a) >= self.min_distance_frames for a in accepted):
                accepted.append(idx)

        return sorted(accepted)
