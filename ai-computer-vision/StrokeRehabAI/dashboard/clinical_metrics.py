"""
clinical_metrics.py
======================
Computes the clinical scoring suite for one session (or aggregated
across sessions), all normalized to 0-100 for consistent clinician-
facing display:

    Shoulder Mobility Score      Compensation Score
    Elbow Mobility Score          Exercise Quality Score
    Movement Stability             Recovery Index
    Movement Smoothness             Overall Rehabilitation Score
    Symmetry Score

Built on top of the joint-angle and event data already stored per
session (`session_frames`, `session_events`, `session_reps`) — no new
data collection required, this is a scoring layer over what the
real-time inference engine (see `inference/`) already logs via
`inference/session_logger.py`.

TODO (next development phase): weight/blend coefficients below are
principled starting defaults, not clinician-validated — see the
similar TODOs in inference/exercise_library.py and
feature_extraction/clinical_features.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ClinicalScores:
    shoulder_mobility_score: float
    elbow_mobility_score: float
    movement_stability: float
    movement_smoothness: float
    symmetry_score: float
    compensation_score: float
    exercise_quality_score: float
    recovery_index: float
    overall_rehabilitation_score: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "shoulder_mobility_score": self.shoulder_mobility_score,
            "elbow_mobility_score": self.elbow_mobility_score,
            "movement_stability": self.movement_stability,
            "movement_smoothness": self.movement_smoothness,
            "symmetry_score": self.symmetry_score,
            "compensation_score": self.compensation_score,
            "exercise_quality_score": self.exercise_quality_score,
            "recovery_index": self.recovery_index,
            "overall_rehabilitation_score": self.overall_rehabilitation_score,
        }


def _clip_0_100(value: float) -> float:
    return float(np.clip(value, 0.0, 100.0))


class ClinicalMetricsCalculator:
    """Computes the 0-100 clinical score suite from one session's stored
    frames/events, or from an aggregate of several sessions' scores."""

    def __init__(
        self,
        shoulder_rom_reference_deg: float = 150.0,
        elbow_rom_reference_deg: float = 130.0,
    ):
        self.shoulder_rom_reference_deg = shoulder_rom_reference_deg
        self.elbow_rom_reference_deg = elbow_rom_reference_deg

    def compute_for_session(
        self,
        frames: List[Dict],
        events: List[Dict],
        reps: List[Dict],
    ) -> ClinicalScores:
        """`frames` are session_frames rows (with parsed joint_angles
        dicts), `events` are session_events rows, `reps` are session_reps
        rows — see dashboard/session_manager.py:get_session_detail."""
        shoulder_mobility = self._mobility_score(frames, ("left_shoulder_angle", "right_shoulder_angle"), self.shoulder_rom_reference_deg)
        elbow_mobility = self._mobility_score(frames, ("left_elbow_angle", "right_elbow_angle"), self.elbow_rom_reference_deg)
        stability = self._movement_stability(frames)
        smoothness = self._movement_smoothness(frames)
        symmetry = self._symmetry_score(frames)
        compensation = self._compensation_score(events, num_frames=max(1, len(frames)))
        quality = self._exercise_quality_score(frames, reps)

        # Recovery index: a session-level blend emphasizing ROM + quality,
        # since those most directly reflect functional recovery.
        recovery_index = _clip_0_100(0.35 * shoulder_mobility + 0.25 * elbow_mobility + 0.40 * quality)

        overall = _clip_0_100(
            0.15 * shoulder_mobility + 0.15 * elbow_mobility + 0.15 * stability +
            0.10 * smoothness + 0.10 * symmetry + 0.15 * compensation + 0.20 * quality
        )

        return ClinicalScores(
            shoulder_mobility_score=shoulder_mobility,
            elbow_mobility_score=elbow_mobility,
            movement_stability=stability,
            movement_smoothness=smoothness,
            symmetry_score=symmetry,
            compensation_score=compensation,
            exercise_quality_score=quality,
            recovery_index=recovery_index,
            overall_rehabilitation_score=overall,
        )

    # ------------------------------------------------------------------
    # Individual scores
    # ------------------------------------------------------------------

    def _mobility_score(self, frames: List[Dict], angle_keys: tuple, reference_rom_deg: float) -> float:
        """Ratio of observed ROM (max - min of the tracked angle) to a
        reference full-mobility ROM, scaled to 0-100."""
        values = self._collect_angle_values(frames, angle_keys)
        if not values:
            return 0.0
        observed_rom = max(values) - min(values)
        return _clip_0_100((observed_rom / reference_rom_deg) * 100.0)

    def _movement_stability(self, frames: List[Dict]) -> float:
        """Inverse of frame-to-frame movement-quality variance — a
        patient whose quality score bounces around erratically is judged
        less stable than one with a consistent (even if imperfect) score."""
        qualities = [f.get("movement_quality") for f in frames if f.get("movement_quality") is not None]
        if len(qualities) < 2:
            return 50.0  # insufficient data — neutral default, not zero
        std = float(np.std(qualities))
        return _clip_0_100(100.0 * (1.0 - min(std * 4.0, 1.0)))

    def _movement_smoothness(self, frames: List[Dict]) -> float:
        """Approximated from frame-to-frame jerk of the elbow angle (the
        most consistently tracked joint across exercises). Uses a single,
        temporally-ordered trajectory (preferring whichever side has more
        data) rather than combining both sides, since interleaving two
        independent angle sequences would corrupt the velocity/jerk
        calculation with spurious left/right differences."""
        values = self._collect_ordered_trajectory(frames, ("left_elbow_angle", "right_elbow_angle"))
        if len(values) < 4:
            return 50.0
        arr = np.array(values)
        velocity = np.diff(arr)
        jerk = np.diff(velocity)
        jerk_std = float(np.std(jerk)) if len(jerk) else 0.0
        # Lower jerk std = smoother; scale so a jerk_std of ~30deg/frame^2 -> near 0.
        return _clip_0_100(100.0 * max(0.0, 1.0 - jerk_std / 30.0))

    def _symmetry_score(self, frames: List[Dict]) -> float:
        left_values = self._collect_angle_values(frames, ("left_elbow_angle",))
        right_values = self._collect_angle_values(frames, ("right_elbow_angle",))
        if not left_values or not right_values:
            return 50.0

        left_rom = max(left_values) - min(left_values)
        right_rom = max(right_values) - min(right_values)
        total = left_rom + right_rom
        if total < 1e-6:
            return 100.0
        asymmetry = abs(left_rom - right_rom) / total
        return _clip_0_100(100.0 * (1.0 - asymmetry))

    def _compensation_score(self, events: List[Dict], num_frames: int) -> float:
        """100 = no compensation detected; decreases with compensation
        event frequency (events per 100 frames)."""
        compensation_types = {"trunk_compensation", "shoulder_hiking", "body_lean", "poor_alignment"}
        compensation_events = [e for e in events if e.get("error_type") in compensation_types]
        rate_per_100_frames = (len(compensation_events) / num_frames) * 100.0
        return _clip_0_100(100.0 - rate_per_100_frames * 8.0)

    def _exercise_quality_score(self, frames: List[Dict], reps: List[Dict]) -> float:
        qualities = [f.get("movement_quality") for f in frames if f.get("movement_quality") is not None]
        mean_quality = float(np.mean(qualities)) * 100.0 if qualities else 50.0

        if reps:
            completion_rate = sum(1 for r in reps if r.get("completed")) / len(reps) * 100.0
        else:
            completion_rate = 100.0  # no rep data — don't penalize

        return _clip_0_100(0.6 * mean_quality + 0.4 * completion_rate)

    @staticmethod
    def _collect_angle_values(frames: List[Dict], angle_keys: tuple) -> List[float]:
        values = []
        for frame in frames:
            angles = frame.get("joint_angles") or {}
            for key in angle_keys:
                value = angles.get(key)
                if value is not None and not (isinstance(value, float) and np.isnan(value)):
                    values.append(value)
        return values

    @staticmethod
    def _collect_ordered_trajectory(frames: List[Dict], angle_keys: tuple) -> List[float]:
        """Like _collect_angle_values, but returns one value per frame
        (preferring the first key in `angle_keys` that has data for that
        frame) rather than concatenating every key's values — preserving
        correct temporal order for velocity/jerk-style calculations."""
        values = []
        for frame in frames:
            angles = frame.get("joint_angles") or {}
            for key in angle_keys:
                value = angles.get(key)
                if value is not None and not (isinstance(value, float) and np.isnan(value)):
                    values.append(value)
                    break
        return values

    # ------------------------------------------------------------------
    # Aggregation across sessions
    # ------------------------------------------------------------------

    @staticmethod
    def average_scores(score_dicts: List[Dict[str, float]]) -> Dict[str, float]:
        """Average a list of per-session score dicts (e.g. for a
        patient-level summary) into one dict of the same keys."""
        if not score_dicts:
            return {}
        keys = score_dicts[0].keys()
        return {key: float(np.mean([s[key] for s in score_dicts if s.get(key) is not None])) for key in keys}
