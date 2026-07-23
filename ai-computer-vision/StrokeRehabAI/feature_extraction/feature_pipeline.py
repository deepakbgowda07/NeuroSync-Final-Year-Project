"""
feature_pipeline.py
====================
Combines all feature extractors (joint angles, kinematics, clinical
descriptors, repetition counting) into a single feature dict per
sample — used both for offline dataset feature generation (saved
alongside converted .npz samples) and evaluation reporting.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from feature_extraction.angle_features import AngleFeatureExtractor
from feature_extraction.clinical_features import ClinicalFeatureExtractor
from feature_extraction.kinematic_features import KinematicFeatureExtractor
from feature_extraction.rep_counter import RepCounter
from utils.joint_angles import JOINT_ANGLE_DEFINITIONS, MEDIAPIPE_LANDMARK_INDEX
from utils.logger import get_logger

logger = get_logger(__name__)


class FeatureExtractionPipeline:
    """Runs the full clinical feature-extraction suite over a sequence."""

    def __init__(self, fps: float = 30.0, rep_counting_angle: str = "left_elbow_angle"):
        self.angle_extractor = AngleFeatureExtractor()
        self.kinematic_extractor = KinematicFeatureExtractor(fps=fps)
        self.clinical_extractor = ClinicalFeatureExtractor(fps=fps)
        self.rep_counter = RepCounter()
        self.rep_counting_angle = rep_counting_angle

    def run(self, sequence_xyz: np.ndarray) -> Dict:
        """Returns a report-ready dict of every implemented feature group.

        `sequence_xyz` is a (T, 33, 3) landmark sequence (post-smoothing,
        pre- or post-normalization — angle/ratio-based features are scale
        invariant either way; absolute-distance features assume the raw,
        un-normalized coordinate scale).
        """
        angle_array = self.angle_extractor.extract_sequence(sequence_xyz)
        angle_names = self.angle_extractor.angle_names

        features: Dict = {
            "joint_angle_summary": self.angle_extractor.summary_statistics(sequence_xyz),
            "mean_joint_speed": self.kinematic_extractor.mean_speed_per_joint(sequence_xyz).tolist(),
            "trunk_lean": {
                "mean_deg": float(np.mean(self.clinical_extractor.trunk_lean_deg(sequence_xyz))),
                "max_deg": float(np.max(self.clinical_extractor.trunk_lean_deg(sequence_xyz))),
            },
            "shoulder_elevation": {
                "left_mean": float(np.mean(self.clinical_extractor.shoulder_elevation(sequence_xyz, "left"))),
                "right_mean": float(np.mean(self.clinical_extractor.shoulder_elevation(sequence_xyz, "right"))),
            },
            "arm_extension": {
                "left_max": float(np.max(self.clinical_extractor.arm_extension(sequence_xyz, "left"))),
                "right_max": float(np.max(self.clinical_extractor.arm_extension(sequence_xyz, "right"))),
            },
            "movement_smoothness": {
                "left_wrist": self.clinical_extractor.movement_smoothness(
                    sequence_xyz, MEDIAPIPE_LANDMARK_INDEX["left_wrist"]
                ),
                "right_wrist": self.clinical_extractor.movement_smoothness(
                    sequence_xyz, MEDIAPIPE_LANDMARK_INDEX["right_wrist"]
                ),
            },
            "exercise_duration_seconds": self.clinical_extractor.exercise_duration_seconds(sequence_xyz),
            "compensation_indicators": self.clinical_extractor.compensation_indicators(sequence_xyz),
            "bilateral_symmetry_index": self.kinematic_extractor.bilateral_symmetry_index(
                sequence_xyz,
                MEDIAPIPE_LANDMARK_INDEX["left_wrist"],
                MEDIAPIPE_LANDMARK_INDEX["right_wrist"],
            ),
        }

        if self.rep_counting_angle in angle_names:
            angle_idx = angle_names.index(self.rep_counting_angle)
            rep_result = self.rep_counter.count(angle_array[:, angle_idx])
            features["rep_count"] = rep_result.rep_count

        return features
