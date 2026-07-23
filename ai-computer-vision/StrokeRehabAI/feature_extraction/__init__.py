"""Feature extraction package: derives higher-level clinical features
(joint angles, kinematics, posture, smoothness, rep counts) on top of
raw landmarks."""

from .angle_features import AngleFeatureExtractor
from .kinematic_features import KinematicFeatureExtractor
from .clinical_features import ClinicalFeatureExtractor
from .rep_counter import RepCounter, RepCountResult
from .feature_pipeline import FeatureExtractionPipeline

__all__ = [
    "AngleFeatureExtractor",
    "KinematicFeatureExtractor",
    "ClinicalFeatureExtractor",
    "RepCounter",
    "RepCountResult",
    "FeatureExtractionPipeline",
]
