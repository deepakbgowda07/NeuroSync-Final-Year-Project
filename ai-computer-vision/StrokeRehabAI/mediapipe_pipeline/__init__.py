"""
mediapipe_pipeline package
===========================
Wraps Google's MediaPipe Pose solution for real-time and offline body
landmark extraction. Named `mediapipe_pipeline` (rather than plain
`mediapipe`) specifically to avoid shadowing the third-party
`mediapipe` PyPI package within this project's import namespace.
"""

from .pose_estimator import PoseEstimator, PoseResult
from .landmark_extractor import LandmarkExtractor, LANDMARK_NAMES
from .pose_smoothing import LandmarkSmoother, PoseGapHandler
from .view_detector import ViewDetector, CameraView, ViewDetectionResult

__all__ = [
    "PoseEstimator",
    "PoseResult",
    "LandmarkExtractor",
    "LANDMARK_NAMES",
    "LandmarkSmoother",
    "PoseGapHandler",
    "ViewDetector",
    "CameraView",
    "ViewDetectionResult",
]
