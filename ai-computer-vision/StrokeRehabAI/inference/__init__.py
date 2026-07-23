"""Inference package: real-time camera -> pose -> view detection ->
features -> movement analysis -> feedback -> dashboard pipeline."""

from .predictor import RehabPredictor
from .feedback_engine import FeedbackEngine, FeedbackMessage
from .realtime_pipeline import RealtimeInferencePipeline
from .exercise_library import ExerciseLibrary, ExerciseDefinition, SUPPORTED_EXERCISES
from .exercise_recognizer import ExerciseRecognizer, RecognitionResult
from .phase_detector import PhaseDetector, ExercisePhase, PhaseResult
from .rep_tracker import RepTracker, RepEvent
from .error_detector import ErrorDetector, ErrorType, DetectedError
from .calibration import CalibrationSession, CalibrationProfile
from .movement_analyzer import MovementAnalyzer, MovementAnalysisResult
from .smoothing import ConfidenceSmoother, AngleSmoother, PredictionSmoother
from .session_logger import SessionLogger

__all__ = [
    "RehabPredictor",
    "FeedbackEngine",
    "FeedbackMessage",
    "RealtimeInferencePipeline",
    "ExerciseLibrary",
    "ExerciseDefinition",
    "SUPPORTED_EXERCISES",
    "ExerciseRecognizer",
    "RecognitionResult",
    "PhaseDetector",
    "ExercisePhase",
    "PhaseResult",
    "RepTracker",
    "RepEvent",
    "ErrorDetector",
    "ErrorType",
    "DetectedError",
    "CalibrationSession",
    "CalibrationProfile",
    "MovementAnalyzer",
    "MovementAnalysisResult",
    "ConfidenceSmoother",
    "AngleSmoother",
    "PredictionSmoother",
    "SessionLogger",
]
