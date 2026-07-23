"""Camera package: webcam access, frame buffering, and video-file loading."""

from .camera_manager import CameraManager
from .webcam import Webcam
from .video_loader import VideoLoader
from .frame_queue import FrameQueue, TimestampedFrame
from .fps_controller import FPSController
from .resolution_manager import ResolutionManager, ResolutionResult

__all__ = [
    "CameraManager",
    "Webcam",
    "VideoLoader",
    "FrameQueue",
    "TimestampedFrame",
    "FPSController",
    "ResolutionManager",
    "ResolutionResult",
]
