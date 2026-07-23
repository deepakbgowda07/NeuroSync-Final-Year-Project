"""
camera_manager.py
==================
High-level orchestration on top of `Webcam` / `VideoLoader`: webcam
detection, config-driven initialization, resolution negotiation,
low-latency background capture (via a bounded FrameQueue), and
continuous automatic recovery if the camera drops out mid-session
(distinct from the initial connect() retry, which only covers startup).
"""

from __future__ import annotations

import threading
import time
from typing import Callable, List, Optional

from camera.camera import FrameSource
from camera.camera_utils import detect_webcam_capabilities, list_available_camera_indices
from camera.frame_queue import FrameQueue, TimestampedFrame
from camera.resolution_manager import ResolutionManager
from camera.video_loader import VideoLoader
from camera.webcam import Webcam
from utils.logger import get_logger

logger = get_logger(__name__)


class CameraManager:
    """Factory + retry wrapper around concrete FrameSource implementations,
    with an optional low-latency background-thread streaming mode."""

    def __init__(
        self,
        source=0,
        width: int = 1280,
        height: int = 720,
        fps_target: int = 30,
        fps_min: int = 15,
        backend: str = "DSHOW",
        flip_horizontal: bool = True,
        buffer_size: int = 1,
        retry_attempts: int = 5,
        retry_delay_seconds: float = 1.0,
        auto_recovery_enabled: bool = True,
        max_consecutive_failures: int = 10,
        reconnect_backoff_seconds: float = 0.5,
        frame_queue_enabled: bool = True,
        frame_queue_max_size: int = 2,
        frame_queue_drop_policy: str = "oldest",
    ):
        self.source = source
        self.width = width
        self.height = height
        self.fps_target = fps_target
        self.fps_min = fps_min
        self.backend = backend
        self.flip_horizontal = flip_horizontal
        self.buffer_size = buffer_size
        self.retry_attempts = retry_attempts
        self.retry_delay_seconds = retry_delay_seconds

        self.auto_recovery_enabled = auto_recovery_enabled
        self.max_consecutive_failures = max_consecutive_failures
        self.reconnect_backoff_seconds = reconnect_backoff_seconds

        self.frame_queue_enabled = frame_queue_enabled
        self.frame_queue = FrameQueue(max_size=frame_queue_max_size, drop_policy=frame_queue_drop_policy) if frame_queue_enabled else None

        self._frame_source: Optional[FrameSource] = None
        self._resolution_manager = ResolutionManager()

        self._capture_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._consecutive_failures = 0

    @classmethod
    def from_config(cls, camera_cfg) -> "CameraManager":
        """Build a CameraManager directly from the `camera` section of the
        merged application config (see configs/camera.yaml)."""
        auto_recovery = camera_cfg.auto_recovery if hasattr(camera_cfg, "auto_recovery") else {}
        frame_queue_cfg = camera_cfg.frame_queue if hasattr(camera_cfg, "frame_queue") else {}

        return cls(
            source=camera_cfg.source,
            width=camera_cfg.width,
            height=camera_cfg.height,
            fps_target=camera_cfg.fps_target,
            fps_min=camera_cfg.get("fps_min", 15) if hasattr(camera_cfg, "get") else 15,
            backend=camera_cfg.backend,
            flip_horizontal=camera_cfg.flip_horizontal,
            buffer_size=camera_cfg.buffer_size,
            retry_attempts=camera_cfg.retry_attempts,
            retry_delay_seconds=camera_cfg.retry_delay_seconds,
            auto_recovery_enabled=auto_recovery.get("enabled", True) if auto_recovery else True,
            max_consecutive_failures=auto_recovery.get("max_consecutive_failures", 10) if auto_recovery else 10,
            reconnect_backoff_seconds=auto_recovery.get("reconnect_backoff_seconds", 0.5) if auto_recovery else 0.5,
            frame_queue_enabled=frame_queue_cfg.get("enabled", True) if frame_queue_cfg else True,
            frame_queue_max_size=frame_queue_cfg.get("max_queue_size", 2) if frame_queue_cfg else 2,
            frame_queue_drop_policy=frame_queue_cfg.get("drop_policy", "oldest") if frame_queue_cfg else "oldest",
        )

    # ------------------------------------------------------------------
    # Webcam detection
    # ------------------------------------------------------------------

    @staticmethod
    def detect_webcams(max_index: int = 5) -> List[dict]:
        """Probe camera indices and return capability info for each
        available one (resolution, FPS, supported 720p/1080p presets) —
        a first-run diagnostic, and the basis for the "webcam detection"
        requirement (the system never asks the patient to configure a
        camera by hand)."""
        results = []
        for idx in list_available_camera_indices(max_index):
            results.append(detect_webcam_capabilities(idx))
        logger.info("Detected %d available webcam(s): %s", len(results), [r["index"] for r in results])
        return results

    # ------------------------------------------------------------------
    # Simple (non-threaded) connect/read — used by tests and the offline paths
    # ------------------------------------------------------------------

    def _build_source(self) -> FrameSource:
        if isinstance(self.source, str) and self.source.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            return VideoLoader(self.source)
        return Webcam(
            source=self.source,
            width=self.width,
            height=self.height,
            fps_target=self.fps_target,
            backend=self.backend,
            flip_horizontal=self.flip_horizontal,
            buffer_size=self.buffer_size,
        )

    def connect(self) -> FrameSource:
        """Open a frame source, retrying on transient failure (e.g. the
        webcam being briefly held by another process)."""
        last_error: Optional[Exception] = None
        for attempt in range(1, self.retry_attempts + 1):
            try:
                self._frame_source = self._build_source()
                self._frame_source.open()
                logger.info("Camera connected on attempt %d/%d.", attempt, self.retry_attempts)
                return self._frame_source
            except Exception as exc:  # noqa: BLE001 - intentionally broad for hardware retry
                last_error = exc
                logger.warning("Camera connect attempt %d/%d failed: %s", attempt, self.retry_attempts, exc)
                time.sleep(self.retry_delay_seconds)

        raise RuntimeError(f"Failed to connect camera after {self.retry_attempts} attempts.") from last_error

    def release(self) -> None:
        self.stop_streaming()
        if self._frame_source is not None:
            self._frame_source.release()
            self._frame_source = None

    # ------------------------------------------------------------------
    # Low-latency background-thread streaming
    # ------------------------------------------------------------------

    def start_streaming(self) -> None:
        """Connect (if needed) and launch a background thread that
        continuously reads frames into `self.frame_queue`, with
        automatic reconnect on repeated read failures. This is the
        recommended mode for the real-time inference pipeline: it keeps
        camera I/O off the pose-estimation/analysis thread, and the
        consumer always sees the newest available frame.
        """
        if self._frame_source is None:
            self.connect()

        if self.frame_queue is None:
            self.frame_queue = FrameQueue(max_size=2, drop_policy="oldest")

        self._stop_event.clear()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True, name="camera-capture")
        self._capture_thread.start()
        logger.info("Camera streaming started (background capture thread).")

    def stop_streaming(self) -> None:
        if self._capture_thread is not None:
            self._stop_event.set()
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None
            logger.info("Camera streaming stopped.")

    def read_latest_frame(self, timeout: float = 1.0) -> Optional[TimestampedFrame]:
        """Consumer-side read from the background capture thread's queue."""
        if self.frame_queue is None:
            raise RuntimeError("start_streaming() must be called before read_latest_frame().")
        return self.frame_queue.get(timeout=timeout)

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                ok, frame = self._frame_source.read()
            except Exception as exc:  # noqa: BLE001 - hardware I/O can fail in many ways
                ok, frame = False, None
                logger.error("Camera read raised an exception: %s", exc)

            if ok and frame is not None:
                self._consecutive_failures = 0
                self.frame_queue.put(frame)
            else:
                self._consecutive_failures += 1
                logger.warning("Camera read failed (%d/%d consecutive).", self._consecutive_failures, self.max_consecutive_failures)

                if self.auto_recovery_enabled and self._consecutive_failures >= self.max_consecutive_failures:
                    self._attempt_recovery()

    def _attempt_recovery(self) -> None:
        logger.error("Too many consecutive camera failures — attempting automatic reconnect.")
        try:
            if self._frame_source is not None:
                self._frame_source.release()
        except Exception:  # noqa: BLE001
            pass

        time.sleep(self.reconnect_backoff_seconds)
        try:
            self.connect()
            self._consecutive_failures = 0
            logger.info("Camera automatically recovered.")
        except RuntimeError as exc:
            logger.error("Automatic camera recovery failed: %s", exc)
            time.sleep(self.reconnect_backoff_seconds)
