"""
async_inference.py
=====================
Runs model inference on a background thread, decoupled from the main
capture/analysis loop — while GPU inference for one window is running,
the main thread continues reading and buffering the next frames rather
than blocking, improving achieved FPS on modest hardware (RTX 3050,
6GB VRAM) toward the 30+ FPS / 720p target (see docs/inference_guide.md).

This complements (not replaces) `camera/frame_queue.py`'s low-latency
frame queue: that decouples *capture* from *processing*; this decouples
*model inference* from the rest of *processing* (pose estimation,
feature extraction, movement analysis), which is the more expensive
step when a trained model checkpoint is in use.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class InferenceJob:
    job_id: int
    window: np.ndarray
    submitted_at: float


@dataclass
class InferenceResult:
    job_id: int
    result: dict
    latency_ms: float


class AsyncInferenceWorker:
    """Background-thread inference worker with bounded input/output
    queues. Drops stale jobs if the queue backs up (mirrors
    camera/frame_queue.py's "always process the newest" policy), so
    inference latency never compounds into ever-growing lag.
    """

    def __init__(self, predict_fn: Callable[[np.ndarray], dict], max_queue_size: int = 2):
        self.predict_fn = predict_fn
        self.max_queue_size = max_queue_size

        self._input_queue: "queue.Queue[InferenceJob]" = queue.Queue(maxsize=max_queue_size)
        self._output_queue: "queue.Queue[InferenceResult]" = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._job_counter = 0
        self._dropped_jobs = 0

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="async-inference")
        self._thread.start()
        logger.info("AsyncInferenceWorker started.")

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        logger.info("AsyncInferenceWorker stopped (%d jobs dropped due to backlog).", self._dropped_jobs)

    def submit(self, window: np.ndarray) -> int:
        """Submit a window for inference; returns a job_id. If the queue
        is full, the oldest pending job is dropped in favor of the new
        one (matches the real-time pipeline's "always use the freshest
        data" design)."""
        self._job_counter += 1
        job = InferenceJob(job_id=self._job_counter, window=window, submitted_at=time.perf_counter())

        if self._input_queue.full():
            try:
                self._input_queue.get_nowait()
                self._dropped_jobs += 1
            except queue.Empty:
                pass

        try:
            self._input_queue.put_nowait(job)
        except queue.Full:
            self._dropped_jobs += 1

        return job.job_id

    def get_latest_result(self, timeout: Optional[float] = None) -> Optional[InferenceResult]:
        """Drain the output queue and return only the newest completed
        result, discarding any stale ones — the caller only ever wants
        the most current prediction."""
        latest = None
        try:
            latest = self._output_queue.get(timeout=timeout)
        except queue.Empty:
            return None

        while True:
            try:
                latest = self._output_queue.get_nowait()
            except queue.Empty:
                break
        return latest

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = self._input_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            start = time.perf_counter()
            try:
                prediction = self.predict_fn(job.window)
            except Exception as exc:  # noqa: BLE001 - keep the worker alive across transient inference errors
                logger.error("Async inference job %d failed: %s", job.job_id, exc)
                continue
            latency_ms = (time.perf_counter() - start) * 1000

            result = InferenceResult(job_id=job.job_id, result=prediction, latency_ms=latency_ms)

            if self._output_queue.full():
                try:
                    self._output_queue.get_nowait()
                except queue.Empty:
                    pass
            try:
                self._output_queue.put_nowait(result)
            except queue.Full:
                pass

    @property
    def dropped_jobs(self) -> int:
        return self._dropped_jobs
