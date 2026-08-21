"""Tests for inference.async_inference.AsyncInferenceWorker."""

import time

import numpy as np
import pytest

from inference.async_inference import AsyncInferenceWorker


@pytest.fixture
def worker():
    def predict(window):
        return {"sum": float(window.sum())}
    w = AsyncInferenceWorker(predict, max_queue_size=2)
    w.start()
    yield w
    w.stop()


def test_submit_and_get_result(worker):
    worker.submit(np.ones((3, 3)))
    result = worker.get_latest_result(timeout=2.0)
    assert result is not None
    assert result.result["sum"] == 9.0


def test_get_latest_result_returns_newest_when_backlog(worker):
    for i in range(5):
        worker.submit(np.full((2, 2), i))
        time.sleep(0.01)
    time.sleep(0.2)
    result = worker.get_latest_result(timeout=1.0)
    assert result is not None  # should get the most recent, not error


def test_get_latest_result_timeout_returns_none():
    def slow_predict(window):
        time.sleep(1.0)
        return {}
    worker = AsyncInferenceWorker(slow_predict)
    worker.start()
    result = worker.get_latest_result(timeout=0.05)
    assert result is None
    worker.stop()


def test_worker_survives_prediction_exceptions():
    def flaky_predict(window):
        raise ValueError("simulated failure")

    worker = AsyncInferenceWorker(flaky_predict)
    worker.start()
    worker.submit(np.ones((2, 2)))
    time.sleep(0.2)
    # Worker thread should still be alive and accept new jobs without crashing.
    worker.submit(np.ones((2, 2)))
    time.sleep(0.1)
    worker.stop()  # should not hang or raise


def test_dropped_jobs_counted_when_queue_full():
    def slow_predict(window):
        time.sleep(0.3)
        return {}
    worker = AsyncInferenceWorker(slow_predict, max_queue_size=1)
    worker.start()
    for i in range(5):
        worker.submit(np.full((2, 2), i))
    time.sleep(0.05)
    worker.stop()
    assert worker.dropped_jobs > 0
