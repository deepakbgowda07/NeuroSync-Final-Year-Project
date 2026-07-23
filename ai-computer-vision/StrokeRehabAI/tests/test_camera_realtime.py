"""Tests for camera.frame_queue, camera.fps_controller, camera.resolution_manager."""

import time

import numpy as np
import pytest

from camera.frame_queue import FrameQueue
from camera.fps_controller import FPSController
from camera.resolution_manager import ResolutionManager


def test_frame_queue_drops_oldest_when_full():
    fq = FrameQueue(max_size=2, drop_policy="oldest")
    for i in range(5):
        fq.put(np.full((2, 2, 3), i, dtype=np.uint8))
    assert fq.qsize() == 2
    assert fq.dropped_frames == 3
    newest = fq.get(timeout=0.1)
    assert newest.frame_index == 3  # frame 3 (second-to-last enqueued) should remain


def test_frame_queue_drops_newest_when_policy_is_newest():
    fq = FrameQueue(max_size=1, drop_policy="newest")
    fq.put(np.zeros((2, 2, 3), dtype=np.uint8))
    fq.put(np.ones((2, 2, 3), dtype=np.uint8))  # should be dropped
    assert fq.qsize() == 1
    kept = fq.get(timeout=0.1)
    assert kept.frame_index == 0


def test_frame_queue_get_timeout_returns_none():
    fq = FrameQueue(max_size=2)
    assert fq.get(timeout=0.05) is None


def test_frame_queue_get_latest_drains_backlog():
    fq = FrameQueue(max_size=5)
    for i in range(3):
        fq.put(np.full((2, 2, 3), i, dtype=np.uint8))
    latest = fq.get_latest()
    assert latest.frame_index == 2
    assert fq.qsize() == 0


def test_fps_controller_measures_reasonable_fps():
    controller = FPSController(target_fps=30, min_fps=15, adjust_every_n_frames=100)
    for _ in range(5):
        controller.frame_start()
        time.sleep(0.01)
        controller.frame_end()
    assert 50 < controller.measured_fps < 200  # ~1/0.01s = 100fps, loose bounds


def test_fps_controller_steps_down_when_slow():
    controller = FPSController(target_fps=30, min_fps=10, adjust_every_n_frames=3)
    for _ in range(3):
        controller.frame_start()
        time.sleep(0.1)  # much slower than 30fps
        controller.frame_end()
    assert controller.current_fps_cap < 30


def test_resolution_manager_resolves_known_presets():
    manager = ResolutionManager()
    assert manager.resolve("720p") == (1280, 720)
    assert manager.resolve("1080p") == (1920, 1080)


def test_resolution_manager_unknown_preset_raises():
    manager = ResolutionManager()
    with pytest.raises(ValueError):
        manager.resolve("4k")


def test_resolution_manager_validate_actual_detects_mismatch():
    manager = ResolutionManager()
    result = manager.validate_actual("720p", 640, 480)
    assert not result.matched_preset
    assert result.actual_size == (640, 480)
