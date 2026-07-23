"""
camera_utils.py
================
Small stateless helpers for frame preprocessing that are specific to
raw camera output (as opposed to model-input preprocessing, which
lives in the `preprocessing` package).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def resize_frame(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize a BGR frame to (width, height), preserving aspect via letterbox padding."""
    import cv2

    target_w, target_h = target_size
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))

    canvas = np.zeros((target_h, target_w, 3), dtype=frame.dtype)
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def list_available_camera_indices(max_index: int = 5) -> list:
    """Probe camera indices 0..max_index and return the ones that open
    successfully. Useful for a first-run setup / diagnostics script.
    """
    import cv2

    available = []
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx)
        if cap is not None and cap.isOpened():
            available.append(idx)
        cap.release()
    return available


def detect_webcam_capabilities(index: int) -> dict:
    """Probe a single camera index for its default resolution/FPS and
    whether it accepts the 720p/1080p presets — used by the camera
    module's webcam-detection step (see camera/camera_manager.py) to
    report what's actually available before a session starts.
    """
    import cv2

    from camera.resolution_manager import DEFAULT_PRESETS

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return {"index": index, "available": False}

    default_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    default_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    default_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

    supported_presets = []
    for preset_name, (width, height) in DEFAULT_PRESETS.items():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (actual_w, actual_h) == (width, height):
            supported_presets.append(preset_name)

    cap.release()

    return {
        "index": index,
        "available": True,
        "default_resolution": (default_width, default_height),
        "default_fps": default_fps,
        "supported_presets": supported_presets,
    }
