"""
video_augmentations.py
========================
Pixel-space augmentations applied to raw video frames *before* pose
estimation — brightness, contrast, and Gaussian sensor noise — used
during dataset conversion to synthesize lighting/camera variation from
a fixed set of recorded videos.

These are distinct from `preprocessing/augmentations.py` (landmark-
space augmentations applied after pose estimation): frame augmentations
change what MediaPipe "sees", useful for testing/improving pose
estimation robustness, while landmark augmentations change what the
downstream model sees, useful for exercise-assessment robustness.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class VideoAugmentationConfig:
    brightness_delta_range: tuple = (-30, 30)     # additive, in 0-255 pixel units
    contrast_factor_range: tuple = (0.85, 1.15)   # multiplicative around mid-gray
    gaussian_noise_std: float = 5.0               # pixel-value std-dev


class FrameAugmenter:
    """Applies randomized pixel-space augmentations to a single BGR frame."""

    def __init__(self, config: VideoAugmentationConfig = None, rng: np.random.Generator = None):
        self.config = config or VideoAugmentationConfig()
        self.rng = rng or np.random.default_rng()

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        frame = self.adjust_brightness(frame)
        frame = self.adjust_contrast(frame)
        frame = self.add_gaussian_noise(frame)
        return frame

    def adjust_brightness(self, frame: np.ndarray) -> np.ndarray:
        delta = self.rng.uniform(*self.config.brightness_delta_range)
        return np.clip(frame.astype(np.float32) + delta, 0, 255).astype(np.uint8)

    def adjust_contrast(self, frame: np.ndarray) -> np.ndarray:
        factor = self.rng.uniform(*self.config.contrast_factor_range)
        mean = frame.mean()
        return np.clip((frame.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)

    def add_gaussian_noise(self, frame: np.ndarray) -> np.ndarray:
        if self.config.gaussian_noise_std <= 0:
            return frame
        noise = self.rng.normal(0, self.config.gaussian_noise_std, size=frame.shape)
        return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
