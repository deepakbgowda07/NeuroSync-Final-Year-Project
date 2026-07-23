"""
augmentation.py
================
Training-loop-facing augmentation entry point. Builds a configured
SequenceAugmenter from configs/augmentation.yaml; kept as a separate
module (per the requested training/ file layout) so augmentation
policy can evolve independently of the low-level transform code in
preprocessing/augmentations.py.
"""

from __future__ import annotations

from preprocessing.augmentations import AugmentationConfig, SequenceAugmenter
from preprocessing.video_augmentations import FrameAugmenter, VideoAugmentationConfig
from utils.logger import get_logger

logger = get_logger(__name__)


def build_augmenter(augmentation_cfg=None) -> SequenceAugmenter:
    """Build the landmark-space augmenter used during training, from the
    `augmentation.landmark` section of the merged application config."""
    if augmentation_cfg is None or not augmentation_cfg.get("enabled", True):
        logger.info("Augmentation disabled; returning a no-op-equivalent augmenter.")
        return SequenceAugmenter(config=AugmentationConfig(
            rotation_deg_std=0.0, noise_std=0.0, scale_jitter_std=0.0,
            temporal_crop_prob=0.0, random_frame_drop_prob=0.0, temporal_stretch_prob=0.0,
        ))

    landmark_cfg = augmentation_cfg.landmark
    config = AugmentationConfig(
        rotation_deg_std=landmark_cfg.rotation_deg_std,
        noise_std=landmark_cfg.noise_std,
        scale_jitter_std=landmark_cfg.scale_jitter_std,
        mirror_prob=landmark_cfg.mirror_prob,
        valid_for_mirroring=landmark_cfg.valid_for_mirroring,
        temporal_crop_prob=landmark_cfg.temporal_crop_prob,
        temporal_crop_min_ratio=landmark_cfg.temporal_crop_min_ratio,
        random_frame_drop_prob=landmark_cfg.random_frame_drop_prob,
        random_frame_drop_max_ratio=landmark_cfg.random_frame_drop_max_ratio,
        temporal_stretch_prob=landmark_cfg.temporal_stretch_prob,
        temporal_stretch_range=tuple(landmark_cfg.temporal_stretch_range),
    )
    return SequenceAugmenter(config=config)


def build_frame_augmenter(augmentation_cfg=None) -> FrameAugmenter:
    """Build the frame/pixel-space augmenter used during dataset
    conversion, from the `augmentation.video` section of the config."""
    video_cfg = augmentation_cfg.video if augmentation_cfg else None
    if video_cfg is None or not video_cfg.get("enabled", False):
        return FrameAugmenter(config=VideoAugmentationConfig(gaussian_noise_std=0.0))

    config = VideoAugmentationConfig(
        brightness_delta_range=tuple(video_cfg.brightness_delta_range),
        contrast_factor_range=tuple(video_cfg.contrast_factor_range),
        gaussian_noise_std=video_cfg.gaussian_noise_std,
    )
    return FrameAugmenter(config=config)
