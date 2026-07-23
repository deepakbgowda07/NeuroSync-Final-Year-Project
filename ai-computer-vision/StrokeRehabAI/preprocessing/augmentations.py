"""
augmentations.py
=================
Skeleton-sequence (landmark-level) data augmentation for training-time
robustness: rotation, scale jitter, Gaussian noise, horizontal
mirroring (with correct left/right joint relabeling), temporal
cropping, random frame drop, and temporal stretch/warp.

These mimic natural variation between patients and camera setups
without altering exercise semantics. Frame-pixel-level augmentations
(brightness/contrast/noise on raw video frames) live separately in
`preprocessing/video_augmentations.py`, since they operate before pose
estimation rather than on extracted landmarks.

TODO (next development phase): validate that augmentation ranges are
clinically sensible once real data is used — e.g. confirm with a
clinician which exercises are safe to mirror for a hemiparetic
patient (see `mirror_prob` / `valid_for_mirroring` below).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from mediapipe_pipeline.landmark_extractor import LANDMARK_NAMES
from utils.logger import get_logger

logger = get_logger(__name__)

# Build the left<->right landmark index swap table once, from LANDMARK_NAMES,
# so mirroring both negates x and swaps sides correctly (rather than just
# negating x, which would leave a "left elbow" positioned as a mirrored
# "left elbow" instead of becoming the "right elbow").
def _build_mirror_index_map() -> List[int]:
    name_to_index = {name: i for i, name in enumerate(LANDMARK_NAMES)}
    mirror_map = list(range(len(LANDMARK_NAMES)))
    for i, name in enumerate(LANDMARK_NAMES):
        if name.startswith("left_"):
            counterpart = "right_" + name[len("left_"):]
        elif name.startswith("right_"):
            counterpart = "left_" + name[len("right_"):]
        else:
            continue
        if counterpart in name_to_index:
            mirror_map[i] = name_to_index[counterpart]
    return mirror_map


MIRROR_INDEX_MAP = _build_mirror_index_map()


@dataclass
class AugmentationConfig:
    rotation_deg_std: float = 5.0
    noise_std: float = 0.01
    scale_jitter_std: float = 0.03
    mirror_prob: float = 0.0          # disabled by default; see module TODO
    valid_for_mirroring: bool = True  # set False for exercises where L/R matters clinically
    temporal_crop_prob: float = 0.3
    temporal_crop_min_ratio: float = 0.8   # keep at least 80% of frames when cropping
    random_frame_drop_prob: float = 0.2
    random_frame_drop_max_ratio: float = 0.1  # drop at most 10% of frames
    temporal_stretch_prob: float = 0.3
    temporal_stretch_range: tuple = (0.85, 1.15)  # speed up/slow down factor


class SequenceAugmenter:
    """Applies randomized augmentations to a (T, 33, 3) landmark sequence.

    Note: temporal augmentations (crop/drop/stretch) change the sequence
    length. Callers that need a fixed length (e.g. before windowing)
    should apply this *before* preprocessing.sequence_builder, or
    re-pad/re-window after augmenting — see preprocessing/pipeline.py's
    offline path, which does exactly that.
    """

    def __init__(self, config: AugmentationConfig = None, rng: np.random.Generator = None):
        self.config = config or AugmentationConfig()
        self.rng = rng or np.random.default_rng()

    def __call__(self, sequence_xyz: np.ndarray) -> np.ndarray:
        seq = sequence_xyz.copy()

        if self.rng.random() < self.config.temporal_crop_prob:
            seq = self._temporal_crop(seq)
        if self.rng.random() < self.config.random_frame_drop_prob:
            seq = self._random_frame_drop(seq)
        if self.rng.random() < self.config.temporal_stretch_prob:
            seq = self._temporal_stretch(seq)

        seq = self._add_gaussian_noise(seq)
        seq = self._random_rotation(seq)
        seq = self._random_scale_jitter(seq)

        if self.config.valid_for_mirroring and self.rng.random() < self.config.mirror_prob:
            seq = self._mirror(seq)

        return seq

    # ------------------------------------------------------------------
    # Spatial augmentations
    # ------------------------------------------------------------------

    def _add_gaussian_noise(self, seq: np.ndarray) -> np.ndarray:
        noise = self.rng.normal(0, self.config.noise_std, size=seq.shape)
        return seq + noise

    def _random_rotation(self, seq: np.ndarray) -> np.ndarray:
        """Rotate all frames by a small random angle about the vertical (y) axis,
        simulating slightly different camera yaw between recording sessions."""
        angle_deg = self.rng.normal(0, self.config.rotation_deg_std)
        angle_rad = np.deg2rad(angle_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a],
        ])
        return seq @ rotation_matrix.T

    def _random_scale_jitter(self, seq: np.ndarray) -> np.ndarray:
        scale = 1.0 + self.rng.normal(0, self.config.scale_jitter_std)
        return seq * scale

    def _mirror(self, seq: np.ndarray) -> np.ndarray:
        """Negate the x-axis AND swap left/right landmark indices, so a
        mirrored 'left elbow' correctly becomes the (now mirrored) 'right
        elbow' rather than a spatially-flipped-but-still-labeled-left one.

        Only applied when `config.valid_for_mirroring` is True — for
        hemiparetic patients, mirroring an affected-side exercise onto
        the unaffected side (or vice versa) can be clinically invalid,
        so this must be disabled per-exercise by the caller/dataset
        converter where relevant (see module TODO).
        """
        mirrored = seq[:, MIRROR_INDEX_MAP, :].copy()
        mirrored[..., 0] *= -1
        return mirrored

    # ------------------------------------------------------------------
    # Temporal augmentations
    # ------------------------------------------------------------------

    def _temporal_crop(self, seq: np.ndarray) -> np.ndarray:
        """Randomly crop a contiguous sub-window of the sequence,
        keeping at least `temporal_crop_min_ratio` of the original frames —
        simulates imperfect exercise-boundary segmentation."""
        t = len(seq)
        min_len = max(2, int(t * self.config.temporal_crop_min_ratio))
        if min_len >= t:
            return seq
        crop_len = self.rng.integers(min_len, t + 1)
        start = self.rng.integers(0, t - crop_len + 1)
        return seq[start:start + crop_len]

    def _random_frame_drop(self, seq: np.ndarray) -> np.ndarray:
        """Randomly drop a small fraction of frames (simulating dropped
        frames from a live webcam feed) and close the gap, rather than
        leaving a hole — matches how preprocessing/sequence_builder.py
        handles missed pose detections (hold/skip), not insert blanks."""
        t = len(seq)
        max_drop = int(t * self.config.random_frame_drop_max_ratio)
        if max_drop <= 0:
            return seq
        num_drop = self.rng.integers(0, max_drop + 1)
        if num_drop == 0:
            return seq
        drop_indices = self.rng.choice(t, size=num_drop, replace=False)
        keep_mask = np.ones(t, dtype=bool)
        keep_mask[drop_indices] = False
        return seq[keep_mask]

    def _temporal_stretch(self, seq: np.ndarray) -> np.ndarray:
        """Resample the sequence to a slightly different length via linear
        interpolation, simulating a patient performing the exercise
        faster/slower than the reference speed."""
        t = len(seq)
        factor = self.rng.uniform(*self.config.temporal_stretch_range)
        new_len = max(2, int(round(t * factor)))

        original_idx = np.linspace(0, t - 1, num=t)
        new_idx = np.linspace(0, t - 1, num=new_len)

        stretched = np.empty((new_len, *seq.shape[1:]), dtype=seq.dtype)
        for joint in range(seq.shape[1]):
            for axis in range(seq.shape[2]):
                stretched[:, joint, axis] = np.interp(new_idx, original_idx, seq[:, joint, axis])
        return stretched
