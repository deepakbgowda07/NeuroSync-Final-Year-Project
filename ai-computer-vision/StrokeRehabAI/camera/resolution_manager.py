"""
resolution_manager.py
=======================
Manages the supported camera resolution presets (720p / 1080p) and
negotiates the actual resolution a connected camera reports back,
falling back gracefully (1080p -> 720p -> whatever the device reports)
if the requested preset isn't honored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_PRESETS: Dict[str, Tuple[int, int]] = {
    "720p": (1280, 720),
    "1080p": (1920, 1080),
}

# Presets tried, in order, if the requested one doesn't take.
FALLBACK_ORDER = ["1080p", "720p"]


@dataclass
class ResolutionResult:
    requested_preset: str
    requested_size: Tuple[int, int]
    actual_size: Tuple[int, int]
    matched_preset: bool


class ResolutionManager:
    """Resolves a named preset ("720p"/"1080p") to a (width, height) pair,
    and validates what a camera actually delivered against the request."""

    def __init__(self, presets: Optional[Dict[str, Tuple[int, int]]] = None):
        self.presets = presets or dict(DEFAULT_PRESETS)

    def resolve(self, preset: str) -> Tuple[int, int]:
        if preset not in self.presets:
            available = list(self.presets.keys())
            raise ValueError(f"Unknown resolution preset '{preset}'. Available: {available}")
        return self.presets[preset]

    def validate_actual(self, preset: str, actual_width: int, actual_height: int) -> ResolutionResult:
        """Compare what the camera actually reports against what was
        requested — cameras frequently ignore an unsupported resolution
        and silently fall back to their native default."""
        requested = self.resolve(preset)
        actual = (int(actual_width), int(actual_height))
        matched = actual == requested

        if not matched:
            logger.warning(
                "Camera did not honor requested resolution: asked for %s (%s), got %s.",
                preset, requested, actual,
            )
        return ResolutionResult(requested_preset=preset, requested_size=requested, actual_size=actual, matched_preset=matched)

    def negotiate(self, apply_fn, preferred_preset: str = "720p") -> ResolutionResult:
        """Try `preferred_preset` first, then fall back through
        FALLBACK_ORDER, calling `apply_fn(width, height) -> (actual_w, actual_h)`
        for each candidate until one is honored (or all are exhausted, in
        which case the last attempted result is returned)."""
        candidates = [preferred_preset] + [p for p in FALLBACK_ORDER if p != preferred_preset]
        result = None

        for preset in candidates:
            width, height = self.resolve(preset)
            actual_w, actual_h = apply_fn(width, height)
            result = self.validate_actual(preset, actual_w, actual_h)
            if result.matched_preset:
                logger.info("Camera resolution negotiated: %s (%dx%d).", preset, actual_w, actual_h)
                return result
            logger.info("Preset '%s' not honored (got %dx%d); trying next fallback.", preset, actual_w, actual_h)

        logger.warning("No resolution preset was honored; proceeding with the camera's actual reported size.")
        return result
