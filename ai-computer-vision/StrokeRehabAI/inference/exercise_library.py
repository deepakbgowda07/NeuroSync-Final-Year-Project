"""
exercise_library.py
=====================
Defines the exactly-10 supported rehabilitation exercises and the
per-exercise parameters (primary tracked joint angle, expected camera
view, clinical ROM range, rep-detection direction) that drive exercise
recognition, phase detection, rep counting, and error detection.

    Shoulder Flexion            Forearm Pronation
    Shoulder Abduction          Forearm Supination
    Elbow Flexion                Shoulder External Rotation
    Elbow Extension               Shoulder Internal Rotation
    Hand-to-Mouth Reach            Hand-to-Head Reach

All numeric thresholds are loaded from configs/exercises.yaml —
nothing here is hardcoded — and are principled starting defaults, not
clinician-validated values (see the TODO below).

TODO (next development phase): validate `min_acceptable_rom_deg` /
`max_acceptable_rom_deg` / `neutral_deg` / `target_deg` per exercise
against real clinician-annotated sessions; current values are drawn
from general physiotherapy ROM references, not this project's own data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_EXERCISES: List[str] = [
    "shoulder_flexion",
    "shoulder_abduction",
    "elbow_flexion",
    "elbow_extension",
    "forearm_pronation",
    "forearm_supination",
    "shoulder_external_rotation",
    "shoulder_internal_rotation",
    "hand_to_mouth",
    "hand_to_head",
]


@dataclass
class ExerciseDefinition:
    key: str
    display_name: str
    primary_angle: str          # angle name in utils.joint_angles output (left_* variant)
    secondary_angle: str        # the right_* counterpart
    required_view: str          # "front" | "side" | "any"
    neutral_deg: float
    target_deg: float
    min_acceptable_rom_deg: float
    max_acceptable_rom_deg: float
    rep_direction: str          # "increasing" | "decreasing"
    symmetric: bool

    def angle_name_for_side(self, side: str) -> str:
        return self.primary_angle if side == "left" else self.secondary_angle

    def rom_span(self) -> float:
        """Expected total range-of-motion span, neutral -> target."""
        return abs(self.target_deg - self.neutral_deg)

    def progress_fraction(self, current_deg: float) -> float:
        """Fraction of the way from neutral to target, clamped to [0, 1] —
        the basis for the pipeline's "Completion Percentage" output."""
        span = self.target_deg - self.neutral_deg
        if abs(span) < 1e-6:
            return 0.0
        fraction = (current_deg - self.neutral_deg) / span
        return max(0.0, min(1.0, fraction))


class ExerciseLibrary:
    """Loads and exposes ExerciseDefinition objects from configs/exercises.yaml."""

    def __init__(self, exercises_cfg):
        self._definitions: Dict[str, ExerciseDefinition] = {}
        self._load(exercises_cfg)

    def _load(self, exercises_cfg) -> None:
        from configs.config_loader import DotDict

        raw_definitions = exercises_cfg.definitions
        for key in SUPPORTED_EXERCISES:
            if key not in raw_definitions:
                logger.error("Exercise '%s' is missing from configs/exercises.yaml; skipping.", key)
                continue
            entry = DotDict(raw_definitions[key])
            self._definitions[key] = ExerciseDefinition(
                key=key,
                display_name=entry.display_name,
                primary_angle=entry.primary_angle,
                secondary_angle=entry.secondary_angle,
                required_view=entry.required_view,
                neutral_deg=entry.neutral_deg,
                target_deg=entry.target_deg,
                min_acceptable_rom_deg=entry.min_acceptable_rom_deg,
                max_acceptable_rom_deg=entry.max_acceptable_rom_deg,
                rep_direction=entry.rep_direction,
                symmetric=entry.symmetric,
            )

        missing = set(SUPPORTED_EXERCISES) - set(self._definitions.keys())
        if missing:
            logger.warning("Exercise library is missing definitions for: %s", missing)

    def get(self, key: str) -> ExerciseDefinition:
        if key not in self._definitions:
            raise KeyError(f"Unknown or unconfigured exercise: {key}")
        return self._definitions[key]

    def all(self) -> Dict[str, ExerciseDefinition]:
        return dict(self._definitions)

    def keys(self) -> List[str]:
        return list(self._definitions.keys())
