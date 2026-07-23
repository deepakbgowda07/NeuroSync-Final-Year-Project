"""
rom_scoring.py
===============
Range-of-motion (ROM) based exercise-quality scoring — a domain-
specific evaluation metric (listed in configs/evaluation.yaml as
`rom_error_degrees`) that compares a patient's measured joint-angle
range against a clinical reference range for a given exercise.

TODO (next development phase): populate REFERENCE_ROM_TABLE with
values sourced from physiotherapy literature per exercise type, once
the target exercise set is finalized.
"""

from __future__ import annotations

from typing import Dict

from utils.logger import get_logger

logger = get_logger(__name__)

# Placeholder — TODO: replace with clinically sourced reference ranges,
# keyed by exercise name -> {angle_name: (min_deg, max_deg)}.
REFERENCE_ROM_TABLE: Dict[str, Dict[str, tuple]] = {
    "shoulder_flexion": {"left_shoulder_angle": (0, 180), "right_shoulder_angle": (0, 180)},
    "elbow_flexion": {"left_elbow_angle": (0, 150), "right_elbow_angle": (0, 150)},
}


def rom_error_degrees(exercise_name: str, measured_ranges: Dict[str, float]) -> Dict[str, float]:
    """Compute, per joint angle, how far the measured ROM falls short of
    (or exceeds) the clinical reference range for the given exercise."""
    reference = REFERENCE_ROM_TABLE.get(exercise_name)
    if reference is None:
        logger.warning("No reference ROM table for exercise '%s'.", exercise_name)
        return {}

    errors = {}
    for angle_name, (ref_min, ref_max) in reference.items():
        measured = measured_ranges.get(angle_name)
        if measured is None:
            continue
        if measured < ref_min:
            errors[angle_name] = ref_min - measured
        elif measured > ref_max:
            errors[angle_name] = measured - ref_max
        else:
            errors[angle_name] = 0.0
    return errors
