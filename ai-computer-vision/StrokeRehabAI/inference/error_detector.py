"""
error_detector.py
===================
Detects movement-quality errors in real time from the current frame's
joint angles, phase, rep state, and calibration baseline:

    Incorrect Joint Angles       Body Lean
    Insufficient ROM              Poor Alignment
    Excessive ROM                  Incorrect Exercise Sequence
    Incomplete Repetition           Asymmetrical Motion
    Fast Movement                    Late Movement
    Jerky Movement                    Premature Return
    Trunk Compensation                 Shoulder Hiking

Each detector is a small, independently-testable rule built on the
exercise's config-driven thresholds (`configs/exercises.yaml`) and the
patient's own calibration baseline (`inference/calibration.py`) —
nothing here is a population-average hardcoded number where a
personalized one is available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from feature_extraction.clinical_features import ClinicalFeatureExtractor
from inference.calibration import CalibrationProfile
from inference.exercise_library import ExerciseDefinition
from inference.phase_detector import ExercisePhase, PhaseResult
from inference.rep_tracker import RepEvent
from utils.logger import get_logger

logger = get_logger(__name__)


class ErrorType(str, Enum):
    INCORRECT_JOINT_ANGLE = "incorrect_joint_angle"
    INSUFFICIENT_ROM = "insufficient_rom"
    EXCESSIVE_ROM = "excessive_rom"
    INCOMPLETE_REPETITION = "incomplete_repetition"
    FAST_MOVEMENT = "fast_movement"
    JERKY_MOVEMENT = "jerky_movement"
    TRUNK_COMPENSATION = "trunk_compensation"
    SHOULDER_HIKING = "shoulder_hiking"
    BODY_LEAN = "body_lean"
    POOR_ALIGNMENT = "poor_alignment"
    INCORRECT_SEQUENCE = "incorrect_sequence"
    ASYMMETRICAL_MOTION = "asymmetrical_motion"
    LATE_MOVEMENT = "late_movement"
    PREMATURE_RETURN = "premature_return"


@dataclass
class DetectedError:
    error_type: ErrorType
    severity: str  # "minor" | "moderate" | "severe"
    detail: Dict = field(default_factory=dict)


class ErrorDetector:
    """Runs the full suite of movement-quality error checks for one frame."""

    def __init__(
        self,
        fast_movement_deg_per_sec: float = 200.0,
        jerk_std_threshold: float = 3.0,
        trunk_lean_threshold_deg: float = 15.0,
        shoulder_hiking_threshold: float = 0.15,
        asymmetry_threshold: float = 0.35,
        late_movement_max_frames_after_cue: int = 90,
        fps: float = 30.0,
    ):
        self.fast_movement_deg_per_sec = fast_movement_deg_per_sec
        self.jerk_std_threshold = jerk_std_threshold
        self.trunk_lean_threshold_deg = trunk_lean_threshold_deg
        self.shoulder_hiking_threshold = shoulder_hiking_threshold
        self.asymmetry_threshold = asymmetry_threshold
        self.late_movement_max_frames_after_cue = late_movement_max_frames_after_cue
        self.clinical_extractor = ClinicalFeatureExtractor(fps=fps)

        self._velocity_history: List[float] = []
        self._frames_since_cue = 0
        self._expected_next_exercise: Optional[str] = None

    def reset(self) -> None:
        self._velocity_history.clear()
        self._frames_since_cue = 0

    def detect(
        self,
        landmarks_xyz: np.ndarray,
        angles: Dict[str, float],
        definition: ExerciseDefinition,
        phase_result: PhaseResult,
        calibration: Optional[CalibrationProfile] = None,
        recognized_exercise_key: Optional[str] = None,
        expected_exercise_key: Optional[str] = None,
    ) -> List[DetectedError]:
        """Run every check for the current frame; returns the list of
        errors currently active (empty list = no errors detected)."""
        errors: List[DetectedError] = []

        errors += self._check_joint_angle_and_rom(angles, definition, phase_result, calibration)
        errors += self._check_movement_speed(phase_result)
        errors += self._check_compensation(landmarks_xyz)
        errors += self._check_symmetry(angles, definition)
        errors += self._check_sequence(recognized_exercise_key, expected_exercise_key)

        return errors

    # ------------------------------------------------------------------
    # Angle / ROM
    # ------------------------------------------------------------------

    def _check_joint_angle_and_rom(
        self, angles: Dict[str, float], definition: ExerciseDefinition,
        phase_result: PhaseResult, calibration: Optional[CalibrationProfile],
    ) -> List[DetectedError]:
        errors = []
        current_angle = phase_result.current_angle_deg

        if phase_result.phase == ExercisePhase.PEAK:
            if definition.rep_direction == "increasing":
                if current_angle < definition.min_acceptable_rom_deg:
                    errors.append(DetectedError(ErrorType.INSUFFICIENT_ROM, "moderate", {"angle_deg": current_angle, "min_required_deg": definition.min_acceptable_rom_deg}))
                elif current_angle > definition.max_acceptable_rom_deg:
                    errors.append(DetectedError(ErrorType.EXCESSIVE_ROM, "minor", {"angle_deg": current_angle, "max_allowed_deg": definition.max_acceptable_rom_deg}))
            else:
                # "decreasing" exercises (e.g. elbow flexion: neutral=170 -> target=45):
                # max_acceptable_rom_deg is the "didn't move enough" ceiling (still
                # too close to neutral), min_acceptable_rom_deg is the "moved too
                # far" floor (over-flexed past the clinically expected endpoint).
                if current_angle > definition.max_acceptable_rom_deg:
                    errors.append(DetectedError(ErrorType.INSUFFICIENT_ROM, "moderate", {"angle_deg": current_angle, "max_required_deg": definition.max_acceptable_rom_deg}))
                elif current_angle < definition.min_acceptable_rom_deg:
                    errors.append(DetectedError(ErrorType.EXCESSIVE_ROM, "minor", {"angle_deg": current_angle, "min_allowed_deg": definition.min_acceptable_rom_deg}))

        # Personalized sanity check: if calibration captured this angle at
        # baseline and the current reading deviates wildly even at
        # neutral, flag as an incorrect joint angle (e.g. compensatory
        # starting posture rather than a genuine neutral position).
        if calibration is not None and phase_result.phase == ExercisePhase.NEUTRAL:
            baseline = calibration.baseline_joint_angles.get(definition.primary_angle)
            if baseline is not None and not np.isnan(baseline) and abs(current_angle - baseline) > 25.0:
                errors.append(DetectedError(
                    ErrorType.INCORRECT_JOINT_ANGLE, "minor",
                    {"angle_deg": current_angle, "calibrated_baseline_deg": baseline},
                ))

        return errors

    def register_incomplete_rep(self, rep_event: RepEvent) -> Optional[DetectedError]:
        """Called by the pipeline whenever RepTracker finalizes an
        incomplete rep — kept separate from `detect()` since it's an
        event, not a per-frame condition."""
        if rep_event.flagged_incomplete:
            return DetectedError(
                ErrorType.INCOMPLETE_REPETITION, "moderate",
                {"peak_progress_fraction": rep_event.peak_progress_fraction, "duration_frames": rep_event.duration_frames},
            )
        return None

    # ------------------------------------------------------------------
    # Speed / smoothness
    # ------------------------------------------------------------------

    def _check_movement_speed(self, phase_result: PhaseResult) -> List[DetectedError]:
        errors = []
        velocity = phase_result.velocity_deg_per_sec
        self._velocity_history.append(velocity)
        if len(self._velocity_history) > 30:
            self._velocity_history.pop(0)

        if abs(velocity) > self.fast_movement_deg_per_sec:
            errors.append(DetectedError(ErrorType.FAST_MOVEMENT, "minor", {"velocity_deg_per_sec": velocity}))

        if len(self._velocity_history) >= 5:
            jerk = np.diff(self._velocity_history)
            if float(np.std(jerk)) > self.jerk_std_threshold * max(1.0, np.std(self._velocity_history)):
                errors.append(DetectedError(ErrorType.JERKY_MOVEMENT, "minor", {"jerk_std": float(np.std(jerk))}))

        return errors

    # ------------------------------------------------------------------
    # Compensation / posture
    # ------------------------------------------------------------------

    def _check_compensation(self, landmarks_xyz: np.ndarray) -> List[DetectedError]:
        errors = []
        landmarks_seq = landmarks_xyz[np.newaxis, ...]  # ClinicalFeatureExtractor expects (T, 33, 3)

        trunk_lean = float(self.clinical_extractor.trunk_lean_deg(landmarks_seq)[0])
        if trunk_lean > self.trunk_lean_threshold_deg:
            errors.append(DetectedError(ErrorType.TRUNK_COMPENSATION, "moderate", {"trunk_lean_deg": trunk_lean}))
            errors.append(DetectedError(ErrorType.BODY_LEAN, "minor", {"trunk_lean_deg": trunk_lean}))

        for side in ("left", "right"):
            elevation = float(self.clinical_extractor.shoulder_elevation(landmarks_seq, side=side)[0])
            if elevation > self.shoulder_hiking_threshold:
                errors.append(DetectedError(ErrorType.SHOULDER_HIKING, "minor", {"side": side, "elevation": elevation}))

        # Alignment: coarse check that the shoulder line stays roughly
        # perpendicular to the spine (torso) axis — a large deviation
        # suggests the patient has rotated/twisted out of a square posture.
        from utils.joint_angles import MEDIAPIPE_LANDMARK_INDEX

        left_shoulder = landmarks_xyz[MEDIAPIPE_LANDMARK_INDEX["left_shoulder"]]
        right_shoulder = landmarks_xyz[MEDIAPIPE_LANDMARK_INDEX["right_shoulder"]]
        left_hip = landmarks_xyz[MEDIAPIPE_LANDMARK_INDEX["left_hip"]]
        right_hip = landmarks_xyz[MEDIAPIPE_LANDMARK_INDEX["right_hip"]]

        shoulder_vec = right_shoulder - left_shoulder
        hip_vec = right_hip - left_hip
        denom = (np.linalg.norm(shoulder_vec) * np.linalg.norm(hip_vec)) or 1e-8
        cos_angle = np.clip(np.dot(shoulder_vec, hip_vec) / denom, -1.0, 1.0)
        misalignment_deg = float(np.degrees(np.arccos(cos_angle)))
        if misalignment_deg > 20.0:
            errors.append(DetectedError(ErrorType.POOR_ALIGNMENT, "minor", {"shoulder_hip_misalignment_deg": misalignment_deg}))

        return errors

    # ------------------------------------------------------------------
    # Symmetry
    # ------------------------------------------------------------------

    def _check_symmetry(self, angles: Dict[str, float], definition: ExerciseDefinition) -> List[DetectedError]:
        if not definition.symmetric:
            return []

        left = angles.get(definition.primary_angle)
        right = angles.get(definition.secondary_angle)
        if left is None or right is None or np.isnan(left) or np.isnan(right):
            return []

        span = max(definition.rom_span(), 1e-6)
        asymmetry = abs(left - right) / span
        if asymmetry > self.asymmetry_threshold:
            return [DetectedError(ErrorType.ASYMMETRICAL_MOTION, "moderate", {"left_deg": left, "right_deg": right, "asymmetry_fraction": asymmetry})]
        return []

    # ------------------------------------------------------------------
    # Sequence / timing
    # ------------------------------------------------------------------

    def _check_sequence(self, recognized_key: Optional[str], expected_key: Optional[str]) -> List[DetectedError]:
        if expected_key is None or recognized_key is None:
            return []
        if recognized_key != expected_key:
            return [DetectedError(ErrorType.INCORRECT_SEQUENCE, "minor", {"expected": expected_key, "recognized": recognized_key})]
        return []

    def check_late_movement(self, frames_since_cue: int) -> Optional[DetectedError]:
        """Called by the pipeline once per rep cue (e.g. "begin rep now")
        with the number of frames elapsed since the cue was issued —
        flags the patient as slow to initiate movement."""
        if frames_since_cue > self.late_movement_max_frames_after_cue:
            return DetectedError(ErrorType.LATE_MOVEMENT, "minor", {"frames_since_cue": frames_since_cue})
        return None

    def check_premature_return(self, rep_event: RepEvent) -> Optional[DetectedError]:
        if rep_event.flagged_premature_return:
            return DetectedError(ErrorType.PREMATURE_RETURN, "moderate", {"peak_progress_fraction": rep_event.peak_progress_fraction})
        return None
