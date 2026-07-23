"""Tests for inference.phase_detector.PhaseDetector."""

import numpy as np
import pytest

from configs.config_loader import load_config
from inference.exercise_library import ExerciseLibrary
from inference.phase_detector import ExercisePhase, PhaseDetector


@pytest.fixture
def elbow_flexion_defn():
    cfg = load_config(force_reload=True)
    return ExerciseLibrary(cfg.exercises).get("elbow_flexion")


def test_full_cycle_moves_through_expected_phases(elbow_flexion_defn):
    detector = PhaseDetector(fps=30)
    sequence = list(np.linspace(170, 45, 20)) + list(np.linspace(45, 170, 20))
    phases = [detector.update(a, elbow_flexion_defn).phase for a in sequence]

    assert phases[0] == ExercisePhase.NEUTRAL
    assert ExercisePhase.MOVING_TO_TARGET in phases
    assert ExercisePhase.PEAK in phases
    assert ExercisePhase.RETURNING in phases
    assert phases[-1] == ExercisePhase.NEUTRAL


def test_progress_fraction_increases_toward_target(elbow_flexion_defn):
    detector = PhaseDetector(fps=30)
    progress_values = [detector.update(a, elbow_flexion_defn).progress_fraction for a in np.linspace(170, 45, 20)]
    assert progress_values[-1] > progress_values[0]


def test_reset_clears_phase_state(elbow_flexion_defn):
    detector = PhaseDetector(fps=30)
    for a in np.linspace(170, 45, 10):
        detector.update(a, elbow_flexion_defn)
    detector.reset()
    result = detector.update(170.0, elbow_flexion_defn)
    assert result.phase == ExercisePhase.NEUTRAL
