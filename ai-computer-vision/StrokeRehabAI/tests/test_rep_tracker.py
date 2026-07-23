"""Tests for inference.rep_tracker (streaming rep counting)."""

from inference.phase_detector import ExercisePhase, PhaseResult
from inference.rep_tracker import RepTracker


def _phase_sequence(phases_and_progress):
    return [PhaseResult(phase=p, progress_fraction=prog, current_angle_deg=0.0, velocity_deg_per_sec=0.0) for p, prog in phases_and_progress]


def test_full_rep_cycle_counts_one():
    tracker = RepTracker(min_rep_duration_frames=2)
    sequence = _phase_sequence([
        (ExercisePhase.NEUTRAL, 0.0),
        (ExercisePhase.MOVING_TO_TARGET, 0.5),
        (ExercisePhase.PEAK, 1.0),
        (ExercisePhase.RETURNING, 0.5),
        (ExercisePhase.NEUTRAL, 0.0),
    ])
    events = [tracker.update(p) for p in sequence]
    completed_events = [e for e in events if e is not None]
    assert len(completed_events) == 1
    assert completed_events[0].completed
    assert tracker.rep_count == 1


def test_small_movement_never_reaching_peak_is_incomplete():
    tracker = RepTracker(min_rep_duration_frames=2)
    sequence = _phase_sequence([
        (ExercisePhase.NEUTRAL, 0.0),
        (ExercisePhase.MOVING_TO_TARGET, 0.2),
        (ExercisePhase.RETURNING, 0.1),
        (ExercisePhase.NEUTRAL, 0.0),
    ])
    events = [tracker.update(p) for p in sequence]
    completed_events = [e for e in events if e is not None]
    assert len(completed_events) == 1
    assert not completed_events[0].completed
    assert tracker.rep_count == 0


def test_two_full_reps_count_correctly():
    tracker = RepTracker(min_rep_duration_frames=2)
    one_rep = [
        (ExercisePhase.NEUTRAL, 0.0),
        (ExercisePhase.MOVING_TO_TARGET, 0.5),
        (ExercisePhase.PEAK, 1.0),
        (ExercisePhase.RETURNING, 0.5),
        (ExercisePhase.NEUTRAL, 0.0),
    ]
    sequence = _phase_sequence(one_rep + one_rep)
    for p in sequence:
        tracker.update(p)
    assert tracker.rep_count == 2


def test_reset_clears_rep_count():
    tracker = RepTracker(min_rep_duration_frames=2)
    sequence = _phase_sequence([
        (ExercisePhase.NEUTRAL, 0.0), (ExercisePhase.PEAK, 1.0), (ExercisePhase.NEUTRAL, 0.0),
    ])
    for p in sequence:
        tracker.update(p)
    tracker.reset()
    assert tracker.rep_count == 0
    assert tracker.events == []
