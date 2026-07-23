"""Tests for feature_extraction.rep_counter."""

import numpy as np

from feature_extraction.rep_counter import RepCounter


def _sine_wave_angles(num_reps=3, frames_per_rep=30, amplitude=60, baseline=90):
    t = np.linspace(0, num_reps * 2 * np.pi, num_reps * frames_per_rep)
    return baseline + amplitude * np.sin(t)


def test_rep_counter_finds_correct_number_of_peaks():
    signal = _sine_wave_angles(num_reps=4)
    counter = RepCounter(smoothing_window=3, min_prominence_deg=20.0, min_distance_frames=10)
    result = counter.count(signal)
    assert result.rep_count == 4


def test_rep_counter_ignores_low_prominence_noise():
    rng = np.random.default_rng(0)
    flat_noisy_signal = 90 + rng.normal(0, 1.0, size=100)  # noise only, no real reps
    counter = RepCounter(smoothing_window=5, min_prominence_deg=15.0, min_distance_frames=10)
    result = counter.count(flat_noisy_signal)
    assert result.rep_count == 0


def test_rep_counter_handles_short_sequence_gracefully():
    counter = RepCounter()
    result = counter.count(np.array([1.0, 2.0]))
    assert result.rep_count == 0
    assert result.peak_frame_indices == []
