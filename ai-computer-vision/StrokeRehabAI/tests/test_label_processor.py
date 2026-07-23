"""Tests for datasets.label_processor."""

from datasets.label_processor import LabelEncoder, LabelProcessor


def test_label_encoder_assigns_stable_indices():
    encoder = LabelEncoder()
    idx_a = encoder.encode("shoulder_flexion")
    idx_b = encoder.encode("elbow_flexion")
    idx_a_again = encoder.encode("shoulder_flexion")
    assert idx_a == idx_a_again
    assert idx_a != idx_b


def test_label_encoder_decode_roundtrip():
    encoder = LabelEncoder()
    idx = encoder.encode("knee_extension")
    assert encoder.decode(idx) == "knee_extension"


def test_normalize_clinician_score_scales_to_unit_interval():
    processor = LabelProcessor(clinician_score_range=(0, 100))
    assert processor.normalize_clinician_score(0) == 0.0
    assert processor.normalize_clinician_score(100) == 1.0
    assert processor.normalize_clinician_score(50) == 0.5


def test_normalize_clinician_score_clips_out_of_range():
    processor = LabelProcessor(clinician_score_range=(0, 10))
    assert processor.normalize_clinician_score(-5) == 0.0
    assert processor.normalize_clinician_score(15) == 1.0


def test_derive_binary_correctness_uses_threshold():
    processor = LabelProcessor(binary_threshold=0.6)
    assert processor.derive_binary_correctness(0.7) == 1
    assert processor.derive_binary_correctness(0.5) == 0


def test_build_label_derives_binary_and_class_from_clinician_score():
    processor = LabelProcessor(clinician_score_range=(0, 100), binary_threshold=0.6)
    label = processor.build_label(exercise_name="shoulder_flexion", raw_clinician_score=80)
    assert label.exercise_name == "shoulder_flexion"
    assert label.clinician_score == 80
    assert label.regression_target == 0.8
    assert label.binary_correct == 1
    assert label.quality_class == 1


def test_build_label_with_explicit_quality_class_overrides_derivation():
    processor = LabelProcessor(clinician_score_range=(0, 100))
    label = processor.build_label(exercise_name="elbow_flexion", quality_class=3, raw_clinician_score=20)
    assert label.quality_class == 3
    assert label.binary_correct == 0  # still derived from the low score
