"""Integration test for datasets.dataset_converter.DatasetConverter.convert_custom.

Requires opencv-python and mediapipe; skipped automatically if either is
unavailable in the current environment.
"""

import csv
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")
pytest.importorskip("mediapipe")

from datasets.dataset_converter import DatasetConverter
from datasets.label_processor import LabelProcessor
from datasets.landmark_cache import LandmarkCache
from datasets.video_processor import VideoProcessor


def _make_video(path, num_frames=20, size=(160, 120), fps=15.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    for i in range(num_frames):
        frame = np.full((size[1], size[0], 3), fill_value=(i * 7) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


@pytest.fixture
def custom_raw_dir(tmp_path):
    raw_dir = tmp_path / "raw"
    _make_video(raw_dir / "subj01" / "shoulder_flexion" / "trial1.mp4")
    _make_video(raw_dir / "subj02" / "shoulder_flexion" / "trial1.mp4")

    labels_csv = raw_dir / "labels.csv"
    with open(labels_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["subject_id", "exercise_name", "trial_name", "clinician_score"])
        writer.writerow(["subj01", "shoulder_flexion", "trial1", 8])
        writer.writerow(["subj02", "shoulder_flexion", "trial1", 4])

    return raw_dir, labels_csv


def test_convert_custom_produces_valid_unified_samples(tmp_path, custom_raw_dir):
    raw_dir, labels_csv = custom_raw_dir
    output_dir = tmp_path / "processed"

    converter = DatasetConverter(
        output_dir=str(output_dir),
        video_processor=VideoProcessor(target_resolution=(160, 120), target_fps=15, min_valid_frames=3),
        landmark_cache=LandmarkCache(cache_dir=str(tmp_path / "landmarks")),
        label_processor=LabelProcessor(clinician_score_range=(0, 10)),
    )

    results = converter.convert_custom(raw_dir, labels_csv=labels_csv)

    assert len(results) == 2
    assert all(r.success for r in results)

    npz_files = list(output_dir.glob("*.npz"))
    assert len(npz_files) == 2

    with np.load(npz_files[0], allow_pickle=True) as data:
        assert data["landmarks"].shape[1:] == (33, 3)
        assert data["landmarks"].shape[0] > 0
        metadata = data["metadata"].item()
        assert metadata["subject_id"] in ("subj01", "subj02")
        assert metadata["source_dataset"] == "CUSTOM"


def test_convert_custom_handles_missing_video_gracefully(tmp_path):
    raw_dir = tmp_path / "empty_raw"
    (raw_dir / "subj01" / "shoulder_flexion").mkdir(parents=True)
    output_dir = tmp_path / "processed"

    converter = DatasetConverter(output_dir=str(output_dir))
    results = converter.convert_custom(raw_dir)
    assert results == []  # no video files present -> nothing to convert, no crash
