"""Tests for datasets.video_processor. Requires opencv-python."""

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from datasets.video_processor import VideoProcessor


def _make_video(path, num_frames=30, size=(160, 120), fps=15.0):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    for i in range(num_frames):
        frame = np.full((size[1], size[0], 3), fill_value=(i * 5) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


@pytest.fixture
def synthetic_video(tmp_path):
    video_path = tmp_path / "sample.mp4"
    _make_video(video_path)
    return video_path


def test_extract_frames_from_valid_video(synthetic_video):
    processor = VideoProcessor(target_resolution=(160, 120), target_fps=15, min_valid_frames=5)
    frames, report = processor.extract_frames(str(synthetic_video))
    assert report.is_valid
    assert len(frames) > 0
    assert frames[0].shape == (120, 160, 3)


def test_extract_frames_missing_file_is_invalid(tmp_path):
    processor = VideoProcessor()
    frames, report = processor.extract_frames(str(tmp_path / "does_not_exist.mp4"))
    assert frames == []
    assert not report.is_valid
    assert "does not exist" in report.reason.lower()


def test_extract_frames_respects_min_valid_frames(synthetic_video):
    # A video with only 30 frames sampled at a high frame_sample_rate should
    # end up below an unreasonably high min_valid_frames threshold.
    processor = VideoProcessor(target_fps=15, frame_sample_rate=10, min_valid_frames=1000)
    frames, report = processor.extract_frames(str(synthetic_video))
    assert not report.is_valid
    assert "too few valid frames" in report.reason.lower()


def test_detect_missing_frames():
    processor = VideoProcessor()
    assert processor.detect_missing_frames(expected_count=100, actual_count=80) == 20
    assert processor.detect_missing_frames(expected_count=50, actual_count=60) == 0
