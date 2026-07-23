"""End-to-end integration test for inference.realtime_pipeline.RealtimeInferencePipeline.

Exercises the assembled pipeline's per-frame processing directly
(bypassing the live camera capture thread and cv2.imshow loop, which
need a real display/camera) against synthetic frames, covering both
the "no pose detected" and "pose detected" code paths without crashing.
"""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("mediapipe")
torch = pytest.importorskip("torch")


@pytest.fixture
def pipeline(tmp_path):
    from configs.config_loader import load_config
    from inference.realtime_pipeline import RealtimeInferencePipeline

    cfg = load_config(force_reload=True)
    cfg.dashboard.database_path = str(tmp_path / "session.db")

    p = RealtimeInferencePipeline(cfg=cfg, patient_id=None, skip_calibration=True)
    p.pose_estimator.open()
    p.session_logger.start_session()
    import time
    p._session_start_time = time.time()
    yield p
    p.session_logger.end_session()
    p.pose_estimator.close()


def test_process_frame_handles_no_pose_detected(pipeline):
    frame = np.random.randint(0, 255, size=(240, 320, 3), dtype=np.uint8)
    display = pipeline._process_frame(frame)
    assert display.shape == frame.shape


def test_process_frame_runs_multiple_frames_without_crashing(pipeline):
    for _ in range(5):
        frame = np.random.randint(0, 255, size=(240, 320, 3), dtype=np.uint8)
        display = pipeline._process_frame(frame)
        assert display is not None


def test_process_frame_with_synthetic_person_shape(pipeline):
    frame = np.full((480, 640, 3), 220, dtype=np.uint8)
    cv2.circle(frame, (320, 100), 40, (150, 120, 100), -1)
    cv2.rectangle(frame, (270, 140), (370, 320), (100, 80, 200), -1)
    cv2.line(frame, (270, 160), (180, 260), (100, 80, 200), 20)
    cv2.line(frame, (370, 160), (460, 260), (100, 80, 200), 20)

    for _ in range(5):
        display = pipeline._process_frame(frame)
        assert display.shape == frame.shape
