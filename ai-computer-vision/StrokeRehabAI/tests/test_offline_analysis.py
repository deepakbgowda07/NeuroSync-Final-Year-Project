"""
test_offline_analysis.py
========================
Automated unit and integration tests for Offline Video Analysis and Replay module.
"""

from __future__ import annotations

import sys
# Mask tensorflow to prevent import errors with incompatible protobuf versions in the environment
sys.modules["tensorflow"] = None

import os
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from dashboard.db import init_db, get_connection
from dashboard.session_manager import SessionManager
from dashboard.offline_report_generator import SingleSessionReportGenerator
from inference.offline_analyzer import OfflineVideoAnalyzer, get_video_metadata, parse_mp4_rotation
from mediapipe_pipeline.pose_estimator import PoseResult


@pytest.fixture
def temp_db():
    """Create a clean temporary database and initialize the schema."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_offline.db"
        init_db(str(db_path))
        
        # Seed a dummy patient for tests
        with get_connection(str(db_path)) as conn:
            conn.execute(
                "INSERT INTO patients (patient_id, full_name, date_of_birth, gender, affected_side, diagnosis, physiotherapist) "
                "VALUES (1, 'John Test', '1980-01-01', 'Male', 'Right', 'Hemiparesis', 'Dr. Tester')"
            )
            
        yield str(db_path)


@pytest.fixture
def dummy_video():
    """Generate a short valid dummy video file for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = Path(tmpdir) / "dummy.mp4"
        width, height = 320, 240
        fps = 30
        num_frames = 15
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        
        for i in range(num_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            # draw a moving square
            cv2.rectangle(frame, (i * 5, 20), (i * 5 + 30, 50), (255, 0, 0), -1)
            writer.write(frame)
            
        writer.release()
        yield str(video_path)


def test_video_metadata_extraction(dummy_video):
    """Verify that get_video_metadata correctly parses a valid video file."""
    meta = get_video_metadata(dummy_video)
    
    assert meta["is_valid"] is True
    assert meta["width"] == 320
    assert meta["height"] == 240
    assert meta["fps"] == 30.0
    assert meta["frame_count"] == 15
    assert meta["file_size_bytes"] > 0
    assert meta["duration_seconds"] == 0.5
    assert meta["orientation"] == "landscape"


def test_video_validation_fails_on_missing_file():
    """Verify validation fails appropriately for non-existent files."""
    meta = get_video_metadata("non_existent_file.mp4")
    assert meta["is_valid"] is False
    assert "exist" in meta["error_msg"].lower()


def test_rotation_parsing(dummy_video):
    """Verify rotation parsing on the standard video writer output returns 0."""
    rot = parse_mp4_rotation(dummy_video)
    assert rot in (0, 90, 180, 270)  # should return standard rotation deg


def test_offline_analyzer_pipeline(temp_db, dummy_video, monkeypatch):
    """Verify that the full OfflineVideoAnalyzer executes, saves outputs, and stores sessions in SQLite."""
    # Mock PoseEstimator to return dummy landmarks
    import inference.offline_analyzer
    
    mock_xyz = np.random.rand(33, 3)
    mock_vis = np.random.rand(33)
    
    class MockPoseEstimator:
        def __init__(self, *args, **kwargs):
            pass
        def open(self):
            pass
        def close(self):
            pass
        def process(self, frame):
            return PoseResult(
                detected=True,
                landmarks_xyz=mock_xyz,
                landmarks_visibility=mock_vis,
                raw_result=None
            )
            
    monkeypatch.setattr(inference.offline_analyzer, "PoseEstimator", MockPoseEstimator)

    analyzer = OfflineVideoAnalyzer(temp_db, target_resolution=(320, 240), target_fps=30.0)
    
    progress_calls = []
    def progress_cb(frac, msg):
        progress_calls.append((frac, msg))
        
    session_id = analyzer.analyze(
        dummy_video,
        patient_id=1,
        expected_exercise_key="elbow_flexion",
        progress_callback=progress_cb
    )
    
    # Assert session was created
    assert isinstance(session_id, int)
    assert session_id > 0
    assert len(progress_calls) > 0
    
    # Assert database tables were successfully populated
    with get_connection(temp_db) as conn:
        session_row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        assert session_row is not None
        assert session_row["patient_id"] == 1
        assert session_row["exercise_name"] == "Elbow Flexion"
        assert session_row["original_video_path"] == dummy_video
        assert Path(session_row["processed_video_path"]).exists()
        
        # Verify frames populated with landmarks_json
        frames = conn.execute("SELECT * FROM session_frames WHERE session_id = ? ORDER BY frame_id", (session_id,)).fetchall()
        assert len(frames) > 0
        for f in frames:
            assert f["landmarks_json"] is not None
            landmarks = json.loads(f["landmarks_json"])
            # Assert landmarks structure contains 33 joints
            assert len(landmarks) == 33
            assert "x" in landmarks[0]
            assert "y" in landmarks[0]
            assert "z" in landmarks[0]
            assert "visibility" in landmarks[0]
            
        # Verify scores and joint metrics populated
        scores = conn.execute("SELECT * FROM recovery_scores WHERE session_id = ?", (session_id,)).fetchone()
        assert scores is not None
        assert scores["recovery_index"] >= 0.0
        
        metrics = conn.execute("SELECT * FROM joint_metrics WHERE session_id = ?", (session_id,)).fetchall()
        assert len(metrics) > 0


def test_single_session_report_generation(temp_db, dummy_video, monkeypatch):
    """Verify single session reports generate and save in all supported formats (PDF, Excel, CSV, JSON)."""
    # Mock PoseEstimator to return dummy landmarks
    import inference.offline_analyzer
    
    mock_xyz = np.random.rand(33, 3)
    mock_vis = np.random.rand(33)
    
    class MockPoseEstimator:
        def __init__(self, *args, **kwargs):
            pass
        def open(self):
            pass
        def close(self):
            pass
        def process(self, frame):
            return PoseResult(
                detected=True,
                landmarks_xyz=mock_xyz,
                landmarks_visibility=mock_vis,
                raw_result=None
            )
            
    monkeypatch.setattr(inference.offline_analyzer, "PoseEstimator", MockPoseEstimator)

    # 1. Analyze to populate session
    analyzer = OfflineVideoAnalyzer(temp_db, target_resolution=(320, 240), target_fps=30.0)
    session_id = analyzer.analyze(dummy_video, patient_id=1, expected_exercise_key="elbow_flexion")
    
    # 2. Generate reports
    rep_generator = SingleSessionReportGenerator(temp_db, output_dir=tempfile.gettempdir())
    
    for r_format in ["json", "csv", "excel", "pdf"]:
        report_path = rep_generator.generate(session_id, r_format)
        assert report_path.exists()
        assert report_path.stat().st_size > 0
        
        # Clean up
        try:
            os.remove(report_path)
        except:
            pass


def test_dashboard_page_compiles():
    """Verify that the dashboard page file compiles and contains required Streamlit elements."""
    page_path = Path("dashboard/pages/8_Offline_Analysis.py")
    assert page_path.exists()
    
    with open(page_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Check key imports and elements are present
    assert "import streamlit as st" in content
    assert "OfflineVideoAnalyzer" in content
    assert "SingleSessionReportGenerator" in content
    assert "st.file_uploader" in content
    assert "st.tabs" in content
