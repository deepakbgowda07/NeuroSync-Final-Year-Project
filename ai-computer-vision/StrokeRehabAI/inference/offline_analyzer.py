"""
offline_analyzer.py
===================
Offline Video Analysis engine for StrokeRehabAI.
Validates uploaded videos, parses orientation metadata, normalizes FPS/resolution,
executes the pose estimation and biomechanics pipeline, generates an annotated
output video, and persists all session details to the SQLite database.
"""

from __future__ import annotations

import sys
# Mask tensorflow to prevent import errors with incompatible protobuf versions in the environment
sys.modules["tensorflow"] = None

import os
import json
import time
import struct
import argparse
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from configs.config_loader import load_config
from camera.camera_utils import resize_frame
from mediapipe_pipeline.pose_estimator import PoseEstimator, PoseResult
from mediapipe_pipeline.landmark_extractor import LandmarkExtractor
from mediapipe_pipeline.pose_smoothing import LandmarkSmoother, PoseGapHandler
from mediapipe_pipeline.view_detector import ViewDetector
from inference.smoothing import ConfidenceSmoother
from inference.exercise_library import ExerciseLibrary
from inference.movement_analyzer import MovementAnalyzer, MovementAnalysisResult
from inference.session_logger import SessionLogger
from dashboard.analytics import RecoveryAnalyticsEngine
from dashboard.db import get_connection, init_db
from utils.joint_angles import compute_all_joint_angles
from utils.logger import get_logger, log_gpu_info
from utils.timers import FPSCounter, StageTimer
from visualization.hud_overlay import HUDOverlay, HUDState
from visualization.skeleton_renderer import SkeletonRenderer
from visualization.ghost_skeleton import GhostSkeletonRenderer
from visualization.correction_arrows import CorrectionArrowRenderer
from visualization.ideal_pose import generate_ideal_pose

logger = get_logger(__name__)


def parse_mp4_rotation(video_path: str) -> int:
    """Read the rotation matrix from the track header (tkhd) box of an MP4/MOV file.
    Supports both version 0 and version 1 headers. Returns rotation in degrees (0, 90, 180, 270).
    """
    try:
        with open(video_path, "rb") as f:
            # Read first 10MB to search for track header matrix (normally near start or end)
            data = f.read(10 * 1024 * 1024)
            tkhd_idx = data.find(b"tkhd")
            if tkhd_idx == -1:
                return 0
            
            # version is at tkhd_idx + 4
            version = data[tkhd_idx + 4]
            # matrix starts at offset 60 (version 1) or 48 (version 0) inside tkhd box
            matrix_offset = 60 if version == 1 else 48
            matrix_data = data[tkhd_idx + matrix_offset : tkhd_idx + matrix_offset + 36]
            if len(matrix_data) < 36:
                return 0
            
            # matrix elements: 9 32-bit big-endian integers
            matrix = struct.unpack(">9i", matrix_data)
            a, b, _, c, d = matrix[0], matrix[1], matrix[2], matrix[3], matrix[4]
            
            # Normalize 16.16 values to float
            a_f = a / 65536.0
            b_f = b / 65536.0
            c_f = c / 65536.0
            d_f = d / 65536.0
            
            # Classify rotation based on transformation matrix
            if abs(a_f) < 0.1 and abs(d_f) < 0.1:
                if b_f > 0.9 and c_f < -0.9:
                    return 90
                elif b_f < -0.9 and c_f > 0.9:
                    return 270
            elif abs(b_f) < 0.1 and abs(c_f) < 0.1:
                if a_f < -0.9 and d_f < -0.9:
                    return 180
    except Exception as exc:
        logger.debug("Failed parsing rotation from tkhd box: %s", exc)
    return 0


def get_video_metadata(video_path: str) -> Dict:
    """Validate video file exists, can be opened, and extract key metadata features."""
    path = Path(video_path)
    if not path.exists():
        return {"is_valid": False, "error_msg": f"File does not exist: {path}"}
    
    file_size_bytes = os.path.getsize(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"is_valid": False, "error_msg": f"Could not open video file. Video might be corrupted or in an unsupported codec."}
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc_val = int(cap.get(cv2.CAP_PROP_FOURCC))
    
    cap.release()
    
    if frame_count <= 0 or fps <= 0 or width <= 0 or height <= 0:
        return {"is_valid": False, "error_msg": f"Video metadata is corrupt (Frames: {frame_count}, FPS: {fps:.2f}, Size: {width}x{height})"}
    
    codec = "".join([chr((fourcc_val >> 8 * i) & 0xFF) for i in range(4)]).strip()
    duration_seconds = frame_count / fps
    
    # Try parsing rotation
    rotation = parse_mp4_rotation(str(path))
    
    return {
        "is_valid": True,
        "duration_seconds": duration_seconds,
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "codec": codec if codec else "Unknown",
        "file_size_bytes": file_size_bytes,
        "rotation_degrees": rotation,
        "orientation": "portrait" if (rotation in (90, 270) and width > height) or (rotation not in (90, 270) and height > width) else "landscape"
    }


class OfflineVideoAnalyzer:
    """Executes the rehabilitation assessment pipeline offline on video uploads."""

    def __init__(self, db_path: str, target_resolution: Tuple[int, int] = (640, 480), target_fps: float = 30.0):
        self.db_path = db_path
        self.target_width, self.target_height = target_resolution
        self.target_fps = target_fps
        self.cfg = load_config()
        
    def _perceptual_hash(self, frame: np.ndarray) -> str:
        """Simple perceptual hash for duplicate frame detection (8x8 average hash)."""
        import hashlib
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (8, 8))
        avg = small.mean()
        bits = (small > avg).flatten()
        return hashlib.md5(np.packbits(bits).tobytes()).hexdigest()

    def analyze(
        self,
        video_path: str,
        patient_id: int,
        expected_exercise_key: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> int:
        """Run complete offline video analysis, save annotated video, and log to SQLite database."""
        logger.info("Initializing offline analysis for video: %s", video_path)
        
        # 1. Metadata and Validation
        if progress_callback:
            progress_callback(0.01, "Validating video file and extracting metadata...")
        
        metadata = get_video_metadata(video_path)
        if not metadata["is_valid"]:
            raise ValueError(metadata.get("error_msg", "Corrupted or invalid video file."))
        
        rotation = metadata["rotation_degrees"]
        source_fps = metadata["fps"]
        total_frames = metadata["frame_count"]
        
        # Calculate stride for FPS normalization
        # Sample frames to target FPS
        fps_stride = max(1, round(source_fps / self.target_fps))
        
        # 2. Setup output video path
        output_dir = Path("outputs/processed_videos")
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time())
        video_filename = Path(video_path).stem
        processed_video_path = output_dir / f"annotated_{video_filename}_{timestamp}.mp4"
        
        # 3. Initialize components
        pose_estimator = PoseEstimator(
            static_image_mode=False,
            model_complexity=self.cfg.datasets.landmark_extraction.model_complexity,
            min_detection_confidence=self.cfg.datasets.landmark_extraction.min_detection_confidence,
            min_tracking_confidence=self.cfg.datasets.landmark_extraction.min_tracking_confidence,
        )
        pose_estimator.open()
        
        landmark_extractor = LandmarkExtractor()
        smoother = LandmarkSmoother()
        gap_handler = PoseGapHandler(max_hold_frames=10)
        view_detector = ViewDetector()
        confidence_smoother = ConfidenceSmoother()
        
        exercise_library = ExerciseLibrary(self.cfg.exercises)
        # Using self.target_fps since we will feed normalized/sampled frame rates
        movement_analyzer = MovementAnalyzer(exercise_library, fps=self.target_fps, exercises_cfg=self.cfg.exercises)
        if expected_exercise_key:
            movement_analyzer.set_expected_exercise(expected_exercise_key)
            
        skeleton_renderer = SkeletonRenderer(self.cfg.visualization)
        ghost_renderer = GhostSkeletonRenderer(self.cfg.visualization)
        arrow_renderer = CorrectionArrowRenderer(self.cfg.visualization)
        hud = HUDOverlay(self.cfg.visualization)
        
        # 4. Initialize session logger
        logger_exercise = "auto-detect"
        if expected_exercise_key:
            logger_exercise = expected_exercise_key.replace("_", " ").title()
            
        session_logger = SessionLogger(self.db_path, patient_id=patient_id, exercise_name=logger_exercise)
        session_id = session_logger.start_session()
        
        # 5. Open Video Readers & Writers
        cap = cv2.VideoCapture(video_path)
        
        # VideoWriter configuration with H264 fallback to MP4V
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(str(processed_video_path), fourcc, self.target_fps, (self.target_width, self.target_height))
        if not writer.isOpened():
            # Fall back to standard MP4V
            logger.warning("Could not open VideoWriter with H264 codec (avc1), falling back to MP4V (mp4v).")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(processed_video_path), fourcc, self.target_fps, (self.target_width, self.target_height))
            
        if not writer.isOpened():
            cap.release()
            pose_estimator.close()
            raise RuntimeError("Could not initialize VideoWriter for saving annotated video.")
            
        stage_timer = StageTimer()
        frame_idx = 0
        processed_count = 0
        landmarks_cache = []
        frame_hashes = []
        
        last_raw_landmarks = None
        last_landmarks_xyz = None
        last_is_held = False
        
        cuda_available = False
        try:
            import torch
            cuda_available = torch.cuda.is_available()
        except ImportError:
            pass
            
        # 6. Read and process loop
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                
                # Check target sampling rate
                if frame_idx % fps_stride != 0:
                    frame_idx += 1
                    continue
                
                if progress_callback and processed_count % 10 == 0:
                    fraction = 0.05 + 0.90 * (frame_idx / total_frames)
                    progress_callback(min(0.95, fraction), f"Analyzing frame {frame_idx}/{total_frames}...")
                
                # Apply rotation if needed
                if rotation == 90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif rotation == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif rotation == 270:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                # Normalize Resolution
                normalized_frame = resize_frame(frame, (self.target_width, self.target_height))
                display_frame = normalized_frame.copy()
                
                # Duplicate Frame Skip check (Perceptual Hash)
                frame_hash = self._perceptual_hash(normalized_frame)
                is_duplicate = False
                if frame_hash in frame_hashes:
                    is_duplicate = True
                else:
                    frame_hashes.append(frame_hash)
                
                with stage_timer.time("pose_estimation"):
                    if is_duplicate and last_raw_landmarks is not None:
                        # Skip re-estimating pose for duplicate frames, copy previous result
                        raw_landmarks = last_raw_landmarks
                        landmarks_xyz = last_landmarks_xyz
                        is_held = last_is_held
                        pose_detected = (raw_landmarks is not None)
                    else:
                        pose_result = pose_estimator.process(normalized_frame)
                        pose_detected = pose_result.detected
                        raw_landmarks = pose_result.landmarks_xyz if pose_detected else None
                        landmarks_xyz, is_held = gap_handler.update(raw_landmarks)
                        
                        last_raw_landmarks = raw_landmarks
                        last_landmarks_xyz = landmarks_xyz
                        last_is_held = is_held
                
                hud_state = HUDState(
                    fps=self.target_fps,
                    stage_timings=stage_timer.summary(),
                    cuda_available=cuda_available,
                    session_elapsed_seconds=processed_count / self.target_fps
                )
                
                if landmarks_xyz is None:
                    # Pose lost
                    hud_state.model_confidence = 0.0
                    display_frame = hud.draw(display_frame, hud_state)
                    writer.write(display_frame)
                    landmarks_cache.append(None)
                    
                    # Log an empty frame
                    empty_result = MovementAnalysisResult(
                        exercise_key=None, exercise_display_name=None,
                        exercise_recognition_confidence=0.0, phase=None,
                        expected_angle_deg=None, actual_angle_deg=None,
                        progress_fraction=0.0, completion_percentage=0.0,
                        movement_quality=1.0, overall_confidence=0.0,
                        rep_count=0, errors=[]
                    )
                    session_logger.log_frame(empty_result, {}, fps=self.target_fps, cuda_available=cuda_available, session_elapsed_seconds=hud_state.session_elapsed_seconds)
                else:
                    with stage_timer.time("smoothing_and_features"):
                        smoothed = smoother.smooth(landmarks_xyz)
                        raw_confidence = landmark_extractor.overall_confidence(pose_result) if (pose_detected and not is_held) else 0.5
                        confidence = confidence_smoother.smooth(raw_confidence)
                        view = view_detector.detect(smoothed)
                        angles = compute_all_joint_angles(smoothed)
                    
                    with stage_timer.time("movement_analysis"):
                        # We pass calibration baseline as None for offline videos, or can load from previous sessions
                        result = movement_analyzer.analyze_frame(
                            smoothed, angles, pose_confidence=confidence, view=view, calibration=None
                        )
                        
                    # Save serialized landmarks to cache
                    landmarks_cache.append([
                        {
                            "x": float(smoothed[i][0]),
                            "y": float(smoothed[i][1]),
                            "z": float(smoothed[i][2]),
                            "visibility": float(pose_result.landmarks_visibility[i]) if (pose_detected and pose_result.landmarks_visibility is not None) else 1.0
                        }
                        for i in range(33)
                    ])
                    
                    # Log frame into session_frames
                    session_logger.log_frame(
                        result, angles, fps=self.target_fps, cuda_available=cuda_available,
                        session_elapsed_seconds=hud_state.session_elapsed_seconds
                    )
                    
                    # Draw visualizations onto frame
                    display_frame = skeleton_renderer.draw(display_frame, smoothed, errors=result.errors)
                    
                    definition = movement_analyzer.library.get(result.exercise_key) if result.exercise_key else None
                    if definition is not None:
                        ideal = generate_ideal_pose(smoothed, definition, side="left")
                        if ideal is not None:
                            display_frame = ghost_renderer.draw(display_frame, ideal)
                            display_frame = arrow_renderer.from_ideal_pose(
                                display_frame, smoothed, ideal, joint_names=["left_wrist", "left_elbow"]
                            )
                    
                    hud_state.exercise_display_name = result.exercise_display_name
                    hud_state.phase = result.phase.value if result.phase else None
                    hud_state.rep_count = result.rep_count
                    hud_state.movement_quality = result.movement_quality
                    hud_state.model_confidence = result.overall_confidence
                    hud_state.view = view.view.value if view else None
                    
                    display_frame = hud.draw(display_frame, hud_state)
                    writer.write(display_frame)
                    
                processed_count += 1
                frame_idx += 1
                
        finally:
            cap.release()
            writer.release()
            pose_estimator.close()
            session_logger.end_session()
            
        # 7. Post-processing calculations and video metadata updates
        if progress_callback:
            progress_callback(0.96, "Saving session landmarks cache...")
            
        # Write cached landmarks json into the database for future replays
        with get_connection(self.db_path) as conn:
            # Get frame ids in order of insertion for this session
            frame_rows = conn.execute(
                "SELECT frame_id FROM session_frames WHERE session_id = ? ORDER BY frame_id",
                (session_id,)
            ).fetchall()
            
            for idx, row in enumerate(frame_rows):
                if idx < len(landmarks_cache) and landmarks_cache[idx] is not None:
                    conn.execute(
                        "UPDATE session_frames SET landmarks_json = ? WHERE frame_id = ?",
                        (json.dumps(landmarks_cache[idx]), row["frame_id"])
                    )
        
        if progress_callback:
            progress_callback(0.98, "Computing recovery scores and clinical metrics...")
            
        # Save joint metrics & session clinical scores
        analytics_engine = RecoveryAnalyticsEngine(self.db_path)
        # Compute joint metrics. Target angles can be extracted from library config based on detected exercise.
        target_angles = {}
        if expected_exercise_key:
            definition = exercise_library.get(expected_exercise_key)
            if definition:
                target_angles = {definition.primary_angle: definition.target_deg}
        analytics_engine.save_joint_metrics(session_id, target_angles=target_angles)
        analytics_engine.save_session_scores(session_id, patient_id=patient_id)
        
        # Save video path paths in the sessions table
        with get_connection(self.db_path) as conn:
            # Query to find the detected exercise name to update sessions
            session_info = conn.execute("SELECT exercise_name FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
            detected_exercise = session_info["exercise_name"] if session_info else logger_exercise
            
            conn.execute(
                "UPDATE sessions SET original_video_path = ?, processed_video_path = ?, exercise_name = ? WHERE session_id = ?",
                (str(video_path), str(processed_video_path), detected_exercise, session_id)
            )
            
        if progress_callback:
            progress_callback(1.0, "Analysis complete!")
            
        logger.info("Offline video analysis completed. Session ID: %d", session_id)
        return session_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StrokeRehabAI Offline Video Analyzer CLI.")
    parser.add_argument("video_path", help="Path to input video file.")
    parser.add_argument("--patient-id", type=int, required=True, help="Patient ID to log this session under.")
    parser.add_argument("--exercise-key", default=None, help="Force expected exercise key (e.g. elbow_flexion).")
    parser.add_argument("--db-path", default="data/strokerehab.db", help="Path to SQLite database.")
    args = parser.parse_args()
    
    analyzer = OfflineVideoAnalyzer(args.db_path)
    session_id = analyzer.analyze(
        args.video_path,
        args.patient_id,
        expected_exercise_key=args.exercise_key,
        progress_callback=lambda f, msg: print(f"[{f*100:.0f}%] {msg}")
    )
    print(f"Session successfully logged with ID: {session_id}")
