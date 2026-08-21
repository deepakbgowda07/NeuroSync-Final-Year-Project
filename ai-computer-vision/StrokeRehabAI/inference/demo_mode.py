"""
demo_mode.py
==============
Demonstration mode: lets the system be shown end-to-end without a real
patient present. Supports three source modes:

    "webcam"    - the standard live camera pipeline (inference/realtime_pipeline.py)
    "video"     - a pre-recorded sample video file, run through the same
                   pose -> analysis -> feedback pipeline as a live session
    "session"   - replays a previously recorded session's stored landmark
                   data from the database (no camera or video needed at all)

Also provides `generate_demo_dataset()`, which populates the dashboard
database with a synthetic patient and several sessions showing a
plausible recovery trend — so Recovery Analytics, Reports, and Exercise
History all have something to display in a demo with zero real data.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

from dashboard.db import get_connection, init_db
from utils.logger import get_logger

logger = get_logger(__name__)


class DemoSourceMode(str, Enum):
    WEBCAM = "webcam"
    VIDEO = "video"
    SESSION = "session"


class DemoModeRunner:
    """Dispatches to the appropriate real-time pipeline configuration
    for the requested demo source mode."""

    def __init__(self, cfg=None):
        from configs.config_loader import load_config

        self.cfg = cfg or load_config()

    def run(self, mode: DemoSourceMode, source_path: Optional[str] = None, skip_calibration: bool = True) -> None:
        if mode == DemoSourceMode.WEBCAM:
            self._run_webcam(skip_calibration)
        elif mode == DemoSourceMode.VIDEO:
            self._run_video(source_path, skip_calibration)
        elif mode == DemoSourceMode.SESSION:
            self._run_session_replay(source_path)
        else:
            raise ValueError(f"Unknown demo source mode: {mode}")

    def _run_webcam(self, skip_calibration: bool) -> None:
        from inference.realtime_pipeline import RealtimeInferencePipeline

        logger.info("Starting demo mode: live webcam.")
        pipeline = RealtimeInferencePipeline(cfg=self.cfg, skip_calibration=skip_calibration)
        pipeline.run()

    def _run_video(self, video_path: Optional[str], skip_calibration: bool) -> None:
        if not video_path:
            raise ValueError("A sample video path is required for --mode video.")
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Demo video not found: {video_path}")

        from inference.realtime_pipeline import RealtimeInferencePipeline

        logger.info("Starting demo mode: sample video (%s).", video_path)
        # Temporarily point the camera source at the video file — the
        # existing CameraManager already treats a video-extension source
        # as a VideoLoader rather than a live Webcam (see camera/camera_manager.py).
        demo_cfg = self.cfg
        demo_cfg.camera.source = video_path
        pipeline = RealtimeInferencePipeline(cfg=demo_cfg, skip_calibration=skip_calibration)
        pipeline.run()

    def _run_session_replay(self, session_id: Optional[str]) -> None:
        """Replays a stored session's data purely from the database — no
        camera or video required. Delegates to the dashboard's session
        detail view logic and prints an explainable-AI style summary per
        stored frame, demonstrating the analysis pipeline's output
        without needing to re-run pose estimation."""
        from dashboard.session_manager import SessionManager

        if not session_id:
            raise ValueError("A session_id is required for --mode session.")

        manager = SessionManager(self.cfg.dashboard.database_path)
        detail = manager.get_session_detail(int(session_id))

        logger.info(
            "Replaying stored session %s: %d frames, %d successful reps, %d incorrect reps.",
            session_id, len(detail["frames"]), detail["successful_repetitions"], detail["incorrect_repetitions"],
        )
        for frame in detail["frames"][::10]:  # sample every 10th frame for a readable console demo
            print(
                f"[{frame['phase'] or '-'}] quality={frame.get('movement_quality') or 0:.2f} "
                f"rom={frame.get('rom_deg') or 0:.0f}\u00b0"
            )


def generate_demo_dataset(db_path: str, num_sessions: int = 8) -> int:
    """Populates the dashboard database with one synthetic demo patient
    and several sessions showing a plausible gradual-improvement
    recovery trend, so the dashboard has something to display without
    any real patient data. Returns the created patient_id.

    Safe to call repeatedly — creates a new demo patient each time
    (named uniquely by timestamp) rather than silently overwriting
    prior demo data.
    """
    init_db(db_path)
    rng = np.random.default_rng(42)

    demo_patient_name = f"Demo Patient ({datetime.now().strftime('%Y-%m-%d %H:%M')})"

    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """INSERT INTO patients
               (full_name, date_of_birth, gender, affected_side, diagnosis, physiotherapist, treatment_start_date, notes)
               VALUES (?, '1968-03-12', 'Female', 'Left', 'Ischemic stroke (right MCA territory)',
                       'Dr. A. Demo', ?, 'Synthetic demo patient — for demonstration purposes only, not a real patient.')""",
            (demo_patient_name, (datetime.now() - timedelta(days=num_sessions * 3)).strftime("%Y-%m-%d")),
        )
        patient_id = cursor.lastrowid

        conn.execute(
            "INSERT INTO rehabilitation_plans (patient_id, plan_name, exercises_json, sessions_per_week, notes) "
            "VALUES (?, 'Demo Upper-Limb Plan', ?, 3, 'Synthetic demo plan.')",
            (patient_id, json.dumps(["elbow_flexion", "shoulder_flexion"])),
        )

        for i in range(num_sessions):
            session_id_row = conn.execute(
                "INSERT INTO sessions (patient_id, exercise_name, started_at, duration_seconds, total_reps) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    patient_id,
                    "Elbow Flexion" if i % 2 == 0 else "Shoulder Flexion",
                    (datetime.now() - timedelta(days=(num_sessions - i) * 3)).isoformat(),
                    120.0 + rng.normal(0, 15),
                    6 + i // 2,
                ),
            )
            session_id = session_id_row.lastrowid

            # Quality gradually improves across sessions, with realistic noise.
            base_quality = 0.45 + (i / max(1, num_sessions - 1)) * 0.4
            neutral_angle, target_angle = (170, 45) if i % 2 == 0 else (15, 150)

            num_frames = 60
            for f in range(num_frames):
                t = f / num_frames
                angle = neutral_angle + (target_angle - neutral_angle) * np.sin(t * np.pi)
                quality = float(np.clip(base_quality + rng.normal(0, 0.08), 0.0, 1.0))
                conn.execute(
                    "INSERT INTO session_frames (session_id, movement_quality, rom_deg, phase, joint_angles_json) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (session_id, quality, angle, "moving_to_target",
                     json.dumps({"left_elbow_angle": angle if i % 2 == 0 else 170.0,
                                 "left_shoulder_angle": angle if i % 2 != 0 else 20.0})),
                )

            num_reps = 6 + i // 2
            for r in range(num_reps):
                completed = rng.random() < (0.5 + i / max(1, num_sessions - 1) * 0.45)
                conn.execute(
                    "INSERT INTO session_reps (session_id, rep_number, completed, peak_progress_fraction, duration_frames) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (session_id, r + 1, int(completed), float(rng.uniform(0.7, 1.0)), int(rng.integers(20, 40))),
                )

            # Early sessions show more compensation; later sessions show less.
            num_compensation_events = max(0, int(rng.poisson(lam=max(0.5, 4 - i * 0.4))))
            for _ in range(num_compensation_events):
                conn.execute(
                    "INSERT INTO session_events (session_id, event_type, error_type, severity, timestamp) "
                    "VALUES (?, 'error', 'trunk_compensation', 'moderate', ?)",
                    (session_id, datetime.now().timestamp()),
                )

    logger.info("Generated demo dataset: patient_id=%d, %d sessions.", patient_id, num_sessions)
    return patient_id
