"""
realtime_pipeline.py
=====================
The complete real-time rehabilitation assessment engine: camera capture
-> pose estimation -> view detection -> landmark smoothing/gap-handling
-> feature extraction -> calibration-aware movement analysis (exercise
recognition, phase, rep counting, error detection) -> natural-language
feedback -> visualization (skeleton, ghost pose, correction arrows,
HUD) -> live dashboard logging, running continuously at 25-35 FPS.

Run via:
    python -m inference.realtime_pipeline
    python -m inference.realtime_pipeline --patient-id 3 --skip-calibration
"""

from __future__ import annotations

import argparse
import time
from typing import Optional

from camera.camera_manager import CameraManager
from camera.fps_controller import FPSController
from configs.config_loader import load_config
from inference.calibration import CalibrationSession
from inference.exercise_library import ExerciseLibrary
from inference.feedback_engine import FeedbackEngine
from inference.movement_analyzer import MovementAnalyzer
from inference.session_logger import SessionLogger
from inference.smoothing import ConfidenceSmoother
from mediapipe_pipeline.landmark_extractor import LandmarkExtractor
from mediapipe_pipeline.pose_estimator import PoseEstimator
from mediapipe_pipeline.pose_smoothing import LandmarkSmoother, PoseGapHandler
from mediapipe_pipeline.view_detector import ViewDetector
from utils.joint_angles import compute_all_joint_angles
from utils.logger import get_logger
from utils.timers import FPSCounter, StageTimer
from visualization.correction_arrows import CorrectionArrowRenderer
from visualization.ghost_skeleton import GhostSkeletonRenderer
from visualization.hud_overlay import HUDOverlay, HUDState
from visualization.ideal_pose import generate_ideal_pose
from visualization.skeleton_renderer import SkeletonRenderer

logger = get_logger(__name__)


class RealtimeInferencePipeline:
    """Orchestrates the full live camera -> feedback -> dashboard loop."""

    def __init__(self, cfg=None, patient_id: Optional[int] = None, skip_calibration: bool = False):
        self.cfg = cfg or load_config()
        self.patient_id = patient_id
        self.skip_calibration = skip_calibration

        self.camera_manager = CameraManager.from_config(self.cfg.camera)
        self.pose_estimator = PoseEstimator(
            model_complexity=self.cfg.datasets.landmark_extraction.model_complexity,
            min_detection_confidence=self.cfg.datasets.landmark_extraction.min_detection_confidence,
            min_tracking_confidence=self.cfg.datasets.landmark_extraction.min_tracking_confidence,
        )
        self.landmark_extractor = LandmarkExtractor()
        self.smoother = LandmarkSmoother()
        self.gap_handler = PoseGapHandler(max_hold_frames=10)
        self.view_detector = ViewDetector()
        self.confidence_smoother = ConfidenceSmoother()

        self.exercise_library = ExerciseLibrary(self.cfg.exercises)
        self.movement_analyzer = MovementAnalyzer(self.exercise_library, fps=self.cfg.camera.fps_target, exercises_cfg=self.cfg.exercises)
        self.feedback_engine = FeedbackEngine()

        self.skeleton_renderer = SkeletonRenderer(self.cfg.visualization)
        self.ghost_renderer = GhostSkeletonRenderer(self.cfg.visualization)
        self.arrow_renderer = CorrectionArrowRenderer(self.cfg.visualization)
        self.hud = HUDOverlay(self.cfg.visualization)

        self.fps_counter = FPSCounter()
        self.fps_controller = FPSController(target_fps=self.cfg.camera.fps_target, min_fps=self.cfg.camera.fps_min)
        self.stage_timer = StageTimer()

        self.session_logger = SessionLogger(self.cfg.dashboard.database_path, patient_id=patient_id)
        self._cuda_available = self._check_cuda()
        self._calibration_profile = None
        self._session_start_time: Optional[float] = None

    @staticmethod
    def _check_cuda() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        self.camera_manager.start_streaming()
        self.pose_estimator.open()
        self.session_logger.start_session()
        self._session_start_time = time.time()

        logger.info("Real-time inference engine starting (CUDA=%s). Press 'q' to stop.", self._cuda_available)

        try:
            import cv2

            if not self.skip_calibration:
                self._run_calibration()

            while True:
                self.fps_controller.frame_start()
                timestamped = self.camera_manager.read_latest_frame(timeout=1.0)
                if timestamped is None:
                    logger.warning("No frame received within timeout; camera may be recovering.")
                    continue

                display_frame = self._process_frame(timestamped.frame)

                cv2.imshow("StrokeRehabAI - Live Session", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                self.fps_controller.frame_end()

        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            self._shutdown()

    def _run_calibration(self) -> None:
        import cv2

        calib_cfg = self.cfg.calibration
        session = CalibrationSession(
            duration_seconds=calib_cfg.duration_seconds,
            min_frames_required=calib_cfg.min_frames_required,
            stability_std_threshold_deg=calib_cfg.stability_std_threshold_deg,
            required_landmarks=list(calib_cfg.required_landmarks),
        )
        session.start()
        logger.info("Calibration started: please stand still in a neutral pose.")

        while not session.is_complete:
            timestamped = self.camera_manager.read_latest_frame(timeout=1.0)
            if timestamped is None:
                continue

            pose_result = self.pose_estimator.process(timestamped.frame)
            frame = timestamped.frame.copy()

            if pose_result.detected:
                session.add_frame(pose_result.landmarks_xyz)
                frame = self.skeleton_renderer.draw(frame, pose_result.landmarks_xyz)

            cv2.putText(
                frame, f"Calibrating... {session.progress_fraction * 100:.0f}%",
                (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA,
            )
            cv2.imshow("StrokeRehabAI - Live Session", frame)
            cv2.waitKey(1)

        try:
            self._calibration_profile = session.finalize()
            self.session_logger.log_calibration(self._calibration_profile.to_dict())
            logger.info("Calibration accepted.")
        except RuntimeError as exc:
            logger.warning("Calibration did not meet quality requirements (%s); proceeding without a baseline.", exc)
            self._calibration_profile = None

    def _process_frame(self, frame) -> "object":
        display_frame = frame.copy()

        with self.stage_timer.time("pose_estimation"):
            pose_result = self.pose_estimator.process(frame)

        raw_landmarks = pose_result.landmarks_xyz if pose_result.detected else None
        landmarks_xyz, is_held = self.gap_handler.update(raw_landmarks)

        fps = self.fps_counter.tick()
        hud_state = HUDState(
            fps=fps, stage_timings=self.stage_timer.summary(), cuda_available=self._cuda_available,
            session_elapsed_seconds=time.time() - self._session_start_time if self._session_start_time else 0.0,
        )

        if landmarks_xyz is None:
            hud_state.model_confidence = 0.0
            return self.hud.draw(display_frame, hud_state)

        with self.stage_timer.time("smoothing_and_features"):
            smoothed = self.smoother.smooth(landmarks_xyz)
            raw_confidence = self.landmark_extractor.overall_confidence(pose_result) if not is_held else 0.5
            confidence = self.confidence_smoother.smooth(raw_confidence)
            view = self.view_detector.detect(smoothed)
            angles = compute_all_joint_angles(smoothed)

        with self.stage_timer.time("movement_analysis"):
            result = self.movement_analyzer.analyze_frame(
                smoothed, angles, pose_confidence=confidence, view=view, calibration=self._calibration_profile,
            )

        self.session_logger.log_frame(result, angles)

        for message in self.feedback_engine.generate(result, self._active_definition(result)):
            logger.info("[%s] %s", message.severity.upper(), message.text)

        display_frame = self.skeleton_renderer.draw(display_frame, smoothed, errors=result.errors)

        definition = self._active_definition(result)
        if definition is not None:
            ideal = generate_ideal_pose(smoothed, definition, side="left")
            if ideal is not None:
                display_frame = self.ghost_renderer.draw(display_frame, ideal)
                display_frame = self.arrow_renderer.from_ideal_pose(
                    display_frame, smoothed, ideal, joint_names=["left_wrist", "left_elbow"],
                )

        hud_state.exercise_display_name = result.exercise_display_name
        hud_state.phase = result.phase.value if result.phase else None
        hud_state.rep_count = result.rep_count
        hud_state.movement_quality = result.movement_quality
        hud_state.model_confidence = result.overall_confidence
        hud_state.view = view.view.value if view else None

        return self.hud.draw(display_frame, hud_state)

    def _active_definition(self, result):
        if result.exercise_key is None:
            return None
        return self.exercise_library.get(result.exercise_key)

    def _shutdown(self) -> None:
        self.session_logger.end_session()
        self.pose_estimator.close()
        self.camera_manager.release()
        try:
            import cv2

            cv2.destroyAllWindows()
        except ImportError:
            pass
        logger.info("Real-time inference engine stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the StrokeRehabAI real-time inference engine.")
    parser.add_argument("--patient-id", type=int, default=None, help="Patient ID to associate this session with.")
    parser.add_argument("--skip-calibration", action="store_true", help="Skip the pre-exercise calibration step.")
    args = parser.parse_args()

    pipeline = RealtimeInferencePipeline(patient_id=args.patient_id, skip_calibration=args.skip_calibration)
    pipeline.run()


if __name__ == "__main__":
    main()
