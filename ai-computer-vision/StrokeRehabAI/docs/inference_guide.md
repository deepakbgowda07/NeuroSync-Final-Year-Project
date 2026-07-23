# Real-Time Inference Guide

## Quick start

```powershell
python -m inference.realtime_pipeline
python -m inference.realtime_pipeline --patient-id 3
python -m inference.realtime_pipeline --skip-calibration
```

Press `q` in the camera window to stop, or `Ctrl+C` in the terminal.
From the dashboard, use the **Live Session** page — it launches this
same command as a subprocess and shows live-updating stats pulled from
the session database.

## What happens

1. **Camera streaming** (`camera/camera_manager.py`): a background
   thread continuously reads frames into a small bounded `FrameQueue`
   (`camera/frame_queue.py`), so the main loop always processes the
   *newest* frame rather than falling behind. If the camera drops out
   mid-session, it automatically reconnects
   (`camera.auto_recovery` in `configs/camera.yaml`).
2. **Calibration** (`inference/calibration.py`, skippable with
   `--skip-calibration`): a ~12 second "stand still" session
   (`calibration.duration_seconds` in `configs/calibration.yaml`)
   estimates the patient's neutral pose, shoulder width, arm length,
   torso length, and baseline joint angles — used to personalize later
   error checks rather than comparing against a population average.
3. **Pose estimation** (`mediapipe_pipeline/pose_estimator.py`):
   MediaPipe BlazePose extracts 33 landmarks + visibility per frame.
4. **View detection** (`mediapipe_pipeline/view_detector.py`):
   automatically classifies front / left-side / right-side view from
   shoulder geometry — the patient never selects a view manually.
5. **Gap handling + smoothing**
   (`mediapipe_pipeline/pose_smoothing.py`): a brief missed detection
   holds the last known pose (up to 10 frames); beyond that the pose is
   declared genuinely lost rather than silently analyzed as stale data.
   An EMA smoother reduces frame-to-frame landmark jitter.
6. **Feature extraction** (`utils/joint_angles.py`,
   `feature_extraction/`): joint angles (including abduction and
   rotation/pronation *proxy* angles — see the caveat below), velocity,
   symmetry, trunk lean, shoulder elevation.
7. **Movement analysis** (`inference/movement_analyzer.py`), which
   composes:
   - **Exercise recognition** (`inference/exercise_recognizer.py`):
     rule-based scoring against the 10-exercise library
     (`configs/exercises.yaml`) — no manual exercise selection.
   - **Phase detection** (`inference/phase_detector.py`): neutral ->
     moving-to-target -> peak -> returning -> neutral.
   - **Rep counting** (`inference/rep_tracker.py`): counts only
     complete neutral->peak->neutral cycles, so accidental movements
     and incomplete reps are correctly excluded/flagged rather than
     miscounted.
   - **Error detection** (`inference/error_detector.py`): the full
     required error set (incorrect angle, insufficient/excessive ROM,
     incomplete rep, fast/jerky movement, trunk compensation, shoulder
     hiking, body lean, poor alignment, incorrect sequence, asymmetry,
     late movement, premature return).
8. **Feedback generation** (`inference/feedback_engine.py`): natural
   -language messages — never a bare "Wrong"/"Correct".
9. **Visualization** (`visualization/`): green/red skeleton
   (`skeleton_renderer.py`, red at joints implicated by an active
   error), a translucent ghost skeleton at the target pose
   (`ghost_skeleton.py` + `ideal_pose.py`), correction arrows
   (`correction_arrows.py`), and a HUD (`hud_overlay.py`) showing
   exercise, phase, rep count, quality, timer, FPS, CUDA status, and
   confidence.
10. **Dashboard logging** (`inference/session_logger.py`): every frame,
    rep, and error event streams into the SQLite database in real time
    (`dashboard/db.py`), so the dashboard's Exercise History / Recovery
    Analytics pages reflect the session without a separate export step.

## Supported exercises

Exactly ten, defined in `configs/exercises.yaml` /
`inference/exercise_library.py`:

Shoulder Flexion, Shoulder Abduction, Elbow Flexion, Elbow Extension,
Forearm Pronation, Forearm Supination, Shoulder External Rotation,
Shoulder Internal Rotation, Hand-to-Mouth Reach, Hand-to-Head Reach.

## Important limitation: forearm/shoulder rotation angles

MediaPipe **Pose** (used throughout this project) provides only 3
sparse points per hand (thumb, index, pinky tips) — not the full
21-point MediaPipe **Hands** topology. Forearm pronation/supination and
shoulder internal/external rotation therefore use *proxy* angles
(`utils.joint_angles.forearm_rotation_proxy_deg` /
`shoulder_rotation_proxy_deg`) that are directionally useful but not a
clinically validated goniometer replacement. Integrating MediaPipe
Hands is the natural next step for precise tracking of these four
exercises (see `docs/architecture.md`).

## Performance

Target: 25-35 FPS at 720p, <50ms inference latency. In practice:

- `camera/fps_controller.py` measures achieved FPS and adaptively caps
  the target down (never below `camera.fps_min`) if the pipeline can't
  keep up, then steps back up once headroom returns.
- `utils/timers.StageTimer` (surfaced in the HUD) breaks down
  per-stage latency (pose estimation, smoothing/features, movement
  analysis) so a slow stage is easy to identify.
- CUDA is used automatically when available
  (`configs/gpu.yaml -> use_cuda_if_available`); the HUD shows live
  CUDA status.

## Personalization / calibration details

`inference/calibration.py`'s `CalibrationSession` only *accepts* a
calibration once the buffered joint angles are stable (std-dev below
`calibration.stability_std_threshold_deg`) — an unstable session
(patient still moving, poor detection) is rejected with a clear reason
rather than silently calibrating against noisy data, and the pipeline
proceeds without a personalized baseline in that case (falling back to
population-average thresholds).

## Testing without a webcam

Every module above (except the outermost camera-thread + `cv2.imshow`
loop) can be exercised with synthetic data — see
`tests/test_realtime_pipeline_integration.py`, which drives
`RealtimeInferencePipeline._process_frame()` directly against synthetic
frames, and the focused unit tests for each component
(`tests/test_rep_tracker.py`, `test_calibration.py`,
`test_exercise_recognizer.py`, `test_error_detector.py`,
`test_view_detector.py`, `test_camera_realtime.py`,
`test_skeleton_renderer.py`, `test_session_logger.py`).
