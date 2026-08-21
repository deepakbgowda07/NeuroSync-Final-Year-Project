# Inference Guide — Real-Time Rehabilitation Assessment Engine

## Quick start

```powershell
python -m inference.realtime_pipeline
python -m inference.realtime_pipeline --patient-id 3
python -m inference.realtime_pipeline --skip-calibration
```

Or launch from the dashboard's **Live Session** page (`streamlit run
dashboard/app.py`), which runs the same command as a subprocess and
polls live results from the database.

Press `q` in the camera window, or `Ctrl+C` in the terminal, to stop.
Calibration (see below) runs automatically first unless
`--skip-calibration` is passed.

## What happens each frame

```
CameraManager (background capture thread, frame queue, auto-recovery)
        │
        ▼
PoseEstimator (MediaPipe BlazePose: 33 landmarks, visibility, 3D coords)
        │
        ▼
PoseGapHandler (holds last-known pose through brief detection gaps,
                declares "lost" beyond configs/camera.yaml's hold limit)
        │
        ▼
LandmarkSmoother (EMA smoothing) + ViewDetector (front/left/right,
                  majority-voted) + ConfidenceSmoother
        │
        ▼
compute_all_joint_angles() → all tracked angles, incl. abduction and
                              forearm/shoulder rotation proxies
        │
        ▼
MovementAnalyzer
   ├── ExerciseRecognizer   (auto-detects which of the 10 exercises)
   ├── PhaseDetector          (neutral / moving / peak / returning, per exercise)
   ├── RepTracker              (intelligent rep counting)
   └── ErrorDetector            (14 error types, calibration-aware)
        │
        ▼
FeedbackEngine (natural-language messages — never bare "wrong"/"correct")
        │
        ├──▶ SkeletonRenderer (green skeleton, red flagged joints)
        ├──▶ GhostSkeletonRenderer + ideal_pose (target-pose overlay)
        ├──▶ CorrectionArrowRenderer (arrows toward the ideal pose)
        ├──▶ HUDOverlay (exercise, phase, reps, quality, timer, FPS, CUDA, confidence)
        └──▶ SessionLogger → dashboard SQLite (session_frames, session_reps, session_events)
```

## Automatic camera view detection

The patient never selects a view. `mediapipe_pipeline/view_detector.py`
classifies front / left-side / right-side purely from shoulder
geometry (x-span and relative depth), majority-voted over a rolling
window so a single noisy frame can't flip the detected view mid-rep.
Each exercise's `required_view` (see `configs/exercises.yaml`) feeds
into exercise recognition scoring — exercises expecting a view the
patient isn't currently in score lower, without ever blocking the
patient or asking them to change anything manually.

## Automatic exercise recognition

`inference/exercise_recognizer.py` scores all 10 supported exercises
against a rolling buffer of the patient's own joint-angle trajectory —
no manual exercise selection, no per-exercise training data required.
It is intentionally a transparent, tunable rule-based recognizer; the
trained ST-GCN classifier (`models/stgcn_model.py` — see
`docs/training_guide.md`) is a natural drop-in or ensemble replacement
once labeled session data exists.

## Supported exercises

Shoulder Flexion, Shoulder Abduction, Elbow Flexion, Elbow Extension,
Forearm Pronation, Forearm Supination, Shoulder External Rotation,
Shoulder Internal Rotation, Hand-to-Mouth Reach, Hand-to-Head Reach —
defined in `configs/exercises.yaml`, loaded via
`inference/exercise_library.py`.

> **Limitation:** MediaPipe *Pose* (not *Hands*) provides only 3 sparse
> hand landmarks (thumb/index/pinky tips) per side. Forearm
> pronation/supination and shoulder internal/external rotation are
> approximated from these (`utils/joint_angles.py:forearm_rotation_proxy_deg`
> / `shoulder_rotation_proxy_deg`) — directionally useful, not a
> clinically validated goniometer replacement. See those functions'
> docstrings; integrating MediaPipe Hands is the natural next step.

## Calibration (personalization)

Runs automatically before the exercise loop (`inference/calibration.py`),
targeting under 15 seconds (`configs/calibration.yaml`, default 12s +
prompts). The patient stands in a neutral pose while the system
estimates shoulder width, arm lengths, torso length, and baseline joint
angles, rejecting the session (with a clear reason) if too few frames
were captured or the readings weren't stable — never silently
calibrating against noisy data.

## Error detection

14 error types are detected in real time
(`inference/error_detector.py`): incorrect joint angle, insufficient/
excessive ROM, incomplete repetition, fast/jerky movement, trunk
compensation, shoulder hiking, body lean, poor alignment, incorrect
exercise sequence, asymmetrical motion, late movement, and premature
return. Each check is independently testable (see
`tests/test_error_detector.py`) and driven by `configs/exercises.yaml`
thresholds plus the patient's own calibration baseline where available.

## Rep counting

`inference/rep_tracker.py` counts a repetition only on a complete
neutral → peak → neutral cycle (driven by `PhaseDetector`'s phase
stream), which naturally ignores incidental small movements (never
reach peak), tolerates pauses at any phase (state simply holds), and
flags — rather than silently counts — incomplete repetitions and
premature returns.

## Smoothing

Four independent smoothing layers, so different signal types get
appropriately-scaled damping (`inference/smoothing.py` +
`mediapipe_pipeline/pose_smoothing.py`): landmark EMA, per-angle EMA,
confidence EMA, and majority-vote prediction smoothing (used
internally by both `ExerciseRecognizer` and `ViewDetector`).

## Performance

Target: 25-35 FPS at 720p, <50ms inference latency. Contributing
design choices:
- Camera capture runs on a background thread with a small (default
  size-2) frame queue, so the main loop always processes the newest
  frame rather than a backlog (`camera/frame_queue.py`).
- `camera/fps_controller.py` adaptively lowers the target FPS (down to
  a configured floor) if the pipeline falls behind, and steps back up
  once headroom returns.
- `utils/timers.py:StageTimer` measures per-stage latency
  (pose estimation, smoothing/features, movement analysis), shown live
  in the HUD.
- CUDA is used automatically wherever available (`utils/gpu_utils.py`);
  CUDA status is shown in the HUD.

## Camera resilience

`camera/camera_manager.py`'s background capture thread automatically
reconnects after `configs/camera.yaml -> auto_recovery.max_consecutive_failures`
consecutive read failures — a webcam briefly held by another process,
or a momentary USB glitch, doesn't require restarting the session.

## Dashboard communication

`inference/session_logger.py` streams every frame's exercise, phase,
joint angles, quality, confidence, ROM, and any active errors into the
dashboard's SQLite database in real time (`session_frames`,
`session_reps`, `session_events` tables — see `dashboard/db.py`), plus
a session summary (duration, total reps, mean quality/confidence) on
completion. The dashboard's **Live Session** page polls this while a
session is running.

## Troubleshooting

- **"No frame received within timeout"**: check `python main.py
  check-gpu` isn't holding the camera, or increase
  `camera.retry_delay_seconds`.
- **View keeps flipping between front/side**: increase
  `mediapipe_pipeline.view_detector.ViewDetector`'s `smoothing_window`,
  or ensure even lighting so shoulder depth estimates are stable.
- **Reps not counting**: check the HUD's reported `Phase` — if it never
  reaches `peak`, the movement isn't reaching
  `configs/exercises.yaml`'s `near_target_fraction` threshold for that
  exercise; verify camera framing includes the full arm.
