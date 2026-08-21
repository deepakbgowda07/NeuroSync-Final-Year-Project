# Frequently Asked Questions

**Is this a medical device / diagnostic tool?**
No. StrokeRehabAI is a movement-quality assessment and progress-tracking
aid for physiotherapist-supervised exercise practice. It does not
diagnose conditions or make treatment decisions — see the explicit
non-diagnostic note attached to every recovery-tracking report
(`dashboard/recovery_tracking.py`) and personalization plan
(`dashboard/personalization_engine.py`).

**Does it work with just a laptop webcam?**
Yes — that's the target hardware profile (Windows 11, RTX 3050 Laptop
GPU, integrated webcam). It also runs on CPU (slower) if no CUDA GPU
is available.

**Do I need an internet connection to use it?**
No, once installed. All processing (pose estimation, model inference,
dashboard) runs entirely locally; patient data never leaves the
machine.

**How does it know which exercise I'm doing?**
`inference/exercise_recognizer.py` automatically classifies the
current exercise from the live joint-angle trajectory against the 10
supported exercise definitions — there's no manual exercise selector.

**Why do some exercises say their measurement is an "approximation"?**
Forearm pronation/supination and shoulder internal/external rotation
need fine hand-orientation detail that MediaPipe *Pose* doesn't fully
provide (it has only 3 sparse points per hand, vs. the 21-point
MediaPipe *Hands* model). The angles for those exercises
(`utils/joint_angles.py:forearm_rotation_proxy_deg` /
`shoulder_rotation_proxy_deg`) are documented engineering
approximations, not clinical-grade goniometer measurements.

**Can it be used for exercises other than the 10 listed?**
Not out of the box — recognition, rep counting, and error detection
are all defined per-exercise in `configs/exercises.yaml` and the
knowledge base (`knowledge_base/exercises_knowledge.yaml`). Adding a
new exercise means adding entries to both, plus (if it needs a new
joint-angle definition) extending `utils/joint_angles.py`.

**How is the "Recovery Score" calculated — is it clinically validated?**
It's a transparent, documented blend of measured sub-scores (ROM,
smoothness, stability, symmetry, compensation, quality — see
`dashboard/clinical_metrics.py` and `inference/clinical_assessment.py`)
scaled to 0-100. The blend weights are principled engineering
defaults, explicitly flagged throughout the code as pending validation
against real clinician-rated outcomes — not yet clinically validated.

**Why does the system say things like "shoulder angle remained below
target by 15 degrees" instead of just "wrong"?**
By design (`inference/feedback_engine.py`,
`inference/explainable_ai.py`) — every piece of feedback must cite a
specific measured condition. Bare "wrong"/"correct" judgments are
explicitly disallowed; this is tested in
`tests/test_explainable_ai.py` and `tests/test_feedback_engine.py`.

**Can I run a demo without a real patient?**
Yes — `python main.py demo --generate-data` populates the dashboard
with a synthetic patient showing a realistic multi-session recovery
trend, so Recovery Analytics/Reports/History all have something to
show. `python main.py demo --mode video --source <file>` runs the live
pipeline against a sample video instead of a webcam.

**What happens if the camera disconnects mid-session?**
`camera/camera_manager.py`'s background capture thread automatically
attempts reconnection after a configurable number of consecutive
failures (`configs/camera.yaml -> auto_recovery`) rather than crashing
the session.

**What happens if pose detection briefly fails (occlusion, motion blur)?**
`mediapipe_pipeline/pose_smoothing.py:PoseGapHandler` holds the last
known-good pose for a short window; beyond that window it declares the
pose genuinely lost (rather than silently analyzing indefinitely-stale
data) and the HUD reflects zero confidence until detection resumes.

**Is patient data encrypted?**
No — the current SQLite database is unencrypted local storage,
appropriate for a single-clinic research/demo deployment as described
in `docs/deployment_guide.md`. Encrypting the database at rest, and/or
adding authentication to the dashboard, is flagged as future work for
any deployment beyond a single trusted workstation.

**Can I export my data?**
Yes — Reports supports PDF, Excel, CSV, and JSON, each including
patient info, exercise summary, recovery metrics, joint/ROM/
compensation analysis, recommendations, and session history
(`dashboard/report_generator.py`). Model checkpoints can also be
exported to TorchScript/ONNX (`models/export_utils.py`).

**Where do I report a bug or ask something not covered here?**
See `docs/contribution_guide.md` for the development workflow, and
`docs/troubleshooting_guide.md` for common issues with fixes.
