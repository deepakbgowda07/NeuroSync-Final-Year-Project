# Changelog

All notable changes to StrokeRehabAI are documented in this file,
organized by development stage. This project moved from an initial
architecture scaffold through dataset/training, real-time inference,
clinical dashboard, and finally this production-readiness stage.

## [Stage 5] — Final Production-Readiness Stage

### Added

**AI Clinical Assessment**
- `inference/clinical_assessment.py`: per-repetition clinical
  assessment engine computing 8 normalized 0-100 scores (Exercise
  Quality, Range of Motion, Movement Smoothness, Joint Stability,
  Movement Consistency, Compensation Severity, Exercise Completion,
  Overall Rehabilitation Score), integrated directly into
  `inference/movement_analyzer.py` so every completed repetition is
  automatically assessed during a live session.

**Clinical Reasoning & Explainable AI**
- `inference/clinical_reasoning.py`: rule-based physiotherapy
  reasoning engine (ROM deficit, trunk compensation, limited elbow
  extension, fast movement, low smoothness, compensation severity),
  each rule producing a condition, recommendation, and explanation.
- `inference/explainable_ai.py`: structured, patient-facing
  explanations (`Exercise / Confidence / Movement Quality / Reason /
  Recommendation`) generated for every prediction — verified to never
  emit bare "wrong"/"correct" judgments.

**Physiotherapy Knowledge Base**
- `knowledge_base/exercises_knowledge.yaml`: editable clinical
  reference data for all 10 supported exercises — target
  joints/muscles, normal and functional ROM ranges, expected movement
  sequences, compensation patterns, clinical descriptions, and
  correction rules.
- `inference/knowledge_base.py`: loader exposing this data
  programmatically.

**Personalized Rehabilitation & Recovery Tracking**
- `dashboard/personalization_engine.py`: patient-level recommendations
  (exercises to prioritize/reduce, suggested reps/sets, movement
  speed, recovery focus areas) derived from calibration data, session
  history, and recovery trends.
- `dashboard/recovery_tracking.py`: daily/weekly/monthly improvement
  percentages, consistency score, compliance score, recovery velocity,
  and trend classification — every report explicitly states it is a
  progress estimate, not a medical diagnosis or treatment decision.

**Model Optimization & Export**
- `models/export_utils.py`: TorchScript export, ONNX export, dynamic
  quantization, and GPU memory optimization helpers, tuned for the
  RTX 3050 Laptop GPU target.
- Inference benchmarking (via `training/gpu_monitor.py`) validated
  against all three export formats.

**Asynchronous Inference & Frame Buffering**
- `inference/async_inference.py`: thread-safe async inference worker
  decoupling model inference from the capture/render loop, with
  automatic backlog-dropping to always process the newest frame.

**Error Handling**
- `utils/error_handling.py`: a full custom exception hierarchy
  (`CameraError`, `CUDAUnavailableError`, `DatasetError`,
  `ModelWeightsError`, `CorruptedFileError`,
  `InvalidConfigurationError`, `LowMemoryError`) with user-facing
  message formatting, a `handle_gracefully` decorator, and memory-
  availability checks. Wired into `camera/camera_manager.py` (camera
  connection failures) and `utils/checkpoint_utils.py` (missing/
  corrupted model weights).

**Demonstration Mode**
- `inference/demo_mode.py`: run the system against a live webcam, a
  sample video file, or a replay of a previously stored session — plus
  `generate_demo_dataset()`, which populates the dashboard with a
  synthetic patient and a realistic multi-session recovery trend so
  the platform can be demonstrated without a real patient present.
- New `main.py demo` CLI subcommand.

**Final Validation**
- `scripts/validate_project.py`: checks every package import, every
  third-party dependency, all configuration sections, the knowledge
  base, model construction, GPU status, error-handling utilities, and
  the full dashboard-inference integration path (database init →
  session write → analytics → report generation) in one pass. New
  `main.py validate` CLI subcommand.

**Documentation**
- Added `docs/user_manual.md`, `docs/deployment_guide.md`,
  `docs/api_documentation.md`, `docs/troubleshooting_guide.md`,
  `docs/faq.md`, and this `CHANGELOG.md`.

### Fixed

- **`main.py infer` was broken** — it called
  `RealtimeInferencePipeline(checkpoint_path=...)`, but that
  constructor parameter had been removed in an earlier stage's
  rewrite; every invocation would raise `TypeError`. Fixed to match
  the current `(cfg, patient_id, skip_calibration)` signature.
- **Misleading fallback explanation on clean repetition completion** —
  `inference/explainable_ai.py` would, right after a *good* repetition
  completed, compute a raw angle-deficit sentence against the
  post-rep neutral-phase frame (since the patient had returned to
  neutral), producing a confusing "remained below target by 109
  degrees" message alongside a "Recommendation: Excellent form"
  verdict. Now uses the repetition assessment to phrase a coherent,
  positive explanation when the rep was actually good.
- **Windows-unsafe report filenames** — `dashboard/report_generator.py`
  slugified patient names by replacing only spaces, leaving characters
  like `:` and `()` (e.g. from the demo dataset's timestamped patient
  names) in the output filename — invalid on Windows, this project's
  target OS. Now strips all non-alphanumeric characters.
- **Circular import** between `inference/movement_analyzer.py` and
  `inference/explainable_ai.py`, introduced while integrating the new
  clinical-assessment/explainable-AI engines into the analyzer; fixed
  with a `TYPE_CHECKING`-guarded import.

### Testing

- Added `tests/test_clinical_assessment.py`,
  `tests/test_clinical_reasoning.py`, `tests/test_explainable_ai.py`,
  `tests/test_knowledge_base.py`, `tests/test_personalization_engine.py`,
  `tests/test_recovery_tracking.py`, `tests/test_export_utils.py`,
  `tests/test_async_inference.py`, `tests/test_error_handling.py`,
  `tests/test_demo_mode.py`, and `tests/test_validate_project.py`.
- Full project test suite: 296 tests passing across every package.

---

## [Stage 4] — Clinical Dashboard

Doctor dashboard, patient management, recovery analytics, report
generation, and session history. Extended the SQLite schema to 13
normalized tables; added patient CRUD/search/plan-assignment, session
history/replay/comparison, 9 clinical 0-100 scores, patient-level
recovery analytics with trend detection, rule-based alerts and
recommendations, and PDF/Excel/CSV/JSON report generation. All 8
required dashboard pages implemented (Home, Patients, Live Session,
Recovery Analytics, Exercise History, Reports, Settings, About).

## [Stage 3] — Real-Time Inference Engine

Camera management (webcam detection, low-latency streaming, auto-
recovery, resolution negotiation), automatic front/side view
detection, automatic exercise recognition across the 10 supported
exercises, phase detection, intelligent rep counting, 14-type error
detection, sub-15-second calibration, natural-language feedback,
green/red skeleton + ghost-pose + correction-arrow visualization, and
live SQLite session logging.

## [Stage 2] — Dataset Processing & Training Pipeline

Video processing (frame extraction, normalization, integrity checks),
cached MediaPipe landmark extraction, full clinical feature suite,
dataset conversion pipeline (with a fully-tested custom-dataset path
and best-effort adapters for four public datasets), landmark/video
augmentation, the ST-GCN model (default) with an LSTM fallback, and a
complete training loop (checkpointing, resume, early stopping, mixed
precision, TensorBoard, Ranger optimizer).

## [Stage 1] — Project Scaffold

Initial architecture: package structure, configuration system, base
classes for camera/pose/model/dashboard, and documentation scaffold.
