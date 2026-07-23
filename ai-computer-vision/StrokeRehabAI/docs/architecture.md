# Architecture

## Data flow (live inference)

```
Webcam (background capture thread + bounded FrameQueue)  (camera/)
        │  low-latency: consumer always gets the newest frame
        ▼
MediaPipe Pose Estimation  (mediapipe_pipeline/pose_estimator.py)
        │  raw (33, 3) landmarks + visibility
        ▼
Gap Handling  (mediapipe_pipeline/pose_smoothing.py :: PoseGapHandler)
        │  holds last-known pose through brief dropouts, else "lost"
        ▼
View Detection  (mediapipe_pipeline/view_detector.py)
        │  front / left-side / right-side, auto-detected + smoothed
        ▼
Landmark Smoothing  (mediapipe_pipeline/pose_smoothing.py :: LandmarkSmoother)
        ▼
Feature Extraction  (utils/joint_angles.py, feature_extraction/)
        │  joint angles (incl. abduction/rotation proxies), symmetry, etc.
        ▼
Movement Analysis  (inference/movement_analyzer.py)
        │  ┌─ Exercise Recognition  (inference/exercise_recognizer.py)
        │  ├─ Phase Detection        (inference/phase_detector.py)
        │  ├─ Rep Tracking            (inference/rep_tracker.py)
        │  └─ Error Detection          (inference/error_detector.py)
        │  (personalized against inference/calibration.py's baseline, if run)
        ▼
Feedback Generation  (inference/feedback_engine.py)
        │  natural-language messages, never bare "Wrong"/"Correct"
        ▼
        ├──▶ Visualization  (visualization/) — green/red skeleton, ghost
        │     pose (ideal_pose.py), correction arrows, full HUD
        └──▶ Session Logging  (inference/session_logger.py) → SQLite
              (dashboard/db.py) → Dashboard pages, in real time
```

**Relationship to the trained model (ST-GCN/LSTM):** the real-time
engine's movement-quality assessment is currently rule-based
(`inference/movement_analyzer.py`'s transparent, config-driven scoring),
not the trained classifier from `models/`. `inference/predictor.py`
still wraps a trained checkpoint for anyone who wants to run the ML
model instead of or alongside the rule-based path; the two are
intentionally decoupled so the system is fully usable before a labeled
dataset and trained checkpoint exist (see `docs/dataset_guide.md`).

## Data flow (dataset conversion)

```
Raw video (per dataset's documented layout — see docs/dataset_guide.md)
        │
        ▼
VideoProcessor  (datasets/video_processor.py)
        │  frame extraction, resolution/FPS normalization,
        │  corrupted/missing/duplicate-frame detection
        ▼
LandmarkCache  (datasets/landmark_cache.py)
        │  cached MediaPipe extraction, gap-filled for missed detections
        ▼
FeatureExtractionPipeline  (feature_extraction/feature_pipeline.py)
        │  joint angles, kinematics, trunk lean, shoulder elevation,
        │  arm extension, smoothness, rep count, compensation indicators
        ▼
LabelProcessor  (datasets/label_processor.py)
        │  unifies exercise / quality-class / regression-target /
        │  clinician-score / binary-correctness labels
        ▼
Unified .npz sample  (datasets/sample_converter.py)
        │
        └──▶ PreprocessingReportBuilder (datasets/preprocessing_report.py)
             → outputs/evaluation_reports/<dataset>_preprocessing_report.json
```

## Data flow (offline training)

```
data/processed/*.npz  (unified samples, see docs/dataset_guide.md)
        │
        ▼
File-level subject split  (datasets/split_manager.py, via training/dataset_loader.py)
        │  no patient's data crosses a train/val/test boundary
        ▼
RehabExerciseDataset  (datasets/rehab_dataset.py)
        │  windows via preprocessing/pipeline.py (shared with inference),
        │  in either "graph" (T,V,C) or "flat" (T, V*C) representation
        ▼
DataLoaders  (training/dataset_loader.py)
        │  pinned memory, persistent workers, prefetching
        ▼
Trainer  (training/trainer.py)
        │  optimizer (AdamW/Adam/SGD/Ranger) + scheduler + loss factories,
        │  mixed precision, gradient clipping, early stopping,
        │  checkpointing (last.pt + top-K best.pt), TensorBoard
        ▼
Checkpoints  (weights/checkpoints/) ──▶ Evaluation (evaluation/evaluator.py)
                                          accuracy/precision/recall/F1/ROC AUC/
                                          confusion matrix + FPS/inference-time
                                          benchmark (training/gpu_monitor.py)
```

## Design principles

1. **Config-driven, not hardcoded.** Every tunable value (batch size, model
   dims, camera resolution, thresholds) lives in `configs/*.yaml`, loaded
   once via `configs/config_loader.py`.
2. **Pipeline symmetry.** `preprocessing/pipeline.py` is used identically by
   both the training data pipeline and the live inference pipeline, so
   train/serve skew is structurally prevented.
3. **Pose-library agnostic core.** `feature_extraction/` and
   `preprocessing/` operate on plain numpy landmark arrays, not MediaPipe
   objects directly — `mediapipe_pipeline/landmark_extractor.py` is the only
   place that translates between the two.
4. **Fail loud, fall back gracefully.** GPU absence, missing checkpoints,
   and missing datasets are all handled with clear log warnings and safe
   fallbacks (CPU inference, scaffold-mode predictions) rather than crashes.
5. **Architecture-agnostic data pipeline.** `models/model_factory.py:data_representation_for`
   is the single source of truth for which tensor layout ("graph" for
   ST-GCN, "flat" for the LSTM baseline) a given `model.architecture`
   needs; `training/dataset_loader.py` and `inference/predictor.py` both
   read from it rather than hardcoding a layout, so switching
   architectures in `configs/model.yaml` requires no other code changes.
6. **No patient data crosses a split boundary.** Train/val/test splitting
   happens by subject_id (`datasets/split_manager.py`), and — because
   augmentation can change a sample's window count — splitting happens at
   the *file* level before any windowed dataset object is built
   (`training/dataset_loader.py`), avoiding index misalignment between
   differently-augmented dataset instances.
7. **Transparent before learned.** The real-time engine's exercise
   recognition, phase detection, and error detection
   (`inference/exercise_recognizer.py`, `phase_detector.py`,
   `error_detector.py`) are rule-based against config-driven thresholds
   — inspectable and adjustable without retraining, and fully usable
   before any labeled dataset exists. The trained ST-GCN/LSTM classifier
   is an independent, swappable signal (`inference/predictor.py`), not a
   hard dependency.
8. **Personalize, don't assume.** Where a patient-specific baseline is
   available (`inference/calibration.py`), error detection compares
   against it rather than a fixed population-average threshold — see
   `error_detector.py`'s calibration-aware angle check.
