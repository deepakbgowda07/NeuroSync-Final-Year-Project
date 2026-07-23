# Folder Structure Reference

| Path | Contents |
|---|---|
| `configs/` | YAML config files (camera, training, datasets, model, dashboard, visualization, evaluation, gpu, logging, augmentation, features, exercises, calibration) + `config_loader.py` |
| `camera/` | `camera.py` (abstract `FrameSource`), `webcam.py`, `video_loader.py`, `camera_manager.py` (webcam detection, streaming thread, auto-recovery), `camera_utils.py`, `frame_queue.py`, `fps_controller.py`, `resolution_manager.py` |
| `mediapipe_pipeline/` | `pose_estimator.py`, `landmark_extractor.py`, `pose_smoothing.py` (incl. `PoseGapHandler` for temporary landmark loss), `view_detector.py` (front/left/right auto-detection) — named to avoid shadowing the `mediapipe` PyPI package |
| `preprocessing/` | `normalizer.py`, `sequence_builder.py`, `augmentations.py`, `pipeline.py` |
| `datasets/` | `download_dataset.py`, `dataset_checker.py`, `dataset_validator.py`, `dataset_converter.py`, `sample_converter.py`, `video_processor.py`, `landmark_cache.py`, `label_processor.py`, `split_manager.py`, `preprocessing_report.py`, `dataset_statistics.py`, `rehab_dataset.py` |
| `feature_extraction/` | `angle_features.py`, `kinematic_features.py`, `clinical_features.py`, `rep_counter.py`, `feature_pipeline.py` |
| `models/` | `base_model.py`, `lstm_model.py`, `stgcn_model.py`, `graph_utils.py`, `model_factory.py` |
| `training/` | `trainer.py`, `dataset_loader.py`, `checkpoint_manager.py`, `tensorboard_logger.py`, `augmentation.py`, `scheduler.py`, `optimizer_factory.py`, `optimizers/ranger.py`, `loss_factory.py`, `metrics.py`, `gpu_monitor.py` |
| `inference/` | `realtime_pipeline.py` (orchestration), `exercise_library.py`, `exercise_recognizer.py`, `phase_detector.py`, `rep_tracker.py`, `error_detector.py`, `calibration.py`, `movement_analyzer.py`, `feedback_engine.py`, `smoothing.py`, `session_logger.py`, `predictor.py` |
| `evaluation/` | `evaluator.py`, `report_generator.py`, `rom_scoring.py` |
| `visualization/` | `skeleton_renderer.py` (green/red per-joint coloring), `hud_overlay.py`, `ghost_skeleton.py`, `ideal_pose.py`, `correction_arrows.py`, `performance_graphs.py` |
| `dashboard/` | `app.py`, `db.py`, `pages/` (Patient Management, Live Session, Exercise History, Recovery Analytics, Reports, Settings) |
| `reports/` | `session_report.py`, `pdf_report_builder.py` (stub) |
| `utils/` | `logger.py`, `gpu_utils.py`, `seed.py`, `timers.py`, `geometry.py`, `joint_angles.py`, `math_utils.py`, `file_io.py`, `video_io.py`, `csv_export.py`, `json_export.py`, `model_downloader.py`, `checkpoint_utils.py` |
| `weights/` | `checkpoints/` (training checkpoints), `exported/` (ONNX exports) — gitignored |
| `assets/` | Static assets (icons, sample frames) |
| `logs/` | Rotating runtime logs — gitignored |
| `outputs/` | `database/` (SQLite), `evaluation_reports/` — gitignored |
| `docs/` | This documentation set |
| `tests/` | pytest unit tests mirroring the package layout |
| `data/` | `raw/` (manually downloaded datasets), `processed/` (converted `.npz` samples), `unified/` (landmark cache + intermediate layout) — gitignored |
