# Training Guide

## 1. Get data into the unified format

Training reads only from `data/processed/*.npz` (the unified sample
format — see `docs/dataset_guide.md`). Convert raw videos first:

```python
from pathlib import Path
from datasets.dataset_converter import DatasetConverter

converter = DatasetConverter(output_dir="data/processed")

# This project's own recordings (fully supported/tested layout):
converter.convert_custom(Path("data/raw/my_recordings"), labels_csv=Path("data/raw/my_recordings/labels.csv"))

# Public datasets (best-effort adapters — see docs/dataset_guide.md caveats):
converter.convert_ui_prmd(Path("data/raw/ui_prmd"))
converter.convert_kimore(Path("data/raw/kimore"))
converter.convert_intellirehabds(Path("data/raw/intellirehab_ds"))
converter.convert_stroke_rehab_mendeley(Path("data/raw/stroke_rehab_mendeley"))
```

Each call: extracts/normalizes video frames (`datasets/video_processor.py`,
with corrupted/duplicate-frame detection), runs cached MediaPipe pose
extraction (`datasets/landmark_cache.py`), computes the full clinical
feature suite (`feature_extraction/feature_pipeline.py`), normalizes
labels (`datasets/label_processor.py`), and writes one `.npz` per sample
plus a JSON preprocessing report to `outputs/evaluation_reports/`.

## 2. Configuration

All training hyperparameters live in `configs/training.yaml`:

```yaml
training:
  epochs: 100
  batch_size: 16          # tuned default for 6GB VRAM (RTX 3050)
  optimizer: "adamw"      # adamw | adam | sgd | ranger
  scheduler: "cosine"     # cosine | step | plateau | none
  loss: "cross_entropy"   # cross_entropy | mse | huber | weighted_cross_entropy
  mixed_precision: true
  early_stopping_patience: 15
  gradient_clip_norm: 1.0
  resume_from_checkpoint: null
```

Model architecture lives in `configs/model.yaml` — `architecture: "stgcn"`
(default) builds the Spatial-Temporal Graph Convolutional Network over
MediaPipe's 33-joint skeleton (`models/stgcn_model.py`); `architecture:
"lstm"` builds the flattened-feature-vector LSTM baseline
(`models/lstm_model.py`). Switching architectures automatically switches
the data representation used by the dataloaders — no other config
changes needed (see `models/model_factory.py:data_representation_for`).

Dataset splitting (`configs/datasets.yaml`) defaults to **subject-level**
splitting (`split_by: "subject"`) so no patient's recordings appear in
more than one of train/val/test — critical for getting an honest
validation signal on clinical data.

Augmentation (`configs/augmentation.yaml`) and feature toggles
(`configs/features.yaml`) are both fully config-driven; no augmentation
or feature-extraction parameter is hardcoded in Python.

## 3. Running training

```powershell
python -m training.trainer
python -m training.trainer --resume              # resume from last.pt
python -m training.trainer --resume path/to/checkpoint.pt
python -m training.trainer --test-only --checkpoint weights/checkpoints/best.pt
```

`training/trainer.py`'s `Trainer` class implements the complete loop:

- **Automatic CUDA detection** with graceful CPU fallback (`utils/gpu_utils.py`).
- **Mixed precision** (`torch.cuda.amp`) when `training.mixed_precision`
  is true and CUDA is active.
- **Gradient clipping** via `training.gradient_clip_norm`.
- **Early stopping** on the configured `training.early_stopping_metric`
  (`val_loss`, `val_accuracy`, or `val_f1_score`).
- **Checkpointing**: `last.pt` (always, for resume) plus the top-K
  `best_epoch*.pt` checkpoints (ranked by the early-stopping metric) and
  a `best.pt` alias — each bundles model weights, optimizer state,
  scheduler state, epoch, validation metrics, and the training config
  used (`training/checkpoint_manager.py`).
- **TensorBoard logging** of loss, learning rate, and validation
  metrics per step/epoch (`training/tensorboard_logger.py`).
- **Progress bars** via `tqdm` (disable with `training.progress_bar: false`).
- **Class imbalance**: set `training.loss_params.class_weights: "auto"`
  to compute inverse-frequency class weights from the training split
  automatically, or supply an explicit list.

## 4. Optimizers

`training/optimizer_factory.py` supports `adamw`, `adam`, `sgd`, and
`ranger`. Ranger (RAdam + Lookahead) is implemented from scratch in
`training/optimizers/ranger.py` since it isn't part of `torch.optim`;
tune it via `training.optimizer_params.ranger_k` / `ranger_alpha` /
`ranger_betas`.

## 5. Monitoring

```powershell
tensorboard --logdir logs/tensorboard
```

Per-epoch history is also saved as loss/learning-curve plots + CSV/JSON
under `outputs/evaluation_reports/` at the end of a run
(`evaluation/report_generator.py:save_training_curves`).

## 6. Evaluation

```powershell
python -m evaluation.evaluator --checkpoint weights/checkpoints/best.pt
```

Computes accuracy, precision, recall, F1, ROC AUC, and confusion matrix
(`training/metrics.py`, via scikit-learn), plus GPU utilization and an
inference-speed benchmark (mean latency + FPS,
`training/gpu_monitor.py`). Results are saved as JSON, CSV, and a
confusion-matrix image.

## 7. Pretrained weights

No pretrained ST-GCN checkpoint is auto-downloaded: public ST-GCN
releases (e.g. trained on NTU RGB+D) use a different skeleton graph
than MediaPipe's 33-point layout, so they cannot be loaded directly
(see the note in `configs/model.yaml` -> `model.stgcn.pretrained_weights`).
Train from scratch, or implement a joint-mapping adapter first.

## 8. Reproducibility

`utils/seed.set_global_seed(cfg.training.seed)` is called automatically
at `Trainer.__init__`. Dataset splitting is seeded independently via
`datasets.split_seed` (defaults to `training.seed`).
