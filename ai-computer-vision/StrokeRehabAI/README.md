# StrokeRehabAI

**AI-Assisted Stroke Rehabilitation Exercise Assessment System using Computer Vision**

> **Project status: real-time inference engine + training pipeline
> implemented; model untrained on clinical data.** Three things are
> true simultaneously: (1) the **real-time engine is fully usable
> today** — point a webcam at yourself and it detects your exercise,
> counts reps, flags form errors, and gives natural-language feedback,
> all rule-based against clinically-informed defaults, no trained model
> required; (2) the **dataset & training pipeline is fully implemented**
> (video processing, MediaPipe caching, feature extraction, an ST-GCN
> model, a complete training loop) and tested end-to-end against
> synthetic data; (3) what's still missing is **licensed clinical
> training data** to actually train that model and validate the
> rule-based thresholds against real patients — none of the four
> supported public datasets ship with this repo (see
> `docs/dataset_guide.md`). See [`docs/architecture.md`](docs/architecture.md)
> for the full module map, [`docs/inference_guide.md`](docs/inference_guide.md)
> for the real-time engine, and [`docs/training_guide.md`](docs/training_guide.md)
> for the training pipeline.

## What this is

A real-time rehabilitation exercise assessment system: point a laptop
webcam at a patient performing one of 10 supported exercises, and it
automatically detects which exercise and camera view they're in, tracks
repetitions, flags movement-quality errors (insufficient range of
motion, trunk compensation, shoulder hiking, asymmetry, and more), and
gives natural-language corrective feedback — live, on-screen, and
logged to a clinical dashboard. A short calibration step personalizes
error checks to the patient's own body rather than a population
average.

Underneath, it's a modular, config-driven pipeline: MediaPipe extracts
33 body landmarks per frame, a clinical feature-extraction suite
computes joint angles/kinematics/compensation indicators, and — for
the (separately trainable) deep-learning path — an ST-GCN
(Spatial-Temporal Graph Convolutional Network) can score exercise
correctness once trained on real session data. The real-time engine's
exercise recognition and error detection are rule-based today and work
without any trained model; the ST-GCN is the natural upgrade path once
labeled data exists.

## Target hardware

- Windows 11
- Python 3.10.11
- NVIDIA RTX 3050 Laptop GPU (6GB VRAM), CUDA-capable
- Integrated laptop webcam

The system gracefully falls back to CPU if no CUDA-capable GPU is detected
(see `utils/gpu_utils.py`).

## Quick start

```bash
# 1. Create environment (conda)
conda env create -f environment.yml
conda activate strokerehab-ai

# --- OR --- with plain pip/venv:
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 2. Editable install so packages are importable everywhere
pip install -e .

# 3. Sanity checks
python main.py check-gpu
python main.py check-data
pytest

# 4. Run a live session (webcam required)
python -m inference.realtime_pipeline

# 5. Or launch the dashboard (also lets you start a live session from the browser)
streamlit run dashboard/app.py
```

See [`docs/installation_guide.md`](docs/installation_guide.md) for the full,
Windows-specific walkthrough including CUDA setup.

## Project structure

```
StrokeRehabAI/
├── configs/              # YAML configuration + loader (incl. exercises.yaml, calibration.yaml)
├── camera/                # Webcam streaming, frame queue, FPS control, resolution, auto-recovery
├── mediapipe_pipeline/    # MediaPipe Pose wrapper, view detection, gap handling, smoothing
├── preprocessing/         # Normalization, sequence windowing, augmentation (landmark + video)
├── datasets/               # Video processing, landmark caching, conversion, splitting, labels
├── feature_extraction/    # Joint-angle, kinematic, and clinical feature extractors, rep counting
├── models/                 # ST-GCN (default), LSTM baseline, graph utils, model factory
├── training/                # Trainer, dataloaders, optimizers (incl. Ranger), scheduler/loss, checkpointing
├── inference/               # Real-time engine: exercise recognition, phase/rep tracking, error
│                            # detection, calibration, feedback, session logging, orchestration
├── evaluation/              # Offline evaluation, ROM scoring, report generation, GPU/FPS benchmarking
├── visualization/           # Skeleton (green/red), ghost pose, correction arrows, full HUD
├── dashboard/                # Streamlit clinical UI (multipage, live session launch + polling)
├── reports/                  # Session report + (planned) PDF export
├── utils/                     # Logging, GPU, geometry, IO, seeding, checkpoints
├── weights/                    # Trained model checkpoints (gitignored)
├── assets/                      # Static assets (icons, sample images)
├── logs/                         # Runtime logs (gitignored)
├── outputs/                       # Evaluation reports, exported DB, exports (gitignored)
├── docs/                           # Full documentation set (see below)
├── tests/                          # pytest unit + integration tests (148 tests)
├── requirements.txt / environment.yml
├── setup.py
└── main.py                         # CLI dispatcher: train / infer / dashboard / check-gpu / check-data
```

Full breakdown: [`docs/folder_structure.md`](docs/folder_structure.md).

## Why the model isn't trained on clinical data yet

The training pipeline itself is complete and tested (dataset conversion,
augmentation, ST-GCN, the full training loop). What's missing is data:
none of the four supported public rehab-exercise datasets (UI-PRMD,
KIMORE, IntelliRehabDS, the NIAID-hosted Stroke Rehabilitation Exercise
Dataset) can be auto-downloaded — each requires manual download and/or
registration under its own license. See
[`docs/dataset_guide.md`](docs/dataset_guide.md) for exact URLs, expected
local paths, and each dataset converter's documented assumptions.
`datasets/download_dataset.py` gives guided setup instructions. Once a
dataset is in place, `datasets/dataset_converter.py` converts it into
this project's unified `.npz` format and `training/trainer.py` trains
the default `STGCN` model end-to-end — checkpointing, resuming, and
evaluating are all implemented and tested against synthetic data.

## Documentation index

| Doc | Purpose |
|---|---|
| [architecture.md](docs/architecture.md) | System design & data flow |
| [folder_structure.md](docs/folder_structure.md) | Full directory reference |
| [installation_guide.md](docs/installation_guide.md) | Setup on Windows 11 / RTX 3050 |
| [cuda_setup.md](docs/cuda_setup.md) | CUDA/cuDNN/PyTorch GPU setup |
| [dataset_guide.md](docs/dataset_guide.md) | Supported datasets & acquisition |
| [training_guide.md](docs/training_guide.md) | How to train once data exists |
| [inference_guide.md](docs/inference_guide.md) | The real-time engine: exercise recognition, rep counting, error detection, calibration |
| [dashboard_guide.md](docs/dashboard_guide.md) | Streamlit dashboard usage |
| [developer_guide.md](docs/developer_guide.md) | Code layout, conventions, adding modules |
| [contribution_guide.md](docs/contribution_guide.md) | Contribution workflow |

## License / academic use

This project is intended as a foundation for coursework / research
(e.g. a final-year CSE project or IEEE/Springer-style publication after the
modeling phase is completed). No trained weights or patient data are
included.
