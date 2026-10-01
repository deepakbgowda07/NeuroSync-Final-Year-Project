# StrokeRehabAI

StrokeRehabAI is a computer-vision system for stroke rehabilitation. It
uses a webcam to observe exercises, count repetitions, detect movement
problems, show feedback, and save session data in a local dashboard.

The live pipeline currently uses transparent, configuration-based rules.
The ST-GCN and LSTM models are available for training and evaluation, but
this project does not include a clinically trained checkpoint or patient
dataset.

## Workflow

### Live webcam workflow

1. `camera/` captures the newest webcam frame.
2. `mediapipe_pipeline/` detects 33 body landmarks.
3. The landmarks are smoothed and the camera view is detected.
4. `utils/` and `feature_extraction/` calculate joint angles, motion,
   symmetry, and other movement features.
5. `inference/` recognizes the exercise, tracks its phase and repetitions,
   checks movement errors, and calculates a quality score.
6. `visualization/` draws the skeleton, warnings, ideal pose, arrows, and HUD.
7. `inference/session_logger.py` saves session data to SQLite.
8. `dashboard/` displays patients, sessions, history, analytics, and reports.

### Dataset and training workflow

Raw videos are read from `data/raw/`. `datasets/` converts them into
processed `.npz` files in `data/processed/`. The files contain landmarks,
labels, metadata, and features. `preprocessing/` normalizes and windows
the sequences. `training/` loads the windows, trains a model, saves
checkpoints in `weights/checkpoints/`, and writes TensorBoard logs and
evaluation reports.

## Important folders and files

| Path | Responsibility |
|---|---|
| `main.py` | CLI for training, inference, checks, and dashboard help |
| `configs/` | YAML settings for camera, exercises, models, training, and dashboard |
| `camera/` | Webcam/video input, frame queue, FPS control, and recovery |
| `mediapipe_pipeline/` | Pose estimation, landmark handling, smoothing, and view detection |
| `feature_extraction/` | Joint-angle, kinematic, clinical, and repetition features |
| `inference/` | Live recognition, phase detection, rep tracking, errors, feedback, and logging |
| `visualization/` | Skeleton, ideal pose, correction arrows, and HUD rendering |
| `datasets/` | Raw dataset discovery, conversion, labels, caching, and validation |
| `preprocessing/` | Landmark normalization, augmentation, and sequence windows |
| `models/` | ST-GCN model, LSTM model, graph utilities, and model factory |
| `training/` | Data loaders, optimizer, loss, scheduler, checkpoints, and training loop |
| `evaluation/` | Model metrics, reports, ROM scores, and benchmarks |
| `dashboard/` | Streamlit app, SQLite database, and dashboard pages |
| `data/raw/` | Manually supplied videos and labels |
| `data/processed/` | Converted `.npz` training samples |
| `weights/` | Trained model checkpoints and exports |
| `outputs/` | SQLite database and generated reports |
| `tests/` | Automated unit and integration tests |

## Technologies and models

- Python 3.10
- OpenCV for camera and image processing
- MediaPipe Pose for 33 body landmarks
- NumPy and SciPy-style numerical processing
- PyTorch and TorchVision
- ST-GCN for graph-based skeleton sequence classification
- LSTM as a baseline sequence model
- scikit-learn for metrics
- Streamlit for the dashboard
- SQLite for local session storage
- YAML configuration files
- ONNX and ONNX Runtime for model export/inference support
- TensorBoard for training logs

## Local setup

Run these commands from this folder on Windows:

```powershell
py -3.10 -m venv .venv310
.venv310\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install --no-deps -e .
```

No environment variables are required. CUDA is used automatically when a
compatible PyTorch build and NVIDIA driver are available; otherwise the
project uses the CPU.

## Run locally

Check the setup:

```powershell
python main.py check-gpu
python main.py check-data
```

Start the dashboard:

```powershell
streamlit run dashboard/app.py
```

Open `http://localhost:8501` in a browser. The dashboard can start a live
session from its Live Session page.

Run the live camera pipeline directly:

<<<<<<< Updated upstream
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
├── dashboard/                # Clinical dashboard: patients, live session, recovery analytics, reports, alerts
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
=======
```powershell
python -m inference.realtime_pipeline
python -m inference.realtime_pipeline --skip-calibration
>>>>>>> Stashed changes
```

Press `q` in the camera window to stop. A trained checkpoint is optional for
testing the rule-based live pipeline, but real model training requires
processed data in `data/processed/`.

Run tests:

```powershell
pip install pytest pytest-cov
pytest
```

<<<<<<< Updated upstream
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
| [dashboard_guide.md](docs/dashboard_guide.md) | Clinical dashboard: patient management, recovery analytics, alerts, reporting |
| [developer_guide.md](docs/developer_guide.md) | Code layout, conventions, adding modules |
| [contribution_guide.md](docs/contribution_guide.md) | Contribution workflow |
| [user_manual.md](docs/user_manual.md) | Day-to-day clinical usage walkthrough |
| [deployment_guide.md](docs/deployment_guide.md) | Deployment scenarios, model export, performance tuning |
| [api_documentation.md](docs/api_documentation.md) | Programmatic interface reference for every package |
| [troubleshooting_guide.md](docs/troubleshooting_guide.md) | Common issues and fixes |
| [faq.md](docs/faq.md) | Frequently asked questions |
| [CHANGELOG.md](CHANGELOG.md) | Version history and what changed in each development stage |

## License / academic use

This project is intended as a foundation for coursework / research
(e.g. a final-year CSE project or IEEE/Springer-style publication after the
modeling phase is completed). No trained weights or patient data are
included.
=======
For more detail, see the files in `docs/`, especially `architecture.md`,
`inference_guide.md`, `training_guide.md`, and `dataset_guide.md`.
>>>>>>> Stashed changes
