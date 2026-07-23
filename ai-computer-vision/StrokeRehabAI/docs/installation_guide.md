# Installation Guide (Windows 11 / RTX 3050 Laptop GPU)

## 1. Prerequisites

- Windows 11
- Python 3.10.11 (exact minor version matters for prebuilt wheel
  compatibility with `mediapipe` and `onnxruntime-gpu`)
- NVIDIA driver supporting your target CUDA version (see `docs/cuda_setup.md`)
- Git (optional, for cloning/version control)

## 2. Get the code

Unzip the project archive to a working folder, e.g. `C:\dev\StrokeRehabAI`.

## 3. Create an isolated environment

> **MediaPipe version note:** this project uses `mediapipe`'s legacy
> `mp.solutions.pose` API (`mediapipe_pipeline/pose_estimator.py`).
> Releases from roughly 0.10.20 onward dropped that API in favor of the
> newer Tasks API, which requires downloading a separate `.task` model
> file. Install **exactly** `mediapipe==0.10.14` (as pinned in
> `requirements.txt` / `setup.py`) — `pip install mediapipe` with no
> version pin, or any `>=` constraint, can silently pull an incompatible
> newer release.

### Option A — Conda (recommended, simplest CUDA handling)

```powershell
conda env create -f environment.yml
conda activate strokerehab-ai
```

### Option B — venv + pip

```powershell
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you use Option B, install the CUDA-enabled PyTorch build explicitly
(see `docs/cuda_setup.md`) — the default PyPI `torch` wheel may be CPU-only.

## 4. Editable install of the project itself

```powershell
pip install -e .
```

This registers `camera`, `models`, `training`, etc. as importable packages
system-wide within the environment, and installs the `strokerehab-train` /
`strokerehab-infer` console scripts.

## 5. Verify the installation

```powershell
python main.py check-gpu
python main.py check-data
pytest
```

Expected: `check-gpu` reports either your RTX 3050 or a CPU fallback
message; `check-data` reports all configured datasets as `NOT READY` (until
you follow `docs/dataset_guide.md`); `pytest` passes (model-dependent tests
auto-skip if torch isn't installed).

## 6. Launch the dashboard

```powershell
streamlit run dashboard/app.py
```

## 7. Run a live camera session (once a checkpoint exists)

```powershell
python main.py infer --checkpoint weights/checkpoints/<your_checkpoint>.pt
```

Without `--checkpoint`, the pipeline still runs end-to-end using randomly
initialized weights, clearly flagged as scaffold-mode in both logs and the
feedback overlay — useful for verifying the camera/pose/visualization
pipeline before a trained model exists.
