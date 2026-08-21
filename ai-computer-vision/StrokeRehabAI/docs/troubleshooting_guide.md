# Troubleshooting Guide

Run `python main.py validate` first — it checks imports, dependencies,
configuration, model building, and the dashboard-inference integration
path in one pass and will often point directly at the problem.

## Installation & environment

**"No module named 'mediapipe.solutions'" or pose estimation silently fails**
`mediapipe` versions from roughly 0.10.20 onward removed the legacy
`mp.solutions.pose` API this project uses. Reinstall the exact pinned
version: `pip install mediapipe==0.10.14`. See
`docs/installation_guide.md`'s MediaPipe version note.

**`pip install -r requirements.txt` fails on `torch`**
Install the CUDA-matched PyTorch build separately per
`docs/cuda_setup.md`, then install the rest of `requirements.txt`.

**Import errors for project packages (`camera`, `inference`, etc.)**
Run `pip install -e .` from the project root so packages are
importable without manual `PYTHONPATH` changes.

## Camera

**"Could not connect to the camera" (`CameraError`)**
- Check the camera isn't held by another app (close other video apps).
- Try `python -c "from camera.camera_manager import CameraManager; print(CameraManager.detect_webcams())"`
  to see what indices/resolutions are actually available.
- Try `backend: "ANY"` in `configs/camera.yaml` if `DSHOW`/`MSMF` fail.

**Live session freezes or the window stops updating**
The background capture thread has an automatic-recovery mechanism
(`configs/camera.yaml -> auto_recovery`); if it still freezes, the
camera driver itself may have crashed — unplug/replug the camera and
restart the session.

**Requested resolution doesn't seem to apply**
Not every webcam honors every resolution request; check the log for a
"Camera did not honor requested resolution" warning from
`camera/resolution_manager.py` — it will show the actual resolution
the camera fell back to.

## GPU / CUDA

**`python main.py check-gpu` reports CPU fallback unexpectedly**
- Confirm `nvidia-smi` works from a terminal at all.
- Confirm the installed `torch` build includes CUDA support:
  `python -c "import torch; print(torch.version.cuda)"` — `None` means
  a CPU-only wheel was installed; see `docs/cuda_setup.md`.
- The system still runs on CPU (slower) — this is a graceful
  degradation, not a crash (`utils/error_handling.py:CUDAUnavailableError`
  is only raised where CUDA is strictly required, e.g. certain training
  configurations).

**"CUDA out of memory" during training**
Lower `training.batch_size` in `configs/training.yaml` before reducing
model size; `configs/gpu.yaml`'s `max_batch_size_6gb_vram` is a
starting point for an RTX 3050.

## Dataset & training

**`python main.py check-data` reports all datasets "NOT READY"**
Expected until you manually download a dataset — see
`docs/dataset_guide.md`. None are auto-downloaded (licensing).

**Training crashes with an empty train/val/test split**
Very small datasets combined with the default split ratios can produce
an empty split; `training/dataset_loader.py` logs a warning naming
which split is empty. Add more data, or adjust
`datasets.train_split`/`val_split`/`test_split` in `configs/datasets.yaml`.

**Checkpoint won't load ("weights_only" error)**
This project's checkpoints intentionally bundle config/metrics
alongside weights; `utils/checkpoint_utils.py` already loads with
`weights_only=False`. If you see this error, you're likely loading a
checkpoint directly with a raw `torch.load(..., weights_only=True)`
call instead of `utils.checkpoint_utils.load_checkpoint()`.

## Dashboard

**Dashboard shows "No patients registered yet" after upgrading**
The database schema migrates additively (`dashboard/db.py`) — your
existing patients should still be there. Check
`configs/dashboard.yaml -> database_path` points at the same file it
did before (a common cause is accidentally pointing at a fresh/empty
database path).

**Report generation fails**
- PDF/Excel require `reportlab`/`openpyxl` — confirm both are
  installed (`pip show reportlab openpyxl`).
- If a patient name contains unusual characters, filenames are
  slugified to be filesystem-safe automatically
  (`dashboard/report_generator.py:_slugify`) — if you still see a
  filesystem error, check the `outputs/evaluation_reports/clinical_reports/`
  directory exists and is writable.

**Live Session page shows no live stats**
The page polls the database; if the real-time engine subprocess hasn't
written a frame yet (e.g. still in calibration), the page will show
placeholders until the first frame is logged. Click **Refresh**.

## General

**"CorruptedFileError" / "InvalidConfigurationError"**
These come from `utils/error_handling.py` and always include a
specific, actionable message — check the log file
(`logs/strokerehab_*.log`) for the full traceback and context.

**Still stuck?**
Check `logs/errors.log` for the full detail behind any user-facing
error message (`utils/error_handling.py:log_full_detail` always logs
the complete exception even when the on-screen message is
simplified).
