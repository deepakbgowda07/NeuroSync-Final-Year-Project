# Deployment Guide

Guidance for deploying StrokeRehabAI beyond a single development
machine — a research lab, clinic workstation, or demo environment.
For local development setup, see `installation_guide.md`.

## 1. Deployment scenarios

### A. Single clinic workstation (recommended default)
The intended deployment target: one Windows 11 laptop/desktop with an
RTX 3050-class GPU, running both the real-time inference engine and
the Streamlit dashboard locally, with SQLite as local storage. This is
what `installation_guide.md` sets up. No server, no network exposure
required — appropriate for a single-clinician or research-lab use case
with local patient data.

### B. Multi-user / shared-server dashboard
If multiple clinicians need to review data from a shared machine (but
each still runs live sessions on their own local webcam-equipped
device):
1. Run the dashboard on a shared machine: `streamlit run dashboard/app.py --server.address 0.0.0.0`
2. Point `configs/dashboard.yaml -> database_path` at a shared network
   location, or migrate `dashboard/db.py`'s schema to a networked
   database (see the TODO below).
3. **Do not** expose the dashboard directly to the public internet
   without adding authentication — Streamlit has no built-in auth;
   put it behind a reverse proxy (e.g. nginx with basic auth, or a
   VPN) if remote access is required.

> **TODO (next development phase):** `dashboard/db.py` uses SQLite via
> stdlib `sqlite3`, which supports concurrent readers but serializes
> writers — fine for single-clinic use, but a genuine multi-clinician
> concurrent-write deployment should migrate to PostgreSQL (the schema
> in `dashboard/db.py:SCHEMA` is already normalized and would port with
> minor SQL dialect changes).

## 2. Model deployment artifacts

`models/export_utils.py` produces three deployment-ready model formats:

```powershell
python -c "
from configs.config_loader import load_config
from models.model_factory import build_model
from models.export_utils import export_torchscript, export_onnx, export_quantized
from utils.checkpoint_utils import load_model_weights

cfg = load_config()
model = build_model(cfg.model)
load_model_weights(model, 'weights/checkpoints/best.pt')
model.eval()

export_torchscript(model, cfg.model, 'weights/exported/model.ts.pt')
export_onnx(model, cfg.model, 'weights/exported/model.onnx')
export_quantized(model, cfg.model, 'weights/exported/model_quantized.pt')
"
```

- **TorchScript** (`.ts.pt`): fastest path for a pure-PyTorch deployment
  without a Python training environment.
- **ONNX**: for deployment via ONNX Runtime (cross-platform, can run
  without PyTorch installed at all) — verify with
  `ModelExporter.verify_onnx` / the `onnx`/`onnxruntime` packages.
- **Quantized** (dynamic, int8): smaller artifact and faster CPU
  inference; the RTX 3050 GPU path should prefer mixed precision
  (already used in training, see `training/trainer.py`) over
  quantization for GPU inference.

## 3. Performance targets and tuning

Target: 25-35 FPS at 720p, <50ms inference latency (see
`docs/inference_guide.md -> Performance`). If a deployment machine
falls short:

1. Confirm CUDA is actually active: `python main.py check-gpu`.
2. Lower `camera.width`/`camera.height` in `configs/camera.yaml`
   (e.g. drop to 720p if defaulting to 1080p).
3. Reduce `model.stgcn.channels` in `configs/model.yaml` (fewer/narrower
   layers) if the deployment GPU has less than 6GB VRAM.
4. Confirm `camera.frame_queue.enabled: true` and
   `camera.auto_recovery.enabled: true` (both default-on) — these keep
   the pipeline responsive under load.
5. Use the exported quantized model for CPU-only deployment machines.

## 4. Environment reproducibility

Pin exact versions when deploying — `requirements.txt` and
`environment.yml` are both version-pinned already; do not use `pip
install -U` on a deployment machine without re-testing. In particular,
`mediapipe` **must** stay at exactly `0.10.14` (see
`docs/installation_guide.md`'s MediaPipe version note) — newer
releases silently break pose estimation.

## 5. Backups

Use the dashboard's **Settings → Backup** tab to create a timestamped
copy of the SQLite database (`outputs/database/backups/`). For
automated backups on a deployment machine, schedule the same copy
operation (Windows Task Scheduler / cron) rather than relying on
manual backups.

## 6. Pre-deployment checklist

Run the final validation script before demonstrating or handing off a
deployment:

```powershell
python main.py validate
```

This checks: all packages import, all required third-party
dependencies are installed, all config sections/exercise definitions
load, the knowledge base loads, the model builds, GPU status reports
correctly, error-handling utilities function, and the full
dashboard-inference integration path (database init → session write →
analytics → report generation) works end to end.
