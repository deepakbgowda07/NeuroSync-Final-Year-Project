# Dashboard Guide

## Launching

```powershell
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501` by default.

## Pages

| Page | File | Purpose |
|---|---|---|
| Home | `dashboard/app.py` | Overview + navigation |
| Patient Management | `dashboard/pages/1_Patient_Management.py` | Add/view patient records (SQLite) |
| Live Session | `dashboard/pages/2_Live_Session.py` | Launches the real-time engine, shows live-updating session stats |
| Exercise History | `dashboard/pages/3_Exercise_History.py` | Browse past sessions per patient |
| Recovery Analytics | `dashboard/pages/4_Recovery_Analytics.py` | Trend charts across sessions (scaffold) |
| Reports | `dashboard/pages/5_Reports.py` | CSV export of session history |
| Settings | `dashboard/pages/6_Settings.py` | Read-only view of active config + GPU info |

## Data storage

All dashboard data is stored in a local SQLite database at the path
configured by `dashboard.database_path` in `configs/dashboard.yaml`
(default: `outputs/database/stroke_rehab.db`). Schema is defined and
auto-created by `dashboard/db.py:SCHEMA` on first run.

## Live Session integration

Streamlit's rerun-per-interaction execution model isn't well suited to a
tight OpenCV capture loop running inline in the same process, so the
`Live Session` page launches `inference/realtime_pipeline.py` as a
subprocess (the camera window, skeleton overlay, and HUD render in a
native OpenCV window opened by that subprocess) and polls the SQLite
session log (`dashboard/db.py`) for live-updating summary stats —
total reps, mean quality, mean confidence, duration, a movement-quality
chart, and the most recent detected errors. Click "Refresh" to pull the
latest numbers while a session is running.

TODO (next development phase): embed the camera feed directly in the
browser via a streamlit-webrtc component instead of a native window,
for a fully browser-based session — see the module docstring in
`dashboard/pages/2_Live_Session.py`.

## Customizing

- Title/theme/layout: `configs/dashboard.yaml`.
- Add a new page: drop a new `N_Page_Name.py` file into `dashboard/pages/`
  (Streamlit's multipage convention — the leading number controls sidebar
  order).
