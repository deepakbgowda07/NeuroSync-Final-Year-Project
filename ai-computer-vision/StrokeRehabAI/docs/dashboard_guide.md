# Dashboard Guide — Clinical Rehabilitation Platform

This is a **clinical rehabilitation monitoring platform** for
physiotherapists and doctors — not a consumer fitness dashboard. It
consumes live data from the real-time inference engine
(`inference/realtime_pipeline.py`) and stores every session in a local
SQLite database for longitudinal recovery tracking.

## Launching

```powershell
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501` by default.

## Pages

| Page | File | Purpose |
|---|---|---|
| Home | `dashboard/app.py` | Clinic-wide overview: active patient count, sessions logged, cross-patient alerts |
| Patients | `dashboard/pages/1_Patient_Management.py` | Register / edit / deactivate / search patients; assign rehabilitation plans |
| Live Session | `dashboard/pages/2_Live_Session.py` | Launches the real-time engine; live rep/phase/score/ROM/compensation/FPS/CUDA status |
| Recovery Analytics | `dashboard/pages/4_Recovery_Analytics.py` | Clinical 0-100 scores, alerts, recommendations, and all progress graphs |
| Exercise History | `dashboard/pages/3_Exercise_History.py` | Full session history, frame-by-frame replay, session comparison |
| Reports | `dashboard/pages/5_Reports.py` | Generate/download PDF, Excel, CSV, JSON clinical reports |
| Settings | `dashboard/pages/6_Settings.py` | Camera/model/exercise-threshold config (read-only), theme, export defaults, DB backup, logging |
| About | `dashboard/pages/7_About.py` | Platform overview and system status |

## Architecture

```
inference/realtime_pipeline.py
        │  (inference/session_logger.py streams live results)
        ▼
SQLite (dashboard/db.py) — 13 normalized tables:
  patients · rehabilitation_plans · exercises · sessions · session_frames ·
  session_reps · session_events · exercise_results · joint_metrics ·
  recovery_scores · compensation_events · generated_reports · settings
        │
        ├──▶ dashboard/patient_manager.py     (CRUD, search, plan assignment)
        ├──▶ dashboard/session_manager.py     (history, replay, comparison)
        ├──▶ dashboard/clinical_metrics.py     (9 normalized 0-100 scores)
        ├──▶ dashboard/analytics.py             (patient-level trend/recovery analytics)
        ├──▶ dashboard/recommendation_engine.py  (rule-based clinical suggestions)
        ├──▶ dashboard/alert_system.py            (recovery/compensation/adherence alerts)
        └──▶ dashboard/report_generator.py         (PDF/Excel/CSV/JSON export)
```

The schema evolves additively (`dashboard/db.py:_apply_migrations`) —
upgrading this project's dashboard code against an existing database
from an earlier version adds missing tables/columns automatically; it
never requires deleting your data.

## Clinical scores (0-100)

Computed per session by `dashboard/clinical_metrics.py` from the
frames/events/reps the real-time engine already logged — no extra data
collection required:

Shoulder Mobility · Elbow Mobility · Movement Stability · Movement
Smoothness · Symmetry · Compensation · Exercise Quality · Recovery
Index · Overall Rehabilitation Score.

> **TODO (next development phase):** scoring weights and blend
> coefficients are principled starting defaults, not clinician-
> validated — see the TODO in `dashboard/clinical_metrics.py`.

## Recovery analytics

`dashboard/analytics.py:RecoveryAnalyticsEngine` aggregates clinical
scores across a patient's session history into: average exercise
accuracy, average/max/min ROM, average joint-angle error, exercise
completion rate, movement smoothness, compensation frequency, exercise
consistency, average session duration, a fatigue indicator (within-
session late-session quality decline), overall recovery score,
improvement percentage, and a trend classification
(improving/declining/stable).

## Alerts

`dashboard/alert_system.py` checks, per patient: recovery score
decrease, compensation increase, exercise accuracy decrease, ROM
decrease (all session-over-session), skipped sessions (no session
within roughly 2x the expected interval), and unusually short
sessions. Alerts are computed on demand from current data — nothing is
pre-stored, so they always reflect the latest session.

## Recommendations

`dashboard/recommendation_engine.py` generates specific, actionable
suggestions (e.g. "Increase Shoulder Flexion repetitions", "Reduce
trunk compensation") derived from the same analytics — never generic
filler text. See the module's TODO regarding threshold validation.

## Reports

`dashboard/report_generator.py` exports, in any of four formats:
patient information, exercise summary, recovery metrics, joint
analysis, ROM analysis, compensation analysis, recommendations, and a
full session summary. Every generated file is logged to the
`generated_reports` table.

| Format | Library | Notes |
|---|---|---|
| PDF | ReportLab | Formatted tables + recommendations, ready to print/share |
| Excel | OpenPyXL | Multi-sheet workbook (summary, sessions, joints, recommendations) |
| CSV | stdlib `csv` | Flat sectioned export |
| JSON | stdlib `json` | Full structured payload, machine-readable |

## Live Session integration

Streamlit's rerun-per-interaction execution model isn't well suited to
running the tight OpenCV capture loop inline, so the `Live Session`
page launches `inference/realtime_pipeline.py` as a subprocess (camera
window + skeleton/HUD overlay render natively) and polls the SQLite
session log for live-updating summary stats: current exercise, phase,
rep, score, joint angles, ROM, compensation detection, session timer,
FPS, and CUDA status.

## Customizing

- Dashboard title/layout: `configs/dashboard.yaml`.
- Theme / default export format: set via the Settings page (persisted
  in the `settings` table, not the YAML config).
- Camera / model / exercise-threshold values: edit the relevant
  `configs/*.yaml` file directly (shown read-only in Settings for
  transparency).
- Add a new page: drop a new `N_Page_Name.py` file into
  `dashboard/pages/` (Streamlit's multipage convention).
