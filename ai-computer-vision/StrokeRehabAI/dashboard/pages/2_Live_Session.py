"""
Live Session page.

Launches the real-time inference engine (inference/realtime_pipeline.py)
as a subprocess and polls the SQLite session log for live results.
Streamlit's rerun-per-interaction execution model isn't well suited to
running the tight OpenCV capture loop inline, so the actual camera
window and skeleton/HUD overlay render in a separate native window
(opened by the subprocess) while this page shows live-updating summary
stats pulled from the database.

TODO (next development phase): embed the camera feed directly in the
browser via a streamlit-webrtc component instead of a native OpenCV
window, for a fully browser-based session.
"""

import subprocess
import sys

import streamlit as st

from configs.config_loader import load_config
from dashboard.db import get_connection
from inference.exercise_library import SUPPORTED_EXERCISES

st.title("Live Session")

cfg = load_config()
db_path = cfg.dashboard.database_path

with get_connection(db_path) as conn:
    patients = conn.execute("SELECT patient_id, full_name FROM patients").fetchall()

patient_options = {"(none / unassigned)": None}
patient_options.update({row["full_name"]: row["patient_id"] for row in patients})

selected_patient = st.selectbox("Patient", list(patient_options.keys()))
patient_id = patient_options[selected_patient]

st.selectbox(
    "Exercise (auto-detected during the session — for reference only)",
    [name.replace("_", " ").title() for name in SUPPORTED_EXERCISES],
)
skip_calibration = st.checkbox("Skip calibration (not recommended)", value=False)

st.markdown(
    """
    The system automatically detects which exercise you're performing
    and which camera view (front / left-side / right-side) you're in —
    there's nothing to configure by hand beyond standing where the
    camera can see your upper body.
    """
)

if "live_session_process" not in st.session_state:
    st.session_state.live_session_process = None

col1, col2 = st.columns(2)

with col1:
    if st.button("Start Session", type="primary", disabled=st.session_state.live_session_process is not None):
        command = [sys.executable, "-m", "inference.realtime_pipeline"]
        if patient_id is not None:
            command += ["--patient-id", str(patient_id)]
        if skip_calibration:
            command += ["--skip-calibration"]

        st.session_state.live_session_process = subprocess.Popen(command)
        st.success("Session launched — a camera window will open. Return here for live stats, or press 'q' in the camera window to stop.")

with col2:
    if st.button("Stop Session", disabled=st.session_state.live_session_process is None):
        if st.session_state.live_session_process is not None:
            st.session_state.live_session_process.terminate()
            st.session_state.live_session_process = None
        st.info("Session stop requested.")

st.divider()
st.subheader("Most recent session")

with get_connection(db_path) as conn:
    latest_session = conn.execute("SELECT * FROM sessions ORDER BY started_at DESC LIMIT 1").fetchone()

if latest_session is None:
    st.info("No sessions recorded yet.")
else:
    session_id = latest_session["session_id"]
    metric_cols = st.columns(4)
    metric_cols[0].metric("Total Reps", latest_session["total_reps"] or 0)
    metric_cols[1].metric("Mean Quality", f"{(latest_session['mean_quality'] or 0) * 100:.0f}%")
    metric_cols[2].metric("Mean Confidence", f"{(latest_session['mean_confidence'] or 0) * 100:.0f}%")
    metric_cols[3].metric("Duration (s)", f"{latest_session['duration_seconds'] or 0:.0f}")

    with get_connection(db_path) as conn:
        recent_frames = conn.execute(
            "SELECT * FROM session_frames WHERE session_id = ? ORDER BY frame_id DESC LIMIT 50", (session_id,)
        ).fetchall()
        recent_events = conn.execute(
            "SELECT * FROM session_events WHERE session_id = ? AND event_type = 'error' ORDER BY event_id DESC LIMIT 20",
            (session_id,),
        ).fetchall()

    if recent_frames:
        st.line_chart(
            {"movement_quality": [row["movement_quality"] for row in reversed(recent_frames) if row["movement_quality"] is not None]}
        )

    if recent_events:
        st.subheader("Recent errors / compensations")
        st.dataframe([dict(row) for row in recent_events], use_container_width=True)
    else:
        st.caption("No errors recorded in the most recent frames.")

    st.button("Refresh", on_click=lambda: None)
