"""
Exercise History page.

Full session history per patient — date, duration, exercises
performed, successful/incorrect repetitions, average score, and
recovery metrics — plus session replay (a frame-by-frame joint-angle
timeline) and comparison against a previous session/week/month.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from configs.config_loader import load_config
from dashboard.clinical_metrics import ClinicalMetricsCalculator
from dashboard.patient_manager import PatientManager
from dashboard.session_manager import SessionManager

st.title("Exercise History")

cfg = load_config()
db_path = cfg.dashboard.database_path

patient_manager = PatientManager(db_path)
session_manager = SessionManager(db_path)
metrics_calculator = ClinicalMetricsCalculator()

patients = patient_manager.list_all(include_inactive=True)
patient_options = {"All patients": None}
patient_options.update({p.full_name: p.patient_id for p in patients})

selected = st.selectbox("Filter by patient", list(patient_options.keys()))
selected_patient_id = patient_options[selected]

sessions = session_manager.list_sessions(patient_id=selected_patient_id, limit=200)

if not sessions:
    st.info("No sessions recorded yet.")
    st.stop()

st.subheader("Session History")
patient_name_by_id = {p.patient_id: p.full_name for p in patients}
history_rows = []
for s in sessions:
    detail = session_manager.get_session_detail(s.session_id)
    scores = metrics_calculator.compute_for_session(detail["frames"], detail["events"], detail["reps"])
    history_rows.append({
        "Session ID": s.session_id,
        "Patient": patient_name_by_id.get(s.patient_id, "Unknown") if selected_patient_id is None else selected,
        "Date": s.started_at[:19].replace("T", " "),
        "Duration (s)": f"{s.duration_seconds:.0f}" if s.duration_seconds else "-",
        "Exercise(s)": s.exercise_name,
        "Successful Reps": detail["successful_repetitions"],
        "Incorrect Reps": detail["incorrect_repetitions"],
        "Average Score": f"{scores.exercise_quality_score:.0f}",
        "Recovery Index": f"{scores.recovery_index:.0f}",
    })

st.dataframe(history_rows, use_container_width=True)

st.divider()
st.subheader("Session Replay")

session_labels = {f"#{s.session_id} — {s.started_at[:19].replace('T', ' ')} ({s.exercise_name})": s.session_id for s in sessions}
selected_label = st.selectbox("Select a session to replay", list(session_labels.keys()))
replay_session_id = session_labels[selected_label]

detail = session_manager.get_session_detail(replay_session_id)

replay_cols = st.columns(4)
replay_cols[0].metric("Successful Reps", detail["successful_repetitions"])
replay_cols[1].metric("Incorrect Reps", detail["incorrect_repetitions"])
replay_cols[2].metric("Total Frames", len(detail["frames"]))
replay_cols[3].metric("Events Logged", len(detail["events"]))

if detail["frames"]:
    frame_index = st.slider("Replay frame", 0, len(detail["frames"]) - 1, 0)
    frame = detail["frames"][frame_index]

    replay_detail_cols = st.columns(3)
    replay_detail_cols[0].metric("Phase", frame.get("phase") or "-")
    replay_detail_cols[1].metric("Movement Quality", f"{(frame.get('movement_quality') or 0) * 100:.0f}%")
    replay_detail_cols[2].metric("ROM (deg)", f"{frame.get('rom_deg') or 0:.0f}\u00b0")

    st.json(frame.get("joint_angles", {}), expanded=False)

    # Full-session joint angle timeline with a marker for the current replay frame
    angle_series: dict = {}
    for f in detail["frames"]:
        for name, value in (f.get("joint_angles") or {}).items():
            if value is not None:
                angle_series.setdefault(name, []).append(value)

    trackable = [k for k in angle_series if "elbow" in k or "shoulder_angle" in k][:3]
    if trackable:
        fig = go.Figure()
        for name in trackable:
            fig.add_trace(go.Scatter(y=angle_series[name], mode="lines", name=name.replace("_", " ").title()))
        fig.add_vline(x=frame_index, line_dash="dash", line_color="red")
        fig.update_layout(title="Joint Angle Timeline (red line = current replay position)", height=350)
        st.plotly_chart(fig, use_container_width=True)
else:
    st.caption("No frame data recorded for this session.")

if detail["events"]:
    st.subheader("Errors / Compensation Events")
    st.dataframe(detail["events"], use_container_width=True)

st.divider()
st.subheader("Compare This Session")
comparison_type = st.selectbox(
    "Compare against", ["previous_session", "previous_week", "previous_month"], key="history_comparison"
)
comparison = session_manager.compare_sessions(replay_session_id, comparison=comparison_type)
if comparison["baseline"] is None:
    st.info(comparison.get("note", "No baseline session available."))
else:
    verdict_icons = {"improvement": "🟢", "regression": "🔴", "stable": "🟡"}
    for metric, verdict in comparison["verdicts"].items():
        st.write(f"{verdict_icons.get(verdict, '⚪')} **{metric.title()}**: {verdict.title()}")
