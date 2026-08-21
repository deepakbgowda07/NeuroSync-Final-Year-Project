"""
Recovery Analytics page.

For the selected patient: the full recovery-metrics suite, clinical
0-100 scores, active alerts, auto-generated recommendations, and the
required set of interactive Plotly progress graphs (recovery score
over time, ROM improvement, exercise accuracy, joint angle trends,
compensation frequency, session duration, exercise compliance, weekly/
monthly progress, and session-to-session comparison).
"""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from configs.config_loader import load_config
from dashboard.alert_system import AlertSystem
from dashboard.analytics import RecoveryAnalyticsEngine
from dashboard.clinical_metrics import ClinicalMetricsCalculator
from dashboard.patient_manager import PatientManager
from dashboard.recommendation_engine import RecommendationEngine
from dashboard.session_manager import SessionManager

st.title("Recovery Analytics")

cfg = load_config()
db_path = cfg.dashboard.database_path

patient_manager = PatientManager(db_path)
session_manager = SessionManager(db_path)
analytics_engine = RecoveryAnalyticsEngine(db_path)
metrics_calculator = ClinicalMetricsCalculator()
recommendation_engine = RecommendationEngine()
alert_system = AlertSystem(db_path)

patients = patient_manager.list_all()
if not patients:
    st.info("No patients registered yet — add one on the Patients page.")
    st.stop()

selected_name = st.selectbox("Patient", [p.full_name for p in patients])
patient = next(p for p in patients if p.full_name == selected_name)

sessions = session_manager.list_sessions(patient_id=patient.patient_id, limit=100)
if not sessions:
    st.info(f"No sessions recorded yet for {patient.full_name}.")
    st.stop()

analytics = analytics_engine.compute_patient_analytics(patient.patient_id)

# ------------------------------------------------------------------
# Summary metrics
# ------------------------------------------------------------------
st.subheader("Recovery Summary")
row1 = st.columns(4)
row1[0].metric("Recovery Score", f"{analytics.recovery_score:.0f}/100")
row1[1].metric("Trend", analytics.trend.replace("_", " ").title())
row1[2].metric("Improvement", f"{analytics.improvement_percentage:+.1f}%")
row1[3].metric("Sessions Logged", analytics.num_sessions)

row2 = st.columns(4)
row2[0].metric("Avg Exercise Accuracy", f"{analytics.average_exercise_accuracy:.0f}%")
row2[1].metric("Avg ROM", f"{analytics.average_rom_deg:.0f}\u00b0")
row2[2].metric("Compensation Freq.", f"{analytics.compensation_frequency:.1f}/100f")
row2[3].metric("Fatigue Indicator", f"{analytics.fatigue_indicator:.0f}/100")

# ------------------------------------------------------------------
# Alerts + recommendations
# ------------------------------------------------------------------
alerts = alert_system.check_patient(patient.patient_id)
if alerts:
    st.subheader("Active Alerts")
    for alert in alerts:
        icon = {"critical": "🔴", "warning": "🟠", "info": "🔵"}.get(alert.severity, "⚪")
        st.write(f"{icon} {alert.message}")

recommendations = recommendation_engine.generate(analytics, exercise_display_name=sessions[0].exercise_name)
st.subheader("Recommendations")
for rec in recommendations:
    st.write(f"**[{rec.priority.upper()}]** {rec.text}")

st.divider()

# ------------------------------------------------------------------
# Build a per-session dataframe once, reused across all graphs
# ------------------------------------------------------------------
session_rows = []
for s in reversed(sessions):  # chronological
    detail = session_manager.get_session_detail(s.session_id)
    scores = metrics_calculator.compute_for_session(detail["frames"], detail["events"], detail["reps"])
    rom_values = [f["rom_deg"] for f in detail["frames"] if f.get("rom_deg") is not None]
    session_rows.append({
        "session_id": s.session_id,
        "date": s.started_at[:10],
        "datetime": s.started_at,
        "duration_seconds": s.duration_seconds or 0,
        "total_reps": s.total_reps or 0,
        "successful_reps": detail["successful_repetitions"],
        "incorrect_reps": detail["incorrect_repetitions"],
        "recovery_index": scores.recovery_index,
        "overall_score": scores.overall_rehabilitation_score,
        "rom": (max(rom_values) - min(rom_values)) if len(rom_values) > 1 else 0,
        "accuracy": (detail["successful_repetitions"] / max(1, len(detail["reps"]))) * 100,
        "compensation_events": sum(1 for e in detail["events"] if e.get("error_type") in
                                    ("trunk_compensation", "shoulder_hiking", "body_lean", "poor_alignment")),
    })
df = pd.DataFrame(session_rows)
df["date"] = pd.to_datetime(df["date"])

# ------------------------------------------------------------------
# Clinical 0-100 scores (most recent session)
# ------------------------------------------------------------------
st.subheader("Clinical Scores (Most Recent Session)")
latest_detail = session_manager.get_session_detail(sessions[0].session_id)
latest_scores = metrics_calculator.compute_for_session(latest_detail["frames"], latest_detail["events"], latest_detail["reps"])
score_dict = latest_scores.to_dict()

radar_fig = go.Figure()
radar_fig.add_trace(go.Scatterpolar(
    r=list(score_dict.values()), theta=[k.replace("_", " ").title() for k in score_dict.keys()],
    fill="toself", name="Latest session",
))
radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, height=450)
st.plotly_chart(radar_fig, use_container_width=True)

st.divider()

# ------------------------------------------------------------------
# Progress visualization: all required Plotly graphs
# ------------------------------------------------------------------
st.subheader("Progress Visualization")

graph_col1, graph_col2 = st.columns(2)
with graph_col1:
    st.plotly_chart(px.line(df, x="date", y="recovery_index", markers=True, title="Recovery Score Over Time"), use_container_width=True)
with graph_col2:
    st.plotly_chart(px.line(df, x="date", y="rom", markers=True, title="ROM Improvement"), use_container_width=True)

graph_col3, graph_col4 = st.columns(2)
with graph_col3:
    st.plotly_chart(px.line(df, x="date", y="accuracy", markers=True, title="Exercise Accuracy Over Time"), use_container_width=True)
with graph_col4:
    st.plotly_chart(px.bar(df, x="date", y="compensation_events", title="Compensation Frequency"), use_container_width=True)

graph_col5, graph_col6 = st.columns(2)
with graph_col5:
    st.plotly_chart(px.bar(df, x="date", y="duration_seconds", title="Session Duration"), use_container_width=True)
with graph_col6:
    compliance_df = df.copy()
    compliance_df["compliance_pct"] = (compliance_df["successful_reps"] / compliance_df["total_reps"].clip(lower=1)) * 100
    st.plotly_chart(px.line(compliance_df, x="date", y="compliance_pct", markers=True, title="Exercise Compliance"), use_container_width=True)

# Joint angle trends (from raw frames of the most recent session)
st.subheader("Joint Angle Trends (Most Recent Session)")
angle_series: dict = {}
for frame in latest_detail["frames"]:
    for name, value in (frame.get("joint_angles") or {}).items():
        if value is not None:
            angle_series.setdefault(name, []).append(value)

trackable_angles = [k for k in angle_series if "elbow" in k or "shoulder_angle" in k]
if trackable_angles:
    angle_fig = go.Figure()
    for name in trackable_angles[:4]:
        angle_fig.add_trace(go.Scatter(y=angle_series[name], mode="lines", name=name.replace("_", " ").title()))
    angle_fig.update_layout(title="Joint Angle Trends", xaxis_title="Frame", yaxis_title="Degrees", height=400)
    st.plotly_chart(angle_fig, use_container_width=True)
else:
    st.caption("No joint-angle data available for the most recent session.")

# Weekly / Monthly progress
st.subheader("Weekly and Monthly Progress")
weekly_col, monthly_col = st.columns(2)
with weekly_col:
    weekly_df = df.set_index("date").resample("W")["recovery_index"].mean().reset_index()
    st.plotly_chart(px.bar(weekly_df, x="date", y="recovery_index", title="Weekly Progress (avg Recovery Score)"), use_container_width=True)
with monthly_col:
    monthly_df = df.set_index("date").resample("ME")["recovery_index"].mean().reset_index()
    st.plotly_chart(px.bar(monthly_df, x="date", y="recovery_index", title="Monthly Progress (avg Recovery Score)"), use_container_width=True)

st.divider()

# ------------------------------------------------------------------
# Session comparison
# ------------------------------------------------------------------
st.subheader("Session Comparison")
comparison_type = st.selectbox("Compare most recent session against", ["previous_session", "previous_week", "previous_month"])
comparison = session_manager.compare_sessions(sessions[0].session_id, comparison=comparison_type)

if comparison["baseline"] is None:
    st.info(comparison.get("note", "No baseline session available for this comparison."))
else:
    verdict_icons = {"improvement": "🟢", "regression": "🔴", "stable": "🟡"}
    for metric, verdict in comparison["verdicts"].items():
        st.write(f"{verdict_icons.get(verdict, '⚪')} **{metric.title()}**: {verdict.title()}")
