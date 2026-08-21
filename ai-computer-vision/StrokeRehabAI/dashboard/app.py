"""
app.py
======
Streamlit dashboard entry point (the "Home" page). Run via:

    streamlit run dashboard/app.py

This is a lightweight clinical rehabilitation platform for
physiotherapists and doctors — not a consumer fitness app. Page
navigation follows Streamlit's multipage pattern (files under
dashboard/pages/). This file sets global page config, initializes the
database, and shows an at-a-glance clinical overview: active patient
count, recent alerts across all patients, and quick links.
"""

from __future__ import annotations

import streamlit as st

from configs.config_loader import load_config
from dashboard.alert_system import AlertSystem
from dashboard.db import get_connection, init_db
from dashboard.patient_manager import PatientManager
from utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    cfg = load_config()
    st.set_page_config(page_title=cfg.dashboard.title, layout=cfg.dashboard.layout, page_icon="🏥")

    db_path = cfg.dashboard.database_path
    init_db(db_path)

    st.title(cfg.dashboard.title)
    st.caption("Clinical Rehabilitation Monitoring Platform — for physiotherapists and doctors")

    patient_manager = PatientManager(db_path)
    patients = patient_manager.list_all()

    with get_connection(db_path) as conn:
        total_sessions = conn.execute("SELECT COUNT(*) as cnt FROM sessions").fetchone()["cnt"]
        sessions_this_week = conn.execute(
            "SELECT COUNT(*) as cnt FROM sessions WHERE started_at >= datetime('now', '-7 days')"
        ).fetchone()["cnt"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Active Patients", len(patients))
    col2.metric("Total Sessions Logged", total_sessions)
    col3.metric("Sessions This Week", sessions_this_week)

    st.divider()
    st.subheader("Alerts Across All Patients")

    alert_system = AlertSystem(db_path)
    any_alerts = False
    for patient in patients:
        alerts = alert_system.check_patient(patient.patient_id)
        for alert in alerts:
            any_alerts = True
            icon = {"critical": "🔴", "warning": "🟠", "info": "🔵"}.get(alert.severity, "⚪")
            st.write(f"{icon} **{patient.full_name}** — {alert.message}")

    if not any_alerts:
        st.info("No active alerts across any patient.")

    st.divider()
    st.markdown(
        """
        ### Navigation
        - **Patients** — register, edit, search patients and assign rehabilitation plans
        - **Live Session** — run a real-time camera-based exercise session
        - **Recovery Analytics** — clinical scores, trends, and progress graphs per patient
        - **Exercise History** — full session history, replay, and session comparison
        - **Reports** — generate PDF / Excel / CSV / JSON clinical reports
        - **Settings** — camera, model, exercise thresholds, theme, backups
        - **About** — project information
        """
    )


if __name__ == "__main__":
    main()
