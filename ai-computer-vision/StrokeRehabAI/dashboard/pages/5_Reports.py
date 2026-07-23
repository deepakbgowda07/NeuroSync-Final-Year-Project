"""
Reports page.

Generates downloadable per-patient or per-session reports (CSV/JSON
today; PDF planned) summarizing exercise history and progress.
"""

import streamlit as st

from configs.config_loader import load_config
from dashboard.db import get_connection
from utils.csv_export import write_dicts_to_csv

st.title("Reports")

cfg = load_config()
db_path = cfg.dashboard.database_path

with get_connection(db_path) as conn:
    sessions = conn.execute("SELECT * FROM sessions ORDER BY started_at DESC").fetchall()

if not sessions:
    st.info("No session data available yet to generate a report.")
else:
    rows = [dict(row) for row in sessions]
    st.dataframe(rows, use_container_width=True)

    if st.button("Export session history to CSV"):
        output_path = write_dicts_to_csv(rows, "outputs/evaluation_reports/session_history_export.csv")
        st.success(f"Exported to {output_path}")

st.markdown("**TODO (next development phase):** add a branded PDF report template (reports/pdf_report_builder.py).")
