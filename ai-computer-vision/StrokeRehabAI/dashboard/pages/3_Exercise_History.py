"""
Exercise History page.

Lists past sessions, optionally filtered by patient, with links to
per-session detail (joint-angle time series, prediction confidence).
"""

import streamlit as st

from configs.config_loader import load_config
from dashboard.db import get_connection

st.title("Exercise History")

cfg = load_config()
db_path = cfg.dashboard.database_path

with get_connection(db_path) as conn:
    patients = conn.execute("SELECT patient_id, full_name FROM patients").fetchall()

patient_options = {"All patients": None}
patient_options.update({row["full_name"]: row["patient_id"] for row in patients})

selected = st.selectbox("Filter by patient", list(patient_options.keys()))
selected_id = patient_options[selected]

query = "SELECT * FROM sessions"
params = ()
if selected_id is not None:
    query += " WHERE patient_id = ?"
    params = (selected_id,)
query += " ORDER BY started_at DESC"

with get_connection(db_path) as conn:
    sessions = conn.execute(query, params).fetchall()

if sessions:
    st.dataframe([dict(row) for row in sessions], use_container_width=True)
else:
    st.info("No sessions recorded yet.")
