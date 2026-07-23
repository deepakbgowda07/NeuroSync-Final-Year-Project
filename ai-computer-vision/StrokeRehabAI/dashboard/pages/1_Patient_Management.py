"""
Patient Management page.

Add new patients and browse/edit existing patient records stored in
the local SQLite database (dashboard/db.py).
"""

import streamlit as st

from configs.config_loader import load_config
from dashboard.db import get_connection

st.title("Patient Management")

cfg = load_config()
db_path = cfg.dashboard.database_path

with st.form("add_patient_form"):
    st.subheader("Add New Patient")
    full_name = st.text_input("Full name")
    date_of_birth = st.date_input("Date of birth")
    stroke_onset_date = st.date_input("Stroke onset date")
    affected_side = st.selectbox("Affected side", ["Left", "Right", "Bilateral", "Unknown"])
    notes = st.text_area("Clinical notes")
    submitted = st.form_submit_button("Add Patient")

    if submitted:
        if not full_name:
            st.error("Full name is required.")
        else:
            with get_connection(db_path) as conn:
                conn.execute(
                    "INSERT INTO patients (full_name, date_of_birth, stroke_onset_date, affected_side, notes) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (full_name, str(date_of_birth), str(stroke_onset_date), affected_side, notes),
                )
            st.success(f"Patient '{full_name}' added.")

st.subheader("Existing Patients")
with get_connection(db_path) as conn:
    rows = conn.execute("SELECT * FROM patients ORDER BY created_at DESC").fetchall()

if rows:
    st.dataframe([dict(row) for row in rows], use_container_width=True)
else:
    st.info("No patients recorded yet.")
