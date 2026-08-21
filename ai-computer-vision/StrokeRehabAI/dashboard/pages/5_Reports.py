"""
Reports page.

Generates downloadable clinical reports in PDF, Excel, CSV, or JSON —
each including patient information, exercise summary, recovery
metrics, joint/ROM/compensation analysis, recommendations, and a
session summary — and shows previously generated reports.
"""

from __future__ import annotations

import streamlit as st

from configs.config_loader import load_config
from dashboard.db import get_connection
from dashboard.patient_manager import PatientManager
from dashboard.report_generator import ReportGenerator

st.title("Reports")

cfg = load_config()
db_path = cfg.dashboard.database_path

patient_manager = PatientManager(db_path)
report_generator = ReportGenerator(db_path)

patients = patient_manager.list_all()
if not patients:
    st.info("No patients registered yet — add one on the Patients page.")
    st.stop()

selected_name = st.selectbox("Patient", [p.full_name for p in patients])
patient = next(p for p in patients if p.full_name == selected_name)

st.markdown(
    "Each report includes: patient information, exercise summary, recovery metrics, "
    "joint analysis, ROM analysis, compensation analysis, recommendations, and a session summary."
)

format_cols = st.columns(4)
format_labels = {"pdf": "📄 PDF", "excel": "📊 Excel", "csv": "📋 CSV", "json": "🗂️ JSON"}

for col, (fmt, label) in zip(format_cols, format_labels.items()):
    with col:
        if st.button(f"Generate {label}", key=f"generate_{fmt}"):
            try:
                path = report_generator.generate(patient.patient_id, report_format=fmt)
                with open(path, "rb") as fh:
                    st.download_button(
                        f"Download {label}", data=fh.read(), file_name=path.name, key=f"download_{fmt}",
                    )
                st.success(f"{label} report generated.")
            except Exception as exc:  # noqa: BLE001 - surface generation errors to the clinician
                st.error(f"Report generation failed: {exc}")

st.divider()
st.subheader("Previously Generated Reports")

with get_connection(db_path) as conn:
    reports = conn.execute(
        "SELECT * FROM generated_reports WHERE patient_id = ? ORDER BY generated_at DESC LIMIT 20",
        (patient.patient_id,),
    ).fetchall()

if reports:
    st.dataframe(
        [{"Type": r["report_type"].upper(), "File": r["file_path"], "Generated At": r["generated_at"]} for r in reports],
        use_container_width=True,
    )
else:
    st.caption("No reports generated yet for this patient.")
