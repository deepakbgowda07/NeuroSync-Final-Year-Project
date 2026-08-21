"""
Patients page.

Register, edit, delete (deactivate), view, and search patients, and
assign rehabilitation plans — the full patient-management workflow for
physiotherapists and doctors.
"""

import streamlit as st

from configs.config_loader import load_config
from dashboard.patient_manager import Patient, PatientManager
from inference.exercise_library import SUPPORTED_EXERCISES

st.title("Patients")

cfg = load_config()
manager = PatientManager(cfg.dashboard.database_path)

tab_register, tab_directory, tab_plan = st.tabs(["Register / Edit", "Patient Directory", "Assign Plan"])

# ----------------------------------------------------------------------
# Register / Edit
# ----------------------------------------------------------------------
with tab_register:
    st.subheader("Register New Patient")
    with st.form("register_patient_form"):
        col1, col2 = st.columns(2)
        with col1:
            full_name = st.text_input("Full name*")
            date_of_birth = st.date_input("Date of birth", value=None)
            gender = st.selectbox("Gender", ["Female", "Male", "Other", "Prefer not to say"])
            affected_side = st.selectbox("Affected side", ["Left", "Right", "Bilateral", "Unknown"])
        with col2:
            diagnosis = st.text_input("Diagnosis")
            physiotherapist = st.text_input("Physiotherapist")
            treatment_start_date = st.date_input("Treatment start date", value=None)
            stroke_onset_date = st.date_input("Stroke onset date", value=None)
        notes = st.text_area("Clinical notes")
        submitted = st.form_submit_button("Register Patient", type="primary")

        if submitted:
            if not full_name:
                st.error("Full name is required.")
            else:
                patient = Patient(
                    patient_id=None, full_name=full_name,
                    date_of_birth=str(date_of_birth) if date_of_birth else None,
                    gender=gender, affected_side=affected_side, diagnosis=diagnosis or None,
                    physiotherapist=physiotherapist or None,
                    treatment_start_date=str(treatment_start_date) if treatment_start_date else None,
                    stroke_onset_date=str(stroke_onset_date) if stroke_onset_date else None,
                    notes=notes or None,
                )
                new_id = manager.register(patient)
                st.success(f"Patient '{full_name}' registered (ID {new_id}).")
                st.rerun()

    st.divider()
    st.subheader("Edit Existing Patient")
    all_patients = manager.list_all(include_inactive=True)
    if all_patients:
        selected_name = st.selectbox("Select patient to edit", [p.full_name for p in all_patients], key="edit_select")
        selected_patient = next(p for p in all_patients if p.full_name == selected_name)

        with st.form("edit_patient_form"):
            new_diagnosis = st.text_input("Diagnosis", value=selected_patient.diagnosis or "")
            new_physio = st.text_input("Physiotherapist", value=selected_patient.physiotherapist or "")
            new_notes = st.text_area("Clinical notes", value=selected_patient.notes or "")
            edit_submitted = st.form_submit_button("Save Changes")

            if edit_submitted:
                manager.edit(selected_patient.patient_id, diagnosis=new_diagnosis, physiotherapist=new_physio, notes=new_notes)
                st.success("Patient updated.")
                st.rerun()

        if selected_patient.is_active:
            if st.button("Deactivate Patient (soft delete)"):
                manager.delete(selected_patient.patient_id)
                st.warning(f"Patient '{selected_patient.full_name}' deactivated.")
                st.rerun()
        else:
            st.info("This patient is currently deactivated.")
    else:
        st.info("No patients registered yet.")

# ----------------------------------------------------------------------
# Directory / search / view
# ----------------------------------------------------------------------
with tab_directory:
    search_query = st.text_input("Search by name, diagnosis, or physiotherapist")
    include_inactive = st.checkbox("Include deactivated patients", value=False)

    results = manager.search(search_query, include_inactive=include_inactive) if search_query else manager.list_all(include_inactive=include_inactive)

    if not results:
        st.info("No matching patients found.")
    else:
        st.dataframe(
            [
                {
                    "ID": p.patient_id, "Name": p.full_name, "Age": p.age, "Gender": p.gender,
                    "Affected Side": p.affected_side, "Diagnosis": p.diagnosis,
                    "Physiotherapist": p.physiotherapist, "Active": p.is_active,
                }
                for p in results
            ],
            use_container_width=True,
        )

# ----------------------------------------------------------------------
# Rehabilitation plan assignment
# ----------------------------------------------------------------------
with tab_plan:
    active_patients = manager.list_all()
    if not active_patients:
        st.info("Register a patient first to assign a rehabilitation plan.")
    else:
        plan_patient_name = st.selectbox("Patient", [p.full_name for p in active_patients], key="plan_select")
        plan_patient = next(p for p in active_patients if p.full_name == plan_patient_name)

        current_plan = manager.get_active_plan(plan_patient.patient_id)
        if current_plan:
            st.caption(f"Current active plan: **{current_plan['plan_name']}** ({current_plan['sessions_per_week']}x/week)")
            st.write(", ".join(e.replace("_", " ").title() for e in current_plan["exercises"]))

        with st.form("assign_plan_form"):
            plan_name = st.text_input("Plan name", value="Weekly Rehabilitation Plan")
            selected_exercises = st.multiselect(
                "Exercises", [e.replace("_", " ").title() for e in SUPPORTED_EXERCISES],
            )
            sessions_per_week = st.slider("Sessions per week", 1, 7, 3)
            plan_notes = st.text_area("Plan notes")
            assign_submitted = st.form_submit_button("Assign Plan", type="primary")

            if assign_submitted:
                if not selected_exercises:
                    st.error("Select at least one exercise.")
                else:
                    exercise_keys = [e.lower().replace(" ", "_") for e in selected_exercises]
                    manager.assign_plan(plan_patient.patient_id, plan_name, exercise_keys, sessions_per_week, plan_notes)
                    st.success(f"Assigned plan '{plan_name}' to {plan_patient.full_name}.")
                    st.rerun()
