"""Tests for dashboard.patient_manager.PatientManager."""

import pytest

from dashboard.patient_manager import Patient, PatientManager


@pytest.fixture
def manager(tmp_path):
    return PatientManager(str(tmp_path / "test.db"))


def test_register_and_get_patient(manager):
    pid = manager.register(Patient(patient_id=None, full_name="Jane Doe", date_of_birth="1970-05-15"))
    patient = manager.get(pid)
    assert patient.full_name == "Jane Doe"
    assert patient.is_active


def test_age_computed_from_date_of_birth(manager):
    pid = manager.register(Patient(patient_id=None, full_name="Bob", date_of_birth="2000-01-01"))
    patient = manager.get(pid)
    assert patient.age >= 24  # sandbox clock is 2026+


def test_age_none_without_date_of_birth(manager):
    pid = manager.register(Patient(patient_id=None, full_name="No DOB"))
    patient = manager.get(pid)
    assert patient.age is None


def test_edit_updates_fields(manager):
    pid = manager.register(Patient(patient_id=None, full_name="Jane Doe"))
    manager.edit(pid, diagnosis="Ischemic stroke", notes="Recovering well")
    patient = manager.get(pid)
    assert patient.diagnosis == "Ischemic stroke"
    assert patient.notes == "Recovering well"


def test_edit_unknown_field_raises(manager):
    pid = manager.register(Patient(patient_id=None, full_name="Jane Doe"))
    with pytest.raises(ValueError):
        manager.edit(pid, not_a_real_field="x")


def test_soft_delete_deactivates_but_keeps_record(manager):
    pid = manager.register(Patient(patient_id=None, full_name="Jane Doe"))
    manager.delete(pid)
    assert manager.get(pid).is_active is False
    assert pid not in [p.patient_id for p in manager.list_all()]
    assert pid in [p.patient_id for p in manager.list_all(include_inactive=True)]


def test_hard_delete_blocked_if_sessions_exist(manager, tmp_path):
    from dashboard.db import get_connection

    pid = manager.register(Patient(patient_id=None, full_name="Jane Doe"))
    with get_connection(manager.db_path) as conn:
        conn.execute("INSERT INTO sessions (patient_id, exercise_name) VALUES (?, 'Elbow Flexion')", (pid,))

    with pytest.raises(ValueError, match="Cannot hard-delete"):
        manager.delete(pid, hard_delete=True)


def test_search_matches_name_diagnosis_and_physiotherapist(manager):
    manager.register(Patient(patient_id=None, full_name="Jane Doe", diagnosis="Ischemic stroke"))
    manager.register(Patient(patient_id=None, full_name="John Smith", physiotherapist="Dr. Lee"))

    assert len(manager.search("jane")) == 1
    assert len(manager.search("ischemic")) == 1
    assert len(manager.search("lee")) == 1
    assert len(manager.search("nonexistent")) == 0


def test_assign_plan_deactivates_previous_plan(manager):
    pid = manager.register(Patient(patient_id=None, full_name="Jane Doe"))
    manager.assign_plan(pid, "Plan A", ["elbow_flexion"], sessions_per_week=3)
    manager.assign_plan(pid, "Plan B", ["shoulder_flexion"], sessions_per_week=4)

    active = manager.get_active_plan(pid)
    assert active["plan_name"] == "Plan B"
    assert active["exercises"] == ["shoulder_flexion"]


def test_get_active_plan_none_when_no_plan_assigned(manager):
    pid = manager.register(Patient(patient_id=None, full_name="Jane Doe"))
    assert manager.get_active_plan(pid) is None
