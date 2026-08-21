"""
patient_manager.py
=====================
Patient management operations for the clinical dashboard: register,
edit, delete (soft-delete, to preserve session history integrity),
view, search, and rehabilitation-plan assignment.

Kept as a plain Python class (not Streamlit-coupled) so it's directly
unit-testable and reusable from any dashboard page.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

from dashboard.db import get_connection, init_db
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Patient:
    patient_id: Optional[int]
    full_name: str
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None
    stroke_onset_date: Optional[str] = None
    affected_side: Optional[str] = None
    diagnosis: Optional[str] = None
    physiotherapist: Optional[str] = None
    treatment_start_date: Optional[str] = None
    notes: Optional[str] = None
    is_active: bool = True

    @property
    def age(self) -> Optional[int]:
        """Computed age in years from date_of_birth (ISO 'YYYY-MM-DD')."""
        if not self.date_of_birth:
            return None
        try:
            year, month, day = (int(p) for p in self.date_of_birth.split("-"))
            birth = date(year, month, day)
            today = date.today()
            return today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
        except (ValueError, AttributeError):
            return None

    @classmethod
    def from_row(cls, row) -> "Patient":
        return cls(
            patient_id=row["patient_id"],
            full_name=row["full_name"],
            date_of_birth=row["date_of_birth"],
            gender=row["gender"] if "gender" in row.keys() else None,
            stroke_onset_date=row["stroke_onset_date"],
            affected_side=row["affected_side"],
            diagnosis=row["diagnosis"] if "diagnosis" in row.keys() else None,
            physiotherapist=row["physiotherapist"] if "physiotherapist" in row.keys() else None,
            treatment_start_date=row["treatment_start_date"] if "treatment_start_date" in row.keys() else None,
            notes=row["notes"],
            is_active=bool(row["is_active"]) if "is_active" in row.keys() and row["is_active"] is not None else True,
        )


class PatientManager:
    """CRUD + search operations over the `patients` table."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        init_db(db_path)

    def register(self, patient: Patient) -> int:
        """Register a new patient. Returns the new patient_id."""
        with get_connection(self.db_path) as conn:
            cursor = conn.execute(
                """INSERT INTO patients
                   (full_name, date_of_birth, gender, stroke_onset_date, affected_side,
                    diagnosis, physiotherapist, treatment_start_date, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    patient.full_name, patient.date_of_birth, patient.gender, patient.stroke_onset_date,
                    patient.affected_side, patient.diagnosis, patient.physiotherapist,
                    patient.treatment_start_date, patient.notes,
                ),
            )
            patient_id = cursor.lastrowid
        logger.info("Registered new patient: id=%d, name=%s", patient_id, patient.full_name)
        return patient_id

    def edit(self, patient_id: int, **fields) -> None:
        """Update arbitrary fields on an existing patient record."""
        if not fields:
            return
        allowed = {
            "full_name", "date_of_birth", "gender", "stroke_onset_date", "affected_side",
            "diagnosis", "physiotherapist", "treatment_start_date", "notes",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"Unknown patient field(s): {unknown}")

        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [patient_id]

        with get_connection(self.db_path) as conn:
            conn.execute(
                f"UPDATE patients SET {set_clause}, updated_at = CURRENT_TIMESTAMP WHERE patient_id = ?", values
            )
        logger.info("Updated patient id=%d: fields=%s", patient_id, list(fields.keys()))

    def delete(self, patient_id: int, hard_delete: bool = False) -> None:
        """Soft-deletes (default) by setting is_active=0, preserving session
        history for audit/clinical-record purposes. Pass hard_delete=True
        to actually remove the row (only if no sessions reference it)."""
        with get_connection(self.db_path) as conn:
            if hard_delete:
                has_sessions = conn.execute(
                    "SELECT COUNT(*) as cnt FROM sessions WHERE patient_id = ?", (patient_id,)
                ).fetchone()["cnt"]
                if has_sessions:
                    raise ValueError(
                        f"Cannot hard-delete patient {patient_id}: {has_sessions} session(s) reference them. "
                        "Use hard_delete=False (soft delete) instead."
                    )
                conn.execute("DELETE FROM patients WHERE patient_id = ?", (patient_id,))
                logger.warning("Hard-deleted patient id=%d.", patient_id)
            else:
                conn.execute(
                    "UPDATE patients SET is_active = 0, updated_at = CURRENT_TIMESTAMP WHERE patient_id = ?",
                    (patient_id,),
                )
                logger.info("Soft-deleted (deactivated) patient id=%d.", patient_id)

    def get(self, patient_id: int) -> Optional[Patient]:
        with get_connection(self.db_path) as conn:
            row = conn.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,)).fetchone()
        return Patient.from_row(row) if row else None

    def list_all(self, include_inactive: bool = False) -> List[Patient]:
        query = "SELECT * FROM patients"
        if not include_inactive:
            query += " WHERE is_active = 1"
        query += " ORDER BY full_name"
        with get_connection(self.db_path) as conn:
            rows = conn.execute(query).fetchall()
        return [Patient.from_row(r) for r in rows]

    def search(self, query: str, include_inactive: bool = False) -> List[Patient]:
        """Case-insensitive substring search across name, diagnosis, and physiotherapist."""
        like_query = f"%{query.lower()}%"
        sql = """SELECT * FROM patients
                 WHERE (LOWER(full_name) LIKE ? OR LOWER(COALESCE(diagnosis, '')) LIKE ?
                        OR LOWER(COALESCE(physiotherapist, '')) LIKE ?)"""
        params = [like_query, like_query, like_query]
        if not include_inactive:
            sql += " AND is_active = 1"
        sql += " ORDER BY full_name"

        with get_connection(self.db_path) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [Patient.from_row(r) for r in rows]

    # ------------------------------------------------------------------
    # Rehabilitation plans
    # ------------------------------------------------------------------

    def assign_plan(
        self, patient_id: int, plan_name: str, exercise_keys: List[str],
        sessions_per_week: int = 3, notes: Optional[str] = None,
    ) -> int:
        """Assign a rehabilitation plan (a set of exercises + frequency)
        to a patient, deactivating any previous active plan first."""
        with get_connection(self.db_path) as conn:
            conn.execute(
                "UPDATE rehabilitation_plans SET is_active = 0 WHERE patient_id = ? AND is_active = 1",
                (patient_id,),
            )
            cursor = conn.execute(
                """INSERT INTO rehabilitation_plans
                   (patient_id, plan_name, exercises_json, sessions_per_week, notes)
                   VALUES (?, ?, ?, ?, ?)""",
                (patient_id, plan_name, json.dumps(exercise_keys), sessions_per_week, notes),
            )
            plan_id = cursor.lastrowid
        logger.info("Assigned rehabilitation plan '%s' (id=%d) to patient id=%d.", plan_name, plan_id, patient_id)
        return plan_id

    def get_active_plan(self, patient_id: int) -> Optional[Dict]:
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM rehabilitation_plans WHERE patient_id = ? AND is_active = 1 ORDER BY created_at DESC LIMIT 1",
                (patient_id,),
            ).fetchone()
        if row is None:
            return None
        plan = dict(row)
        plan["exercises"] = json.loads(plan["exercises_json"]) if plan.get("exercises_json") else []
        return plan
