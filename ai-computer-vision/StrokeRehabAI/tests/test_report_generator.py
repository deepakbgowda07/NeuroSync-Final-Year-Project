"""Tests for dashboard.report_generator.ReportGenerator (PDF/Excel/CSV/JSON export)."""

import json
from datetime import datetime, timedelta

import pytest

from dashboard.db import get_connection, init_db
from dashboard.report_generator import ReportGenerator

reportlab = pytest.importorskip("reportlab")
openpyxl = pytest.importorskip("openpyxl")


@pytest.fixture
def db_path(tmp_path):
    path = str(tmp_path / "test.db")
    init_db(path)
    with get_connection(path) as conn:
        conn.execute(
            "INSERT INTO patients (patient_id, full_name, date_of_birth, gender, affected_side, diagnosis, physiotherapist) "
            "VALUES (1, 'Jane Doe', '1970-05-15', 'Female', 'Left', 'Ischemic stroke', 'Dr. Smith')"
        )
        for i in range(2):
            sid = i + 1
            started = (datetime.now() - timedelta(days=(2 - i) * 3)).isoformat()
            conn.execute(
                "INSERT INTO sessions (session_id, patient_id, exercise_name, started_at, duration_seconds, total_reps, mean_quality) "
                "VALUES (?, 1, 'Elbow Flexion', ?, 120, 6, 0.75)",
                (sid, started),
            )
            for f in range(15):
                angle = 170 - f * 5
                conn.execute(
                    "INSERT INTO session_frames (session_id, movement_quality, rom_deg, joint_angles_json) VALUES (?, 0.75, ?, ?)",
                    (sid, angle, json.dumps({"left_elbow_angle": angle})),
                )
            for r in range(6):
                conn.execute("INSERT INTO session_reps (session_id, rep_number, completed) VALUES (?, ?, 1)", (sid, r))
            if i == 1:
                conn.execute(
                    "INSERT INTO session_events (session_id, event_type, error_type, severity) VALUES (?, 'error', 'trunk_compensation', 'moderate')",
                    (sid,),
                )
    return path


@pytest.fixture
def generator(db_path, tmp_path):
    return ReportGenerator(db_path, output_dir=str(tmp_path / "reports"))


@pytest.mark.parametrize("fmt,extension", [("json", ".json"), ("csv", ".csv"), ("excel", ".xlsx"), ("pdf", ".pdf")])
def test_generate_produces_nonempty_file(generator, fmt, extension):
    path = generator.generate(patient_id=1, report_format=fmt)
    assert path.exists()
    assert path.suffix == extension
    assert path.stat().st_size > 0


def test_generate_logs_to_generated_reports_table(generator, db_path):
    generator.generate(patient_id=1, report_format="json")
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM generated_reports WHERE patient_id = 1").fetchall()
    assert len(rows) == 1
    assert rows[0]["report_type"] == "json"


def test_unknown_format_raises(generator):
    with pytest.raises(ValueError):
        generator.generate(patient_id=1, report_format="xml")


def test_unknown_patient_raises(generator):
    with pytest.raises(ValueError):
        generator.generate(patient_id=999, report_format="json")


def test_json_report_contains_all_required_sections(generator):
    path = generator.generate(patient_id=1, report_format="json")
    with open(path) as fh:
        data = json.load(fh)

    required_sections = {
        "patient_information", "exercise_summary", "recovery_metrics", "joint_analysis",
        "rom_analysis", "compensation_analysis", "recommendations", "session_summary",
    }
    assert required_sections.issubset(data.keys())
    assert data["patient_information"]["full_name"] == "Jane Doe"
    assert data["compensation_analysis"].get("trunk_compensation") == 1
    assert len(data["session_summary"]) == 2
