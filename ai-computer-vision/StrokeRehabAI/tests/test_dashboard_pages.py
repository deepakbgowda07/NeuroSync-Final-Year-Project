"""Smoke tests for all Streamlit dashboard pages using Streamlit's
AppTest harness — executes each page script headlessly against a
synthetic database and asserts no unhandled exception occurs.

Skipped automatically if streamlit's AppTest module isn't available
(older Streamlit versions).
"""

import json
from datetime import datetime, timedelta

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = st_testing.AppTest

from dashboard.db import get_connection, init_db


@pytest.fixture
def seeded_db(tmp_path, monkeypatch):
    """Seed a synthetic database and point configs.dashboard.database_path at it."""
    db_path = str(tmp_path / "dashboard_test.db")
    init_db(db_path)

    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO patients (patient_id, full_name, date_of_birth, gender, affected_side, diagnosis, physiotherapist) "
            "VALUES (1, 'Jane Doe', '1970-05-15', 'Female', 'Left', 'Ischemic stroke', 'Dr. Smith')"
        )
        for i in range(2):
            sid = i + 1
            started = (datetime.now() - timedelta(days=(2 - i) * 3)).isoformat()
            conn.execute(
                "INSERT INTO sessions (session_id, patient_id, exercise_name, started_at, duration_seconds, total_reps, mean_quality, mean_confidence) "
                "VALUES (?, 1, 'Elbow Flexion', ?, 120, 6, 0.75, 0.85)",
                (sid, started),
            )
            for f in range(15):
                angle = 170 - f * 5
                conn.execute(
                    "INSERT INTO session_frames (session_id, movement_quality, rom_deg, joint_angles_json, phase, exercise_name) VALUES (?, 0.75, ?, ?, 'peak', 'Elbow Flexion')",
                    (sid, angle, json.dumps({"left_elbow_angle": angle, "left_shoulder_angle": 30})),
                )
            for r in range(6):
                conn.execute("INSERT INTO session_reps (session_id, rep_number, completed) VALUES (?, ?, 1)", (sid, r))

    from configs.config_loader import load_config

    cfg = load_config(force_reload=True)
    monkeypatch.setattr(cfg.dashboard, "database_path", db_path)

    # config_loader caches a singleton; patch the cached instance directly too.
    import configs.config_loader as config_loader_module
    if config_loader_module._cached_config is not None:
        config_loader_module._cached_config["dashboard"]["database_path"] = db_path

    return db_path


PAGES = [
    "dashboard/app.py",
    "dashboard/pages/1_Patient_Management.py",
    "dashboard/pages/2_Live_Session.py",
    "dashboard/pages/3_Exercise_History.py",
    "dashboard/pages/4_Recovery_Analytics.py",
    "dashboard/pages/5_Reports.py",
    "dashboard/pages/6_Settings.py",
    "dashboard/pages/7_About.py",
]


@pytest.mark.parametrize("page_path", PAGES)
def test_page_runs_without_exception(seeded_db, page_path):
    at = AppTest.from_file(page_path)
    at.run(timeout=30)
    assert not at.exception, f"{page_path} raised: {[str(e) for e in at.exception]}"
