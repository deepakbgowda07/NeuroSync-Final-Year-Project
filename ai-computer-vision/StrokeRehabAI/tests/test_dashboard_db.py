"""Tests for dashboard.db: schema initialization, migrations, and exercise seeding."""

import sqlite3

import pytest

from dashboard.db import get_connection, init_db


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test.db")


def test_init_db_creates_all_required_tables(db_path):
    init_db(db_path)
    expected_tables = {
        "patients", "rehabilitation_plans", "exercises", "sessions", "session_frames",
        "session_reps", "session_events", "exercise_results", "joint_metrics",
        "recovery_scores", "compensation_events", "generated_reports", "settings",
    }
    with get_connection(db_path) as conn:
        tables = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert expected_tables.issubset(tables)


def test_init_db_seeds_all_ten_exercises(db_path):
    init_db(db_path)
    with get_connection(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) as cnt FROM exercises").fetchone()["cnt"]
    assert count == 10


def test_init_db_is_idempotent(db_path):
    init_db(db_path)
    init_db(db_path)  # should not raise or duplicate exercises
    with get_connection(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) as cnt FROM exercises").fetchone()["cnt"]
    assert count == 10


def test_migration_adds_missing_columns_to_legacy_schema(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE patients (patient_id INTEGER PRIMARY KEY, full_name TEXT)")
    conn.execute("CREATE TABLE sessions (session_id INTEGER PRIMARY KEY, patient_id INTEGER, exercise_name TEXT)")
    conn.commit()
    conn.close()

    init_db(db_path)

    with get_connection(db_path) as conn:
        patient_columns = {row["name"] for row in conn.execute("PRAGMA table_info(patients)")}
        session_columns = {row["name"] for row in conn.execute("PRAGMA table_info(sessions)")}

    assert {"gender", "diagnosis", "physiotherapist", "is_active"}.issubset(patient_columns)
    assert "plan_id" in session_columns


def test_settings_table_supports_upsert(db_path):
    init_db(db_path)
    with get_connection(db_path) as conn:
        conn.execute("INSERT INTO settings (setting_key, setting_value) VALUES ('theme', 'light')")
        conn.execute(
            "INSERT INTO settings (setting_key, setting_value) VALUES ('theme', 'dark') "
            "ON CONFLICT(setting_key) DO UPDATE SET setting_value = excluded.setting_value"
        )
        value = conn.execute("SELECT setting_value FROM settings WHERE setting_key = 'theme'").fetchone()["setting_value"]
    assert value == "dark"
