"""
db.py
=====
SQLite persistence layer for the clinical dashboard: patients,
rehabilitation plans, sessions, exercises, exercise results, joint
metrics, recovery scores, compensation events, generated reports, and
settings. Kept intentionally simple (stdlib sqlite3) for a
single-clinic / research-lab deployment; swap for a proper ORM +
Postgres if this becomes multi-tenant.

Schema evolves additively: `init_db()` is safe to call repeatedly and
against a database created by an earlier version of this schema — new
tables are created if missing, and new columns on existing tables are
added via `_apply_migrations()` rather than requiring a fresh database.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Tuple

from utils.file_io import ensure_dir
from utils.logger import get_logger

logger = get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name TEXT NOT NULL,
    date_of_birth TEXT,
    gender TEXT,
    stroke_onset_date TEXT,
    affected_side TEXT,
    diagnosis TEXT,
    physiotherapist TEXT,
    treatment_start_date TEXT,
    notes TEXT,
    is_active INTEGER DEFAULT 1,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS rehabilitation_plans (
    plan_id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL,
    plan_name TEXT,
    exercises_json TEXT,          -- JSON list of exercise keys + target reps/sets
    sessions_per_week INTEGER,
    notes TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    is_active INTEGER DEFAULT 1,
    FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
);

CREATE TABLE IF NOT EXISTS exercises (
    exercise_key TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    category TEXT,                -- e.g. "shoulder", "elbow", "forearm", "reach"
    target_joint TEXT,
    description TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER,
    plan_id INTEGER,
    exercise_name TEXT NOT NULL,
    started_at TEXT DEFAULT CURRENT_TIMESTAMP,
    ended_at TEXT,
    mean_confidence REAL,
    duration_seconds REAL,
    total_reps INTEGER,
    mean_quality REAL,
    notes TEXT,
    original_video_path TEXT,
    processed_video_path TEXT,
    FOREIGN KEY (patient_id) REFERENCES patients (patient_id),
    FOREIGN KEY (plan_id) REFERENCES rehabilitation_plans (plan_id)
);

CREATE TABLE IF NOT EXISTS session_frames (
    frame_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    frame_timestamp REAL,
    exercise_name TEXT,
    phase TEXT,
    predicted_class INTEGER,
    confidence REAL,
    movement_quality REAL,
    rom_deg REAL,
    joint_angles_json TEXT,
    fps REAL,
    cuda_available INTEGER,
    session_elapsed_seconds REAL,
    landmarks_json TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
);

CREATE TABLE IF NOT EXISTS session_reps (
    rep_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    rep_number INTEGER,
    exercise_name TEXT,
    completed INTEGER,
    peak_progress_fraction REAL,
    duration_frames INTEGER,
    timestamp REAL,
    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
);

CREATE TABLE IF NOT EXISTS session_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    event_type TEXT NOT NULL,        -- e.g. "error", "compensation", "calibration"
    error_type TEXT,
    severity TEXT,
    detail_json TEXT,
    timestamp REAL,
    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
);

CREATE TABLE IF NOT EXISTS exercise_results (
    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    exercise_key TEXT,
    exercise_name TEXT,
    reps_successful INTEGER DEFAULT 0,
    reps_incorrect INTEGER DEFAULT 0,
    avg_score REAL,
    duration_seconds REAL,
    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
);

CREATE TABLE IF NOT EXISTS joint_metrics (
    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    joint_name TEXT NOT NULL,
    avg_angle_deg REAL,
    min_angle_deg REAL,
    max_angle_deg REAL,
    rom_deg REAL,
    angle_error_deg REAL,
    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
);

CREATE TABLE IF NOT EXISTS recovery_scores (
    score_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    patient_id INTEGER NOT NULL,
    shoulder_mobility_score REAL,
    elbow_mobility_score REAL,
    movement_stability REAL,
    movement_smoothness REAL,
    symmetry_score REAL,
    compensation_score REAL,
    exercise_quality_score REAL,
    recovery_index REAL,
    overall_rehabilitation_score REAL,
    computed_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions (session_id),
    FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
);

CREATE TABLE IF NOT EXISTS compensation_events (
    compensation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    compensation_type TEXT NOT NULL,   -- e.g. "trunk_compensation", "shoulder_hiking"
    joint_name TEXT,
    severity TEXT,
    magnitude REAL,
    timestamp REAL,
    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
);

CREATE TABLE IF NOT EXISTS generated_reports (
    report_id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER,
    session_id INTEGER,
    report_type TEXT NOT NULL,   -- "pdf" | "excel" | "csv" | "json"
    file_path TEXT NOT NULL,
    generated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients (patient_id),
    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
);

CREATE TABLE IF NOT EXISTS settings (
    setting_key TEXT PRIMARY KEY,
    setting_value TEXT,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

# (table, column, column_definition) additions applied to databases created
# by an earlier schema version, so existing installations upgrade in place.
_MIGRATIONS: List[Tuple[str, str, str]] = [
    ("patients", "gender", "TEXT"),
    ("patients", "diagnosis", "TEXT"),
    ("patients", "physiotherapist", "TEXT"),
    ("patients", "treatment_start_date", "TEXT"),
    ("patients", "is_active", "INTEGER DEFAULT 1"),
    ("patients", "updated_at", "TEXT"),
    ("sessions", "plan_id", "INTEGER"),
    ("session_frames", "fps", "REAL"),
    ("session_frames", "cuda_available", "INTEGER"),
    ("session_frames", "session_elapsed_seconds", "REAL"),
    ("sessions", "original_video_path", "TEXT"),
    ("sessions", "processed_video_path", "TEXT"),
    ("session_frames", "landmarks_json", "TEXT"),
]


def init_db(db_path: str) -> None:
    ensure_dir(Path(db_path).parent)
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA)
        _apply_migrations(conn)
        _seed_exercises(conn)
    logger.info("Database initialized at %s", db_path)


def _apply_migrations(conn: sqlite3.Connection) -> None:
    for table, column, definition in _MIGRATIONS:
        existing_columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing_columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
            logger.info("Migrated schema: added %s.%s", table, column)


def _seed_exercises(conn: sqlite3.Connection) -> None:
    """Populate the `exercises` catalog table from
    inference.exercise_library's 10 supported exercises, if not already present."""
    from inference.exercise_library import SUPPORTED_EXERCISES

    existing = {row["exercise_key"] for row in conn.execute("SELECT exercise_key FROM exercises")}
    categories = {
        "shoulder_flexion": ("shoulder", "shoulder"),
        "shoulder_abduction": ("shoulder", "shoulder"),
        "elbow_flexion": ("elbow", "elbow"),
        "elbow_extension": ("elbow", "elbow"),
        "forearm_pronation": ("forearm", "forearm"),
        "forearm_supination": ("forearm", "forearm"),
        "shoulder_external_rotation": ("shoulder", "shoulder"),
        "shoulder_internal_rotation": ("shoulder", "shoulder"),
        "hand_to_mouth": ("reach", "elbow"),
        "hand_to_head": ("reach", "elbow"),
    }
    for key in SUPPORTED_EXERCISES:
        if key in existing:
            continue
        category, joint = categories.get(key, ("general", "unknown"))
        conn.execute(
            "INSERT INTO exercises (exercise_key, display_name, category, target_joint, description) VALUES (?, ?, ?, ?, ?)",
            (key, key.replace("_", " ").title(), category, joint, f"Auto-registered exercise: {key}"),
        )


@contextmanager
def get_connection(db_path: str) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
