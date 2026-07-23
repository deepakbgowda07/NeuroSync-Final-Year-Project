"""
db.py
=====
SQLite persistence layer for the dashboard: patients, sessions, and
per-session exercise results. Kept intentionally simple (stdlib
sqlite3) for a single-clinic / research-lab deployment; swap for a
proper ORM + Postgres if this becomes multi-tenant.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from utils.file_io import ensure_dir
from utils.logger import get_logger

logger = get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name TEXT NOT NULL,
    date_of_birth TEXT,
    stroke_onset_date TEXT,
    affected_side TEXT,
    notes TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER,
    exercise_name TEXT NOT NULL,
    started_at TEXT DEFAULT CURRENT_TIMESTAMP,
    ended_at TEXT,
    mean_confidence REAL,
    duration_seconds REAL,
    total_reps INTEGER,
    mean_quality REAL,
    notes TEXT,
    FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
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
"""


def init_db(db_path: str) -> None:
    ensure_dir(Path(db_path).parent)
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA)
    logger.info("Database initialized at %s", db_path)


@contextmanager
def get_connection(db_path: str) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
