"""
Settings page.

Configuration for camera, model, exercise thresholds, dashboard theme,
export defaults, database backup, and logging. Live values from
configs/*.yaml are shown read-only (edit the YAML files directly to
change them permanently — see docs/developer_guide.md); dashboard-only
preferences (theme, default export format, backup schedule) are
persisted in the `settings` table so they survive restarts without
touching the YAML config files.
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import streamlit as st

from configs.config_loader import load_config
from dashboard.db import get_connection
from utils.gpu_utils import get_gpu_info

st.title("Settings")

cfg = load_config()
db_path = cfg.dashboard.database_path


def get_setting(key: str, default: str = "") -> str:
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT setting_value FROM settings WHERE setting_key = ?", (key,)).fetchone()
    return row["setting_value"] if row else default


def set_setting(key: str, value: str) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO settings (setting_key, setting_value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP) "
            "ON CONFLICT(setting_key) DO UPDATE SET setting_value = excluded.setting_value, updated_at = CURRENT_TIMESTAMP",
            (key, value),
        )


tab_camera, tab_model, tab_thresholds, tab_theme, tab_export, tab_backup, tab_logging = st.tabs(
    ["Camera", "Model", "Exercise Thresholds", "Theme", "Export", "Backup", "Logging"]
)

with tab_camera:
    st.subheader("Camera (read-only — edit configs/camera.yaml to change)")
    st.json(dict(cfg.camera))

with tab_model:
    st.subheader("Model (read-only — edit configs/model.yaml to change)")
    st.write(f"**Active architecture:** {cfg.model.architecture}")
    gpu_info = get_gpu_info(cfg.gpu.device_index)
    st.json(gpu_info.__dict__)
    st.json(dict(cfg.model))

with tab_thresholds:
    st.subheader("Exercise Thresholds (read-only — edit configs/exercises.yaml to change)")
    for key, definition in cfg.exercises.definitions.items():
        definition = dict(definition)
        with st.expander(definition.get("display_name", key)):
            st.json(definition)

with tab_theme:
    st.subheader("Dashboard Theme")
    current_theme = get_setting("dashboard_theme", cfg.dashboard.theme)
    new_theme = st.selectbox("Theme", ["light", "dark"], index=0 if current_theme == "light" else 1)
    if st.button("Save Theme"):
        set_setting("dashboard_theme", new_theme)
        st.success(f"Theme preference saved: {new_theme} (takes effect on next reload).")

with tab_export:
    st.subheader("Export Settings")
    current_format = get_setting("default_export_format", "pdf")
    new_format = st.selectbox("Default report format", ["pdf", "excel", "csv", "json"],
                               index=["pdf", "excel", "csv", "json"].index(current_format))
    if st.button("Save Export Preference"):
        set_setting("default_export_format", new_format)
        st.success(f"Default export format saved: {new_format}")

with tab_backup:
    st.subheader("Database Backup")
    st.write(f"Current database: `{db_path}`")

    if st.button("Create Backup Now"):
        backup_dir = Path("outputs/database/backups")
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"stroke_rehab_backup_{timestamp}.db"
        shutil.copy(db_path, backup_path)
        set_setting("last_backup_path", str(backup_path))
        set_setting("last_backup_at", datetime.now().isoformat())
        st.success(f"Backup created: {backup_path}")

    last_backup_path = get_setting("last_backup_path")
    last_backup_at = get_setting("last_backup_at")
    if last_backup_path:
        st.caption(f"Last backup: {last_backup_at} → {last_backup_path}")
    else:
        st.caption("No backups created yet.")

with tab_logging:
    st.subheader("Logging (read-only — edit configs/logging.yaml to change)")
    st.json(dict(cfg.logging))
    st.caption(f"Log directory: `{cfg.logging.log_dir}`")
