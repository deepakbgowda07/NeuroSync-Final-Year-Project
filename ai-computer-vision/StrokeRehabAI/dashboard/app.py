"""
app.py
======
Streamlit dashboard entry point. Run via:

    streamlit run dashboard/app.py

Page navigation follows the multipage pattern (files under
dashboard/pages/). This file sets global page config, initializes the
SQLite database, and renders the Home page content.
"""

from __future__ import annotations

import streamlit as st

from configs.config_loader import load_config
from dashboard.db import init_db
from utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    cfg = load_config()
    st.set_page_config(page_title=cfg.dashboard.title, layout=cfg.dashboard.layout)

    init_db(cfg.dashboard.database_path)

    st.title(cfg.dashboard.title)
    st.caption("AI-Assisted Stroke Rehabilitation Exercise Assessment System")

    st.markdown(
        """
        ### Welcome

        Use the sidebar to navigate between:
        - **Patient Management** — add/edit patient records
        - **Live Session** — run a real-time camera-based exercise session
        - **Exercise History** — review past sessions per patient
        - **Recovery Analytics** — trend charts across sessions
        - **Reports** — export PDF/CSV clinical summaries
        - **Settings** — camera, model, and dashboard configuration

        > **Status:** This is a project scaffold. The assessment model has
        > not yet been trained on clinical data (see `training/trainer.py`),
        > so Live Session predictions are illustrative only until a
        > validated checkpoint is provided.
        """
    )


if __name__ == "__main__":
    main()
