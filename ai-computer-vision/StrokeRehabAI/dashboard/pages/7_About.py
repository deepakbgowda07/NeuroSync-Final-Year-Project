"""
About page.

Project information: what the platform is, the underlying pipeline,
tech stack, and where to find deeper documentation.
"""

from __future__ import annotations

import streamlit as st

from configs.config_loader import load_config
from utils.gpu_utils import get_gpu_info

st.title("About")

cfg = load_config()

st.markdown(
    f"""
    ## {cfg.dashboard.title}

    A lightweight clinical rehabilitation monitoring platform for
    physiotherapists and doctors overseeing stroke rehabilitation —
    **not** a consumer fitness app.

    ### What this platform does
    Patients perform prescribed exercises in front of a laptop webcam.
    A real-time computer-vision engine (MediaPipe BlazePose + an
    ST-GCN movement-quality model) automatically recognizes the
    exercise, tracks repetitions, detects movement-quality errors, and
    gives the patient live natural-language feedback. Every session is
    logged here, where physiotherapists can track recovery trends,
    review session history and replay, generate clinical reports, and
    receive automated alerts when a patient's recovery, compensation
    patterns, range of motion, or adherence change meaningfully.

    ### Pipeline
    ```
    Webcam → MediaPipe Pose → Movement Analysis → Real-Time Feedback
                                        │
                                        ▼
                              This Dashboard (SQLite)
                     Patients · Sessions · Recovery Analytics · Reports
    ```

    ### Tech stack
    - **Computer vision:** MediaPipe BlazePose, OpenCV
    - **Model:** PyTorch (ST-GCN, with an LSTM fallback), CUDA-accelerated where available
    - **Dashboard:** Streamlit, Plotly, Pandas, Matplotlib
    - **Reporting:** ReportLab (PDF), OpenPyXL (Excel)
    - **Storage:** SQLite

    ### Documentation
    See the `docs/` folder in the project repository for the full
    architecture, installation, training, and inference guides.
    """
)

st.divider()
st.subheader("System Status")

col1, col2 = st.columns(2)
with col1:
    st.write("**Active model architecture:**", cfg.model.architecture)
    st.write("**Active dataset (training):**", cfg.datasets.active_dataset)
with col2:
    gpu_info = get_gpu_info(cfg.gpu.device_index)
    st.write("**CUDA available:**", gpu_info.available)
    if gpu_info.available:
        st.write("**GPU:**", gpu_info.device_name)
