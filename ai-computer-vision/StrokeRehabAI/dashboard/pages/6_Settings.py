"""
Settings page.

Read-only view of the current merged configuration (camera, model,
dashboard, GPU) for transparency/debugging. Editing writes back to the
YAML files under configs/.

TODO (next development phase): add a save-to-YAML action with
validation before allowing in-dashboard config edits.
"""

import streamlit as st

from configs.config_loader import load_config
from utils.gpu_utils import get_gpu_info

st.title("Settings")

cfg = load_config()

st.subheader("GPU")
gpu_info = get_gpu_info(cfg.gpu.device_index)
st.json(gpu_info.__dict__)

st.subheader("Camera")
st.json(dict(cfg.camera))

st.subheader("Model")
st.json(dict(cfg.model))

st.subheader("Dashboard")
st.json(dict(cfg.dashboard))
