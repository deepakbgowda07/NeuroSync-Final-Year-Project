"""
Recovery Analytics page.

Trend charts of exercise-quality metrics across sessions over time
per patient — the primary "is the patient improving" view for
clinicians.

TODO (next development phase): once real session data accumulates,
compute a rolling ROM-error trend and confidence trend per exercise
type, and surface a simple improving/plateaued/regressing indicator.
"""

import streamlit as st

st.title("Recovery Analytics")

st.info(
    "Analytics will populate once Live Sessions have been recorded "
    "with a trained model checkpoint. This page is a scaffold pending "
    "real session data."
)

st.markdown(
    """
    Planned charts:
    - Prediction confidence trend over time
    - Range-of-motion (ROM) error trend per exercise
    - Session frequency / adherence calendar
    - Bilateral symmetry index trend (see `feature_extraction/kinematic_features.py`)
    """
)
