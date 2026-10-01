import io
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import json

from gemini_report import generate_neurosync_gemini_report

from neurosync_core import (
    build_calibration_dataset,
    train_personalized_model,
    analyze_new_recording,
    calculate_session_anomaly_summary,
)

st.set_page_config(
    page_title="NeuroSync EEG Dashboard",
    page_icon="🧠",
    layout="wide",
)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("## 🧠 NeuroSync")
    st.caption("Personalized EEG analysis + AI session interpretation")

    st.markdown("### Gemini AI Interpretation")
    st.caption(
        "Optional: generate a patient-friendly and doctor/rehabilitation-professional "
        "session report from the NeuroSync results."
    )

    env_gemini_key = os.getenv("GEMINI_API_KEY", "")
    gemini_api_key = st.text_input(
        "Gemini API Key",
        value=env_gemini_key,
        type="password",
        help="Your key is used only for the current Streamlit session. Do not hard-code it.",
    )
    if gemini_api_key:
        st.success("Gemini key ready")
    else:
        st.info("Add a Gemini API key to enable AI interpretation.")

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}
.main-title {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 0.1rem;
}
.subtitle {
    color: #777;
    margin-bottom: 1.5rem;
}
.section {
    font-size: 1.25rem;
    font-weight: 650;
    margin-top: 1.2rem;
    margin-bottom: 0.6rem;
}
.metric-card {
    padding: 1rem;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,.25);
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🧠 NeuroSync</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Personalized EEG Brain-State Analysis Dashboard</div>',
    unsafe_allow_html=True
)

# -----------------------------
# Session state
# -----------------------------
if "personal_model" not in st.session_state:
    st.session_state.personal_model = None

if "gemini_report" not in st.session_state:
    st.session_state.gemini_report = None
if "gemini_session_summary" not in st.session_state:
    st.session_state.gemini_session_summary = None

if "calibration_df" not in st.session_state:
    st.session_state.calibration_df = None

if "result" not in st.session_state:
    st.session_state.result = None

if "session_summary" not in st.session_state:
    st.session_state.session_summary = None


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("NeuroSync")
    st.caption("EEG analysis configuration")
    st.info(
        "Expected EEG format:\n\n"
        "`time,ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8`\n\n"
        "Default data: 256 Hz\n"
        "Analysis window: 4 sec\n"
        "Step: 2 sec"
    )

    if st.session_state.personal_model is not None:
        st.success("Personal model calibrated")
    else:
        st.warning("Model not calibrated")


# -----------------------------
# Calibration
# -----------------------------
st.markdown('<div class="section">1. Personal Calibration</div>', unsafe_allow_html=True)

st.write(
    "Upload one EEG CSV for each calibration state. "
    "The original NeuroSync pipeline uses HAPPY, NORMAL, SAD and ANGRY."
)

c1, c2 = st.columns(2)
with c1:
    happy_file = st.file_uploader(
        "😊 HAPPY EEG",
        type=["csv"],
        key="happy",
    )
    normal_file = st.file_uploader(
        "😐 NORMAL EEG",
        type=["csv"],
        key="normal",
    )

with c2:
    sad_file = st.file_uploader(
        "😔 SAD EEG",
        type=["csv"],
        key="sad",
    )
    angry_file = st.file_uploader(
        "😠 ANGRY EEG",
        type=["csv"],
        key="angry",
    )

calibrate = st.button(
    "🔧 Build Personal Calibration & Train Model",
    type="primary",
    use_container_width=True,
)

if calibrate:
    files = {
        "happy": happy_file,
        "normal": normal_file,
        "sad": sad_file,
        "angry": angry_file,
    }

    missing = [state.upper() for state, f in files.items() if f is None]

    if missing:
        st.error("Please upload: " + ", ".join(missing))
    else:
        try:
            recordings = []

            for state, uploaded in files.items():
                recordings.append(
                    (
                        io.BytesIO(uploaded.getvalue()),
                        state,
                    )
                )

            with st.spinner("Processing calibration recordings and training model..."):
                calibration_df = build_calibration_dataset(recordings)

                # Keep the same CSV output produced by the original pipeline.
                calibration_df.to_csv(
                    "neurosync_personal_calibration.csv",
                    index=False,
                )

                personal_model = train_personalized_model(calibration_df)

            st.session_state.calibration_df = calibration_df
            st.session_state.personal_model = personal_model
            st.session_state.result = None
            st.session_state.session_summary = None

            st.success("Calibration complete. Personal EEG model is ready.")

        except Exception as e:
            st.error(f"Calibration failed: {e}")


# -----------------------------
# Calibration summary
# -----------------------------
if st.session_state.calibration_df is not None:
    st.markdown('<div class="section">Calibration Summary</div>', unsafe_allow_html=True)

    calibration_df = st.session_state.calibration_df

    counts = (
        calibration_df["label"]
        .value_counts()
        .rename_axis("State")
        .reset_index(name="Windows")
    )

    a, b, c = st.columns(3)
    a.metric("Calibration States", calibration_df["label"].nunique())
    b.metric("Total Windows", len(calibration_df))
    c.metric("Normal Windows", int((calibration_df["label"] == "normal").sum()))

    st.dataframe(counts, use_container_width=True, hide_index=True)


# -----------------------------
# New EEG analysis
# -----------------------------
st.markdown('<div class="section">2. Analyze New EEG Recording</div>', unsafe_allow_html=True)

test_file = st.file_uploader(
    "Upload a NEW unlabeled EEG CSV",
    type=["csv"],
    key="test",
    help="This recording should not have a mental-state label.",
)

analyze = st.button(
    "🧠 Analyze EEG",
    type="primary",
    use_container_width=True,
)

if analyze:
    if st.session_state.personal_model is None:
        st.error("Please complete personal calibration first.")
    elif test_file is None:
        st.error("Please upload a new EEG CSV.")
    else:
        try:
            with st.spinner("Running NeuroSync EEG analysis..."):
                result = analyze_new_recording(
                    io.BytesIO(test_file.getvalue()),
                    st.session_state.personal_model,
                )

                session_summary = calculate_session_anomaly_summary(result)

                result.to_csv(
                    "neurosync_personal_brain_state_results.csv",
                    index=False,
                )

            st.session_state.result = result
            st.session_state.session_summary = session_summary

            st.success("EEG analysis complete.")

        except Exception as e:
            st.error(f"Analysis failed: {e}")


# -----------------------------
# Results
# -----------------------------
result = st.session_state.result

if result is not None and not result.empty:
    latest = result.iloc[-1]
    summary = st.session_state.session_summary

    st.markdown('<div class="section">3. NeuroSync Brain State</div>', unsafe_allow_html=True)

    st.write(
        f"**Time:** {latest['start_time']:.1f} - {latest['end_time']:.1f} sec"
    )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Brain State",
        str(latest["predicted_state"]).upper(),
    )
    m2.metric(
        "State Category",
        str(latest["predicted_polarity"]),
    )
    m3.metric(
        "Confidence",
        f"{latest['state_confidence'] * 100:.1f}%",
    )
    m4.metric(
        "Brain-State Score",
        f"{latest['brain_state_score']:.1f}/100",
    )

    st.markdown("#### Indicators")

    i1, i2, i3, i4, i5 = st.columns(5)

    i1.metric(
        "Cognitive Load",
        f"{latest['cognitive_load_level']}",
        f"{latest['cognitive_load_score']:.1f}",
    )
    i2.metric(
        "Relaxation",
        f"{latest['relaxation_level']}",
        f"{latest['relaxation_score']:.1f}",
    )
    i3.metric(
        "Activation",
        f"{latest['activation_level']}",
        f"{latest['activation_score']:.1f}",
    )
    i4.metric(
        "Fatigue-related",
        f"{latest['fatigue_level']}",
        f"{latest['fatigue_indicator']:.1f}",
    )
    i5.metric(
        "EEG Stability",
        f"{latest['stability_level']}",
        f"{latest['eeg_stability_score']:.1f}",
    )

    st.markdown("#### EEG Deviation")

    d1, d2, d3 = st.columns(3)

    d1.metric(
        "Deviation Status",
        str(latest["deviation_status"]),
    )
    d2.metric(
        "Deviation Score",
        f"{latest['anomaly_score']:.3f}",
    )
    d3.metric(
        "Anomaly Flag",
        str(latest["anomaly_prediction"]),
    )

    st.markdown("#### EEG Bands")

    b1, b2, b3, b4, b5 = st.columns(5)

    b1.metric("Delta", f"{latest['delta_relative'] * 100:.2f}%")
    b2.metric("Theta", f"{latest['theta_relative'] * 100:.2f}%")
    b3.metric("Alpha", f"{latest['alpha_relative'] * 100:.2f}%")
    b4.metric("Beta", f"{latest['beta_relative'] * 100:.2f}%")
    b5.metric("Gamma", f"{latest['gamma_relative'] * 100:.2f}%")


    # -----------------------------
    # Session deviation summary
    # -----------------------------
    st.markdown('<div class="section">Session Deviation Summary</div>', unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)

    s1.metric(
        "Session Status",
        summary["session_anomaly_status"],
    )
    s2.metric(
        "Persistent Deviation",
        f"{summary['persistent_deviation_percent']:.2f}%",
    )
    s3.metric(
        "Persistent Episodes",
        summary["persistent_episodes"],
    )
    s4.metric(
        "Peak Deviation",
        f"{summary['peak_anomaly_score']:.3f}",
    )

    st.caption(
        "Deviation represents difference from the user's calibrated NORMAL EEG baseline. "
        "It is not a medical abnormality or diagnosis."
    )


    # -----------------------------
    # Charts
    # -----------------------------
    st.markdown('<div class="section">EEG Visualizations</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "EEG Bands",
            "Brain-State Score",
            "Indicators",
            "EEG Deviation",
        ]
    )

    x = result["start_time"]

    with tab1:
        fig, ax = plt.subplots(figsize=(12, 4.5))
        for band in ["delta", "theta", "alpha", "beta", "gamma"]:
            ax.plot(
                x,
                result[f"{band}_relative"],
                label=band.capitalize(),
            )
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Relative Power")
        ax.set_title("NeuroSync EEG Band Activity")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with tab2:
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.plot(
            result["start_time"],
            result["brain_state_score"],
            marker="o",
        )
        ax.axhline(50, linestyle="--")
        ax.set_ylim(0, 100)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Brain-state score")
        ax.set_title("NeuroSync Personalized Brain-State Score")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with tab3:
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.plot(x, result["cognitive_load_score"], label="Cognitive Load")
        ax.plot(x, result["relaxation_score"], label="Relaxation")
        ax.plot(x, result["activation_score"], label="Activation")
        ax.plot(x, result["fatigue_indicator"], label="Fatigue-related")
        ax.plot(x, result["eeg_stability_score"], label="EEG Stability")
        ax.set_ylim(0, 100)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Score")
        ax.set_title("NeuroSync EEG-Derived Indicators")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with tab4:
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.plot(
            x,
            result["anomaly_score"],
            label="Personalized Deviation Score",
        )
        ax.axhline(
            0.70,
            linestyle="--",
            label="Deviation Threshold",
        )

        persistent = result["persistent_anomaly"].astype(bool)

        if persistent.any():
            ax.scatter(
                x[persistent],
                result.loc[persistent, "anomaly_score"],
                marker="x",
                label="Persistent Deviation",
            )

        ax.set_ylim(0, 1)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Deviation score")
        ax.set_title("NeuroSync Personalized EEG Deviation")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)



    # -----------------------------
    # Gemini AI Interpretation
    # -----------------------------
    st.markdown('<div class="section">🤖 Gemini AI Session Interpretation</div>', unsafe_allow_html=True)
    st.caption(
        "Gemini receives a compact summary of the NeuroSync result dataframe. "
        "The supplied safety rules are retained: personalized deviation is not a diagnosis, "
        "and NeuroSync indicators are not clinically validated measurements."
    )

    if gemini_api_key:
        if st.button("✨ Generate Patient + Doctor Reports", use_container_width=True):
            with st.spinner("Gemini is interpreting the NeuroSync session..."):
                try:
                    report, session_summary = generate_neurosync_gemini_report(
                        result,
                        gemini_api_key,
                    )
                    st.session_state.gemini_report = report
                    st.session_state.gemini_session_summary = session_summary
                    st.success("AI interpretation generated successfully.")
                except Exception as exc:
                    st.error(f"Gemini report generation failed: {exc}")
    else:
        st.info("Enter a Gemini API key in the sidebar to generate the AI interpretation.")

    if st.session_state.gemini_report:
        report = st.session_state.gemini_report
        patient_response = report.get("patient_response", "")
        doctor_response = report.get("doctor_response", "")

        report_tab1, report_tab2 = st.tabs(["🧑 Patient Report", "🩺 Doctor / Rehab Report"])

        with report_tab1:
            st.markdown(patient_response)

        with report_tab2:
            st.markdown(doctor_response)

        report_payload = {
            "session_summary": st.session_state.gemini_session_summary,
            "ai_report": report,
        }
        json_report = json.dumps(report_payload, indent=2, ensure_ascii=False).encode("utf-8")
        txt_report = (
            "=" * 70 + "\n"
            "NEUROSYNC AI INTERPRETATION\n"
            + "=" * 70 + "\n\n"
            "PATIENT RESPONSE\n"
            + "-" * 70 + "\n"
            + patient_response
            + "\n\nDOCTOR RESPONSE\n"
            + "-" * 70 + "\n"
            + doctor_response
            + "\n\nNOTE:\n"
            "NeuroSync-derived EEG indicators and personalized deviation scores "
            "are not medical diagnoses.\n"
        ).encode("utf-8")

        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "⬇️ Download AI Report JSON",
                data=json_report,
                file_name="neurosync_gemini_report.json",
                mime="application/json",
                use_container_width=True,
            )
        with d2:
            st.download_button(
                "⬇️ Download AI Report TXT",
                data=txt_report,
                file_name="neurosync_gemini_report.txt",
                mime="text/plain",
                use_container_width=True,
            )


    # -----------------------------
    # Latest windows
    # -----------------------------
    st.markdown('<div class="section">Latest EEG Windows</div>', unsafe_allow_html=True)

    display_columns = [
        "window_id",
        "start_time",
        "end_time",
        "predicted_state",
        "predicted_polarity",
        "state_confidence",
        "brain_state_score",
        "cognitive_load_score",
        "relaxation_score",
        "activation_score",
        "fatigue_indicator",
        "eeg_stability_score",
        "anomaly_score",
        "anomaly_prediction",
        "persistent_anomaly",
        "deviation_status",
    ]

    table = result[display_columns].tail(10).copy()

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
    )


    # -----------------------------
    # Downloads
    # -----------------------------
    st.markdown('<div class="section">Export</div>', unsafe_allow_html=True)

    csv_bytes = result.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download NeuroSync Results CSV",
        data=csv_bytes,
        file_name="neurosync_personal_brain_state_results.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.download_button(
        "⬇️ Download Personal Calibration CSV",
        data=st.session_state.calibration_df.to_csv(index=False).encode("utf-8"),
        file_name="neurosync_personal_calibration.csv",
        mime="text/csv",
        use_container_width=True,
    )

else:
    st.info(
        "Upload the four calibration recordings, train the personal model, "
        "then upload a new unlabeled EEG recording to see the NeuroSync results."
    )

st.divider()
st.caption(
    "NeuroSync indicators are EEG-derived proxies based on the supplied pipeline. "
    "They are not clinical measurements or a diagnostic system."
)
