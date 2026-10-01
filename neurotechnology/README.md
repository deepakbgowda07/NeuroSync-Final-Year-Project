# NeuroSync Streamlit Dashboard

Simple Streamlit interface for the existing NeuroSync EEG classification pipeline.

## Files

- `app.py` — Streamlit dashboard
- `neurosync_core.py` — reusable NeuroSync EEG pipeline
- `requirements.txt` — Python dependencies

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Workflow

1. Upload HAPPY EEG CSV.
2. Upload NORMAL EEG CSV.
3. Upload SAD EEG CSV.
4. Upload ANGRY EEG CSV.
5. Click **Build Personal Calibration & Train Model**.
6. Upload a new unlabeled EEG CSV.
7. Click **Analyze EEG**.
8. View the brain state, indicators, EEG bands, deviation analysis, charts, latest windows, and download the result CSV.

Expected EEG CSV format:

```text
time,ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8
```

The dashboard uses the same NeuroSync processing/model logic from the supplied Python file: EEG preprocessing, frequency-band features, personal calibration, RBF SVM classification, Isolation Forest deviation detection, persistence filtering, indicators, and brain-state scoring.

The generated result CSV is:

`neurosync_personal_brain_state_results.csv`


## Gemini AI Session Interpretation

The dashboard also integrates the supplied `GeminiReportGen(1).py` interpretation layer.

1. Add a Gemini API key in the Streamlit sidebar (or set `GEMINI_API_KEY` as an environment variable).
2. Analyze a new EEG recording.
3. Click **Generate Patient + Doctor Reports**.
4. The dashboard displays:
   - **Patient Report** — simple, friendly session summary.
   - **Doctor / Rehab Report** — structured technical interpretation.
5. Download the generated report as JSON or TXT.

The Gemini layer uses the NeuroSync result dataframe and the supplied prompt/safety rules. It does not alter the EEG classification or anomaly-detection pipeline.
