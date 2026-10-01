"""NeuroSync Gemini AI interpretation layer.

Adapted from the supplied GeminiReportGen(1).py for use inside Streamlit.
The NeuroSync EEG result dataframe is summarized and sent to Gemini to
produce patient-friendly and doctor/rehabilitation-professional reports.
"""

import json
import os
import pandas as pd
import numpy as np

GEMINI_MODEL = "gemini-2.5-flash"

NEUROSYNC_GEMINI_PROMPT = """

You are the AI interpretation layer of NeuroSync, a personalized
EEG brain-state analysis system.

You will receive a structured NeuroSync session summary containing
EEG features, brain-state classification, personalized baseline
deviation analysis, and NeuroSync-derived functional indicators.

Your job is NOT to repeat every numerical value.

Your job is to convert the data into a clear, useful and well-organized
report for two audiences:

1. PATIENT
2. DOCTOR / REHABILITATION PROFESSIONAL

============================================================
IMPORTANT SAFETY AND INTERPRETATION RULES
============================================================

1. Never diagnose neurological disease, psychiatric disease,
   brain injury, or any other medical condition from this data.

2. "Anomaly", "deviation", and "abnormal" refer ONLY to deviation
   from the person's personalized NORMAL EEG baseline.

3. Isolation Forest is a personalized EEG deviation detector.
   It is NOT a medical abnormality detector.

4. NeuroSync scores such as:
   - cognitive load
   - relaxation
   - activation
   - fatigue
   - EEG stability
   - brain-state score

   are NeuroSync-derived indicators and must not be described
   as clinically validated measurements.

5. "Happy", "Normal", "Sad", and "Angry" are NeuroSync
   classification labels. They are not definitive diagnoses
   of emotional state.

6. Never invent information.

7. Never change numerical measurements.

8. Do not blindly list mean, minimum and maximum values for every
   parameter.

9. Only mention numerical values when they actually help explain
   an important finding.

10. Focus on PATTERNS, CHANGES, STRENGTHS, CONCERNS and ACTIONABLE
    OBSERVATIONS.

11. If a finding occurs only briefly, describe it as transient.

12. If a finding persists across the session, describe it as
    sustained.

13. Do not call a patient "abnormal".

14. Do not claim that the patient is definitely doing something
    wrong.

15. If potentially harmful or undesirable patterns are present,
    describe them as "things to monitor" or "possible concerns".

16. Do not provide a guaranteed recovery time.

17. If recovery is discussed, explain that recovery depends on
    the underlying condition, rehabilitation program, consistency,
    clinical findings and individual response.

18. Do not recommend medication changes.

19. Do not create unnecessary fear.

============================================================
PATIENT RESPONSE
============================================================

Create a patient-friendly report.

Use short sections and bullet points.

Do NOT overwhelm the patient with technical EEG terminology.

Use this structure:

# 🧠 NEUROSYNC SESSION SUMMARY

## Overall Result
Explain in 2–4 simple sentences:
- dominant detected state
- whether EEG stayed close to personal baseline
- whether any important temporary change occurred

## ✅ What You're Doing Well
Give 3–5 concise bullet points based ONLY on the data.

Examples:
- Stable EEG pattern
- No persistent deviation
- Good overall stability
- Consistent brain-state pattern

Only include points actually supported by the data.

## ⚠️ Things to Pay Attention To
Give 2–5 concise bullet points.

Focus on:
- high cognitive load
- low relaxation
- high activation
- fatigue-related elevation
- temporary EEG changes
- repeated deviation

Do not exaggerate.

## 🔎 Important Moment
If there is a clearly important time period, explain:

"When: X–Y seconds"

Then explain in simple language what changed.

If there is no important period, say:
"No major temporary change was identified."

## 👍 What You Can Do
Give 3–5 practical, non-medical suggestions based on
the detected pattern.

Examples:
- take appropriate rest breaks
- maintain consistent sleep/rest
- reduce prolonged mental overload
- follow the rehabilitation plan
- discuss recurring patterns with the clinician

Do NOT prescribe medical treatment.

## 🚨 When to Discuss With Your Clinician
Only recommend discussion when appropriate.

Explain what recurring or concerning patterns would justify
bringing the result to the clinician.

## Important Note
Clearly state:

"NeuroSync's emotional-state labels are system classifications,
not medical diagnoses. EEG deviation means deviation from your
personal baseline and does not by itself indicate disease."

============================================================
DOCTOR / REHABILITATION PROFESSIONAL RESPONSE
============================================================

Create a concise technical report.

IMPORTANT:

Do NOT dump every mean/min/max value.

Do NOT produce dozens of individual statistics.

Instead, summarize the clinically/research-relevant patterns.

Use this exact structure:

# 🩺 NEUROSYNC TECHNICAL SUMMARY

## 1. Session Overview

Use a compact table:

| Parameter | Finding |
|---|---|
| Session duration | |
| Total windows | |
| Dominant state | |
| Classification consistency | |
| Average confidence | |
| Personalized deviation status | |
| Persistent deviation | |
| EEG stability | |

Only include relevant values.

## 2. Functional Indicators

Use a table:

| Indicator | Overall level | Important observation |
|---|---|---|
| Cognitive load | | |
| Relaxation | | |
| Activation | | |
| Fatigue | | |
| EEG stability | | |
| Brain-state score | | |

Use the NeuroSync level classifications where available.

Do not unnecessarily provide min/max values.

## 3. EEG Pattern

Summarize the important EEG-band behavior.

Discuss:
- Delta
- Theta
- Alpha
- Beta
- Gamma

Focus on relationships and changes rather than dumping numbers.

Example:

"Beta activity was relatively prominent while alpha remained
generally stable, with a temporary reduction during the
14–20 second interval."

Only provide numerical values if they meaningfully support
the interpretation.

## 4. Spectral / Hjorth Findings

Summarize only meaningful changes in:
- spectral entropy
- spectral centroid
- spectral bandwidth
- theta/beta ratio
- alpha/beta ratio
- alpha dominance
- Hjorth activity
- Hjorth mobility
- Hjorth complexity

Do NOT list mean/min/max for every parameter.

Explain what the combined pattern suggests.

## 5. State Classification

Use a compact table:

| State | Probability |
|---|---:|
| Angry | |
| Happy | |
| Normal | |
| Sad | |

Then provide a 1–3 sentence interpretation.

Make clear that these are NeuroSync classifications.

## 6. Personalized EEG Deviation

Use a compact table:

| Parameter | Finding |
|---|---|
| Mean deviation | |
| Peak deviation | |
| Candidate deviation windows | |
| Persistent deviation | |
| Persistent deviation % | |
| Deviation episodes | |
| Session status | |

Then explain whether the deviation appears:
- stable
- transient
- repeated
- persistent

IMPORTANT:
Do not describe personalized deviation as pathology.

## 7. Important Temporal Pattern

Identify the SINGLE most important time interval if one exists.

Format:

### ⏱ X–Y seconds

- Main EEG change
- Main functional-indicator change
- Stability change
- Deviation change
- Overall interpretation

If no meaningful temporal change exists, say so.

## 8. Positive Findings

List 3–5 important positive findings.

Examples:
- High EEG stability
- No persistent personalized deviation
- Stable state classification
- Recovery toward baseline after transient change

Only include supported findings.

## 9. Points to Monitor

List 3–5 things that should be monitored across future sessions.

Examples:
- sustained high cognitive load
- repeated low relaxation
- repeated activation spikes
- repeated fatigue elevation
- recurring personalized EEG deviations

Do NOT call these diseases or abnormalities.

## 10. Overall Technical Interpretation

Give a concise 1-paragraph synthesis of the complete session.

The paragraph should answer:

"What is the most important thing the doctor should take away
from this session?"

## 11. Limitations

Mention briefly:

- NeuroSync-derived indicators are not standalone clinical
  measurements.
- Emotional-state classification is not definitive.
- Personalized deviation is not a diagnosis.
- Results should be correlated with clinical history,
  examination, rehabilitation context and other assessments.

============================================================
STYLE REQUIREMENTS
============================================================

PATIENT:

- Simple English
- Short sentences
- Bullet points
- Friendly
- Reassuring but honest
- No unnecessary technical terminology
- No huge lists of numbers

DOCTOR:

- Professional
- Concise
- Structured
- Tables wherever useful
- Bullet points for observations
- Technical enough to be useful
- Avoid unnecessary statistics
- Focus on clinically/research-relevant patterns

MOST IMPORTANT:

Do NOT simply repeat the supplied session summary.

Interpret it and organize it.

The output should look like a professionally prepared
NeuroSync report rather than raw machine output.

============================================================
OUTPUT FORMAT
============================================================

Return valid JSON:

{
  "patient_response": "...",
  "doctor_response": "..."
}

Do not add any other top-level fields.

============================================================
NEUROSYNC SESSION DATA
============================================================

"""


# ============================================================
# 4. HELPER FUNCTION
# ============================================================


def safe_float(value, default=None):
    """
    Convert values to float safely.
    """
    try:
        if pd.isna(value):
            return default
        return float(value)
    except:
        return default


def safe_round(value, digits=3):
    """
    Safely round numerical values.
    """
    value = safe_float(value)

    if value is None:
        return None

    return round(value, digits)


# ============================================================
# 5. BUILD SESSION SUMMARY FROM NEUROSYNC CSV
# ============================================================

def build_neurosync_session_summary(results_df):
    """
    Converts the complete NeuroSync result dataframe into a
    compact session summary for Gemini.

    The original CSV remains untouched.
    """

    df = results_df.copy()

    summary = {}

    # --------------------------------------------------------
    # BASIC SESSION INFORMATION
    # --------------------------------------------------------

    summary["total_windows"] = int(len(df))

    if "start_time" in df.columns and "end_time" in df.columns:

        start = safe_float(df["start_time"].min())
        end = safe_float(df["end_time"].max())

        if start is not None and end is not None:
            summary["session_duration_seconds"] = round(
                end - start, 2
            )

    # --------------------------------------------------------
    # STATE CLASSIFICATION
    # --------------------------------------------------------

    if "predicted_state" in df.columns:

        state_counts = (
            df["predicted_state"]
            .value_counts()
            .to_dict()
        )

        summary["state_distribution"] = {
            str(k): int(v)
            for k, v in state_counts.items()
        }

        if len(df) > 0:
            summary["dominant_state"] = str(
                df["predicted_state"].mode().iloc[0]
            )

    if "state_confidence" in df.columns:

        summary["average_state_confidence"] = safe_round(
            df["state_confidence"].mean(),
            4
        )

        summary["minimum_state_confidence"] = safe_round(
            df["state_confidence"].min(),
            4
        )

        summary["maximum_state_confidence"] = safe_round(
            df["state_confidence"].max(),
            4
        )

    # --------------------------------------------------------
    # POLARITY
    # --------------------------------------------------------

    if "predicted_polarity" in df.columns:

        polarity_counts = (
            df["predicted_polarity"]
            .value_counts()
            .to_dict()
        )

        summary["polarity_distribution"] = {
            str(k): int(v)
            for k, v in polarity_counts.items()
        }

    # --------------------------------------------------------
    # EEG BAND FEATURES
    # --------------------------------------------------------

    band_features = [
        "delta_power",
        "delta_relative",
        "theta_power",
        "theta_relative",
        "alpha_power",
        "alpha_relative",
        "beta_power",
        "beta_relative",
        "gamma_power",
        "gamma_relative"
    ]

    summary["eeg_band_statistics"] = {}

    for feature in band_features:

        if feature in df.columns:

            summary["eeg_band_statistics"][feature] = {
                "mean": safe_round(df[feature].mean(), 6),
                "min": safe_round(df[feature].min(), 6),
                "max": safe_round(df[feature].max(), 6)
            }

    # --------------------------------------------------------
    # SPECTRAL FEATURES
    # --------------------------------------------------------

    spectral_features = [
        "spectral_entropy",
        "spectral_centroid",
        "spectral_bandwidth",
        "theta_beta_ratio",
        "alpha_beta_ratio",
        "slow_wave_index",
        "fast_wave_index",
        "alpha_dominance"
    ]

    summary["spectral_statistics"] = {}

    for feature in spectral_features:

        if feature in df.columns:

            summary["spectral_statistics"][feature] = {
                "mean": safe_round(df[feature].mean(), 6),
                "min": safe_round(df[feature].min(), 6),
                "max": safe_round(df[feature].max(), 6)
            }

    # --------------------------------------------------------
    # HJORTH PARAMETERS
    # --------------------------------------------------------

    hjorth_features = [
        "hjorth_activity",
        "hjorth_mobility",
        "hjorth_complexity"
    ]

    summary["hjorth_statistics"] = {}

    for feature in hjorth_features:

        if feature in df.columns:

            summary["hjorth_statistics"][feature] = {
                "mean": safe_round(df[feature].mean(), 6),
                "min": safe_round(df[feature].min(), 6),
                "max": safe_round(df[feature].max(), 6)
            }

    # --------------------------------------------------------
    # SIGNAL FEATURES
    # --------------------------------------------------------

    signal_features = [
        "signal_std",
        "signal_rms",
        "signal_ptp"
    ]

    summary["signal_statistics"] = {}

    for feature in signal_features:

        if feature in df.columns:

            summary["signal_statistics"][feature] = {
                "mean": safe_round(df[feature].mean(), 6),
                "min": safe_round(df[feature].min(), 6),
                "max": safe_round(df[feature].max(), 6)
            }

    # --------------------------------------------------------
    # NEUROSYNC INDICATORS
    # --------------------------------------------------------

    indicator_features = [
        "cognitive_load_score",
        "relaxation_score",
        "activation_score",
        "fatigue_indicator",
        "eeg_stability_score",
        "brain_state_score"
    ]

    summary["neurosync_indicators"] = {}

    for feature in indicator_features:

        if feature in df.columns:

            summary["neurosync_indicators"][feature] = {
                "mean": safe_round(df[feature].mean(), 3),
                "min": safe_round(df[feature].min(), 3),
                "max": safe_round(df[feature].max(), 3)
            }

    # --------------------------------------------------------
    # INDICATOR LEVELS
    # --------------------------------------------------------

    level_features = [
        "cognitive_load_level",
        "relaxation_level",
        "activation_level",
        "fatigue_level",
        "stability_level"
    ]

    summary["neurosync_indicator_levels"] = {}

    for feature in level_features:

        if feature in df.columns:

            counts = (
                df[feature]
                .value_counts()
                .to_dict()
            )

            summary["neurosync_indicator_levels"][feature] = {
                str(k): int(v)
                for k, v in counts.items()
            }

    # --------------------------------------------------------
    # ANOMALY / DEVIATION ANALYSIS
    # --------------------------------------------------------

    anomaly_columns = [
        "anomaly_raw_score",
        "anomaly_score"
    ]

    summary["deviation_statistics"] = {}

    for feature in anomaly_columns:

        if feature in df.columns:

            summary["deviation_statistics"][feature] = {
                "mean": safe_round(df[feature].mean(), 4),
                "min": safe_round(df[feature].min(), 4),
                "max": safe_round(df[feature].max(), 4)
            }

    if "anomaly_prediction" in df.columns:

        counts = (
            df["anomaly_prediction"]
            .value_counts()
            .to_dict()
        )

        summary["deviation_statistics"]["prediction_distribution"] = {
            str(k): int(v)
            for k, v in counts.items()
        }

    if "persistent_anomaly" in df.columns:

        persistent = (
            df["persistent_anomaly"]
            .astype(str)
            .str.upper()
            .isin(["TRUE", "1", "YES"])
        )

        summary["deviation_statistics"][
            "persistent_deviation_windows"
        ] = int(persistent.sum())

        summary["deviation_statistics"][
            "persistent_deviation_percentage"
        ] = round(
            persistent.mean() * 100,
            2
        )

    if "deviation_status" in df.columns:

        counts = (
            df["deviation_status"]
            .value_counts()
            .to_dict()
        )

        summary["deviation_statistics"]["status_distribution"] = {
            str(k): int(v)
            for k, v in counts.items()
        }

    # --------------------------------------------------------
    # PROBABILITY FEATURES
    # --------------------------------------------------------

    probability_features = [
        "prob_angry",
        "prob_happy",
        "prob_normal",
        "prob_sad"
    ]

    summary["state_probability_statistics"] = {}

    for feature in probability_features:

        if feature in df.columns:

            summary["state_probability_statistics"][feature] = {
                "mean": safe_round(df[feature].mean(), 4),
                "min": safe_round(df[feature].min(), 4),
                "max": safe_round(df[feature].max(), 4)
            }

    # --------------------------------------------------------
    # IMPORTANT WINDOWS
    # --------------------------------------------------------
    #
    # Instead of sending thousands of windows to Gemini,
    # select the most informative ones.

    important_windows = pd.DataFrame()

    if "anomaly_score" in df.columns:

        important_windows = pd.concat([
            important_windows,
            df.nlargest(
                min(10, len(df)),
                "anomaly_score"
            )
        ])

    if "state_confidence" in df.columns:

        important_windows = pd.concat([
            important_windows,
            df.nsmallest(
                min(10, len(df)),
                "state_confidence"
            )
        ])

    if "persistent_anomaly" in df.columns:

        persistent_mask = (
            df["persistent_anomaly"]
            .astype(str)
            .str.upper()
            .isin(["TRUE", "1", "YES"])
        )

        important_windows = pd.concat([
            important_windows,
            df[persistent_mask]
        ])

    # Remove duplicate windows

    if "window_id" in important_windows.columns:

        important_windows = (
            important_windows
            .drop_duplicates(subset=["window_id"])
            .sort_values("window_id")
        )

    # --------------------------------------------------------
    # SELECT USEFUL COLUMNS FOR GEMINI
    # --------------------------------------------------------

    important_columns = [
        "window_id",
        "start_time",
        "end_time",

        "predicted_state",
        "state_confidence",

        "prob_angry",
        "prob_happy",
        "prob_normal",
        "prob_sad",

        "delta_relative",
        "theta_relative",
        "alpha_relative",
        "beta_relative",
        "gamma_relative",

        "theta_beta_ratio",
        "alpha_beta_ratio",

        "spectral_entropy",
        "spectral_centroid",
        "spectral_bandwidth",

        "hjorth_activity",
        "hjorth_mobility",
        "hjorth_complexity",

        "alpha_dominance",
        "activation_ratio",
        "relaxation_ratio",

        "cognitive_load_score",
        "relaxation_score",
        "activation_score",
        "fatigue_indicator",
        "eeg_stability_score",
        "brain_state_score",

        "anomaly_score",
        "anomaly_prediction",
        "persistent_anomaly",
        "deviation_status"
    ]

    available_columns = [
        c for c in important_columns
        if c in important_windows.columns
    ]

    if len(important_windows) > 0:

        important_window_records = (
            important_windows[available_columns]
            .replace({np.nan: None})
            .to_dict(orient="records")
        )

    else:

        important_window_records = []

    summary["important_windows"] = important_window_records

    return summary


# ============================================================
# 6. SEND SESSION SUMMARY TO GEMINI
# ============================================================


def generate_neurosync_gemini_report(results_df, api_key, model_name=GEMINI_MODEL):

    print("\n")
    print("=" * 70)
    print("        SENDING NEUROSYNC SESSION TO GEMINI")
    print("=" * 70)

    # Build compact session summary
    session_summary = build_neurosync_session_summary(
        results_df
    )

    # Convert to JSON
    session_json = json.dumps(
        session_summary,
        indent=2,
        allow_nan=False
    )

    # Final prompt
    final_prompt = (
        NEUROSYNC_GEMINI_PROMPT
        + "\n\n"
        + "============================================================\n"
        + "NEUROSYNC SESSION DATA\n"
        + "============================================================\n\n"
        + session_json
        + "\n\n"
        + "Analyze ONLY the supplied NeuroSync session data."
    )

    print("Session windows:", session_summary["total_windows"])
    print("Sending data to Gemini...")
    
    # --------------------------------------------------------
    # GEMINI REQUEST
    # --------------------------------------------------------

    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(

        model=model_name,

        contents=final_prompt,

        config=types.GenerateContentConfig(

            temperature=0.2,

            response_mime_type="application/json"
        )
    )

    # --------------------------------------------------------
    # READ RESPONSE
    # --------------------------------------------------------

    response_text = response.text

    try:

        report = json.loads(response_text)

    except json.JSONDecodeError:

        print(
            "\nWARNING: Gemini did not return valid JSON."
        )

        report = {
            "patient_response": response_text,
            "doctor_response":
                "Gemini returned a non-JSON response. "
                "Review the raw response above."
        }

    return report, session_summary


# ============================================================
# 7. PRINT PATIENT + DOCTOR RESPONSES
# ============================================================

