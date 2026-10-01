# NEUROSYNC
# PERSONALIZED EEG BRAIN-STATE ANALYSIS
# ================================================================
#
# PURPOSE
# -------
# This version focuses ONLY on:
#
#   1. EEG preprocessing
#   2. Delta / Theta / Alpha / Beta / Gamma extraction
#   3. Derived EEG features
#   4. Personal calibration
#   5. Personalized brain-state classification
#   6. Positive / Neutral / Negative state
#   7. Cognitive-load / relaxation / activation /
#      fatigue-related / stability indicators
#   8. Personalized EEG deviation / anomaly detection
#      using Isolation Forest
#
# ANOMALY DETECTION:
#   - Trained on the user's NORMAL calibration windows
#   - Runs on every analysis window
#   - Uses persistence filtering
#   - Produces window-level and session-level deviation status
#
# IMPORTANT:
#   ANOMALY means deviation from the user's calibrated NORMAL EEG.
#   It does not mean disease, medical abnormality, or diagnosis.
#
# INPUT FORMAT
# ------------
#
# time,ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8
# 0.0,...
# 0.00390625,...
# 0.0078125,...
#
# Expected sampling rate for your current data:
# 256 Hz
#
# WINDOW:
#   4 seconds
#
# STEP:
#   2 seconds
#
# CALIBRATION STATES:
#   happy
#   normal
#   sad
#   angry
#
# The calibration recordings must contain enough data to produce
# multiple 4-second windows.
#
# Recommended:
#   At least 2 minutes per state
#   Better: 3-5 minutes per state
#
# ================================================================


# ================================================================
# 1. INSTALL / IMPORT
# ================================================================



import io
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, welch

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")


# ================================================================
# 2. CONFIGURATION
# ================================================================

DEFAULT_FS = 256.0

LOWCUT = 0.5
HIGHCUT = 45.0
NOTCH_FREQ = 50.0

WINDOW_SECONDS = 4.0
STEP_SECONDS = 2.0

MIN_CALIBRATION_WINDOWS = 10

RANDOM_STATE = 42

# Personalized EEG deviation / anomaly detection
ANOMALY_CONTAMINATION = 0.10
ANOMALY_N_ESTIMATORS = 200
ANOMALY_SCORE_THRESHOLD = 0.70
ANOMALY_MIN_CONSECUTIVE_WINDOWS = 3

# Session-level prototype thresholds (NOT clinical cutoffs)
SESSION_MILD_DEVIATION_PERCENT = 5.0
SESSION_SIGNIFICANT_DEVIATION_PERCENT = 15.0


BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


# ================================================================
# 3. BASIC INPUT FUNCTIONS
# ================================================================

def detect_timestamp(df):
    """
    Detect the timestamp column.
    """

    preferred = ["time", "timestamp", "ts", "t"]

    lower_map = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for name in preferred:
        if name in lower_map:
            return lower_map[name]

    # Fallback:
    # look for a monotonically increasing numeric column
    for col in df.columns:

        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        x = pd.to_numeric(
            df[col],
            errors="coerce"
        ).dropna().values

        if len(x) < 4:
            continue

        d = np.diff(x)

        if len(d) == 0:
            continue

        if np.all(d > 0):
            return col

    return None


def detect_channels(df, timestamp_col):
    """
    Detect numeric EEG channels.
    """

    channels = []

    for col in df.columns:

        if col == timestamp_col:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            channels.append(col)

    if len(channels) == 0:
        raise ValueError(
            "No numeric EEG channels were detected."
        )

    return channels


def infer_sampling_rate(df, timestamp_col):
    """
    Infer sampling rate from timestamp differences.
    """

    if timestamp_col is None:
        return DEFAULT_FS

    t = pd.to_numeric(
        df[timestamp_col],
        errors="coerce"
    ).dropna().values

    if len(t) < 3:
        return DEFAULT_FS

    differences = np.diff(t)

    differences = differences[
        np.isfinite(differences) &
        (differences > 0)
    ]

    if len(differences) == 0:
        return DEFAULT_FS

    dt = np.median(differences)

    if dt <= 0:
        return DEFAULT_FS

    fs = 1.0 / dt

    return float(fs)


# ================================================================
# 4. LOAD EEG CSV
# ================================================================

def load_eeg_csv(file):
    """
    Load an EEG CSV and automatically detect:
      - timestamp
      - EEG channels
      - sampling rate
    """

    if isinstance(file, (str, Path)):
        df = pd.read_csv(file)
    else:
        df = pd.read_csv(file)

    timestamp_col = detect_timestamp(df)

    channels = detect_channels(
        df,
        timestamp_col
    )

    fs = infer_sampling_rate(
        df,
        timestamp_col
    )

    # Convert EEG channels to numeric
    for ch in channels:
        df[ch] = pd.to_numeric(
            df[ch],
            errors="coerce"
        )

    # Repair missing values
    missing_before = int(
        df[channels].isna().sum().sum()
    )

    df[channels] = (
        df[channels]
        .interpolate(
            method="linear",
            limit_direction="both"
        )
        .fillna(
            df[channels].median()
        )
    )

    missing_after = int(
        df[channels].isna().sum().sum()
    )

    print("\n================ INPUT =================")
    print(f"Rows              : {len(df):,}")
    print(f"Columns           : {len(df.columns)}")
    print(
        f"Timestamp column  : "
        f"{timestamp_col if timestamp_col else 'None'}"
    )
    print(
        f"EEG channels      : {len(channels)}"
    )
    print(
        f"Channels          : {', '.join(map(str, channels))}"
    )
    print(
        f"Sampling rate     : {fs:.2f} Hz"
    )
    print(
        f"Duration          : "
        f"{len(df) / fs:.2f} seconds"
    )
    print(
        f"Missing repaired  : "
        f"{missing_before - missing_after}"
    )
    print("========================================\n")

    return df, channels, fs


# ================================================================
# 5. EEG PREPROCESSING
# ================================================================

def bandpass_filter(data, fs):
    """
    0.5-45 Hz Butterworth band-pass.
    """

    nyquist = fs / 2.0

    high = min(
        HIGHCUT,
        nyquist - 1.0
    )

    if high <= LOWCUT:
        raise ValueError(
            "Sampling rate is too low for the configured "
            "EEG band-pass filter."
        )

    sos = butter(
        4,
        [LOWCUT / nyquist, high / nyquist],
        btype="bandpass",
        output="sos"
    )

    return sosfiltfilt(
        sos,
        data,
        axis=0
    )


def notch_filter(data, fs):
    """
    50 Hz notch filter.

    If the sampling rate does not permit a 50 Hz notch,
    the data is returned unchanged.
    """

    nyquist = fs / 2.0

    if NOTCH_FREQ >= nyquist:
        return data

    b, a = iirnotch(
        NOTCH_FREQ,
        Q=30,
        fs=fs
    )

    return filtfilt(
        b,
        a,
        data,
        axis=0
    )


def preprocess_eeg(df, channels, fs):
    """
    Full EEG preprocessing.
    """

    raw = df[channels].values.astype(float)

    # Remove channel DC offset
    raw = raw - np.mean(
        raw,
        axis=0,
        keepdims=True
    )

    # Band-pass
    filtered = bandpass_filter(
        raw,
        fs
    )

    # Notch
    filtered = notch_filter(
        filtered,
        fs
    )

    return filtered


# ================================================================
# 6. WINDOW CREATION
# ================================================================

def create_windows(
    filtered,
    fs,
    window_seconds=WINDOW_SECONDS,
    step_seconds=STEP_SECONDS
):
    """
    Create overlapping EEG windows.

    Example:
        Window 0 = 0-4 sec
        Window 1 = 2-6 sec
        Window 2 = 4-8 sec
        ...
    """

    window_samples = int(
        round(window_seconds * fs)
    )

    step_samples = int(
        round(step_seconds * fs)
    )

    windows = []

    start = 0
    window_id = 0

    while (
        start + window_samples
        <= len(filtered)
    ):

        end = start + window_samples

        segment = filtered[
            start:end
        ]

        windows.append({
            "id": window_id,
            "start_sample": start,
            "end_sample": end,
            "start_time": start / fs,
            "end_time": end / fs,
            "data": segment
        })

        start += step_samples
        window_id += 1

    return windows


# ================================================================
# 7. HELPER FUNCTIONS FOR FEATURES
# ================================================================

def safe_mean(x):
    return float(np.mean(x))


def safe_std(x):
    return float(np.std(x))


def safe_rms(x):
    return float(
        np.sqrt(
            np.mean(
                np.square(x)
            )
        )
    )


def safe_ptp(x):
    return float(
        np.ptp(x)
    )


def hjorth_parameters(x):
    """
    Hjorth:
      Activity
      Mobility
      Complexity
    """

    x = np.asarray(x)

    activity = np.var(x)

    dx = np.diff(x)

    if len(dx) < 2:
        return (
            float(activity),
            0.0,
            0.0
        )

    var_dx = np.var(dx)

    if activity > 0:
        mobility = np.sqrt(
            var_dx / activity
        )
    else:
        mobility = 0.0

    ddx = np.diff(dx)

    if len(ddx) > 1 and var_dx > 0:
        mobility_dx = np.sqrt(
            np.var(ddx) / var_dx
        )
    else:
        mobility_dx = 0.0

    if mobility > 0:
        complexity = (
            mobility_dx / mobility
        )
    else:
        complexity = 0.0

    return (
        float(activity),
        float(mobility),
        float(complexity)
    )


def _trapezoid(y, x):
    integrate = getattr(np, "trapezoid", None)

    if integrate is not None:
        return integrate(y, x)

    integrate = getattr(np, "trapz", None)

    if integrate is not None:
        return integrate(y, x)

    y = np.asarray(y)
    x = np.asarray(x)

    return np.sum(
        (x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2
    )


def spectral_features(x, fs):
    """
    Welch PSD-based features.
    """

    nperseg = min(
        256,
        len(x)
    )

    freqs, psd = welch(
        x,
        fs=fs,
        nperseg=nperseg
    )

    psd = np.maximum(
        psd,
        1e-12
    )

    total_power = _trapezoid(
        psd,
        freqs
    )

    if total_power <= 0:
        total_power = 1e-12

    probability = (
        psd /
        np.sum(psd)
    )

    entropy = -np.sum(
        probability *
        np.log2(probability)
    )

    centroid = (
        np.sum(
            freqs * psd
        ) /
        np.sum(psd)
    )

    bandwidth = np.sqrt(
        np.sum(
            ((freqs - centroid) ** 2)
            * psd
        ) /
        np.sum(psd)
    )

    return {
        "spectral_entropy": float(
            entropy
        ),
        "spectral_centroid": float(
            centroid
        ),
        "spectral_bandwidth": float(
            bandwidth
        ),
        "total_power": float(
            total_power
        ),
    }


def band_power(
    x,
    fs,
    low,
    high
):
    """
    Calculate absolute power inside
    a frequency band.
    """

    nperseg = min(
        256,
        len(x)
    )

    freqs, psd = welch(
        x,
        fs=fs,
        nperseg=nperseg
    )

    mask = (
        (freqs >= low) &
        (freqs < high)
    )

    if not np.any(mask):
        return 0.0

    power = _trapezoid(
        psd[mask],
        freqs[mask]
    )

    return float(
        max(power, 0.0)
    )


# ================================================================
# 8. EXTRACT EEG FEATURES
# ================================================================

def extract_window_features(
    window_data,
    fs,
    channels
):
    """
    Extract window-level EEG features.

    Features are first calculated per channel
    and then averaged across channels.
    """

    feature = {}

    # ------------------------------------------------------------
    # Per-channel features
    # ------------------------------------------------------------

    channel_band_powers = {
        band: []
        for band in BANDS
    }

    channel_band_relative = {
        band: []
        for band in BANDS
    }

    entropy_values = []
    centroid_values = []
    bandwidth_values = []

    hjorth_activity = []
    hjorth_mobility = []
    hjorth_complexity = []

    signal_std = []
    signal_rms = []
    signal_ptp = []

    for ch_index, ch_name in enumerate(channels):

        x = window_data[:, ch_index]

        # Time domain
        signal_std.append(
            safe_std(x)
        )

        signal_rms.append(
            safe_rms(x)
        )

        signal_ptp.append(
            safe_ptp(x)
        )

        # Hjorth
        activity, mobility, complexity = (
            hjorth_parameters(x)
        )

        hjorth_activity.append(
            activity
        )

        hjorth_mobility.append(
            mobility
        )

        hjorth_complexity.append(
            complexity
        )

        # Spectral
        spectral = spectral_features(
            x,
            fs
        )

        entropy_values.append(
            spectral["spectral_entropy"]
        )

        centroid_values.append(
            spectral["spectral_centroid"]
        )

        bandwidth_values.append(
            spectral["spectral_bandwidth"]
        )

        # Band powers
        powers = {}

        for band, (
            low,
            high
        ) in BANDS.items():

            p = band_power(
                x,
                fs,
                low,
                high
            )

            powers[band] = p

            channel_band_powers[
                band
            ].append(p)

        total = sum(
            powers.values()
        )

        if total <= 0:
            total = 1e-12

        for band in BANDS:

            channel_band_relative[
                band
            ].append(
                powers[band] /
                total
            )

    # ------------------------------------------------------------
    # Aggregate across channels
    # ------------------------------------------------------------

    for band in BANDS:

        feature[
            f"{band}_power"
        ] = float(
            np.mean(
                channel_band_powers[
                    band
                ]
            )
        )

        feature[
            f"{band}_relative"
        ] = float(
            np.mean(
                channel_band_relative[
                    band
                ]
            )
        )

    feature["signal_std"] = float(
        np.mean(signal_std)
    )

    feature["signal_rms"] = float(
        np.mean(signal_rms)
    )

    feature["signal_ptp"] = float(
        np.mean(signal_ptp)
    )

    feature["hjorth_activity"] = float(
        np.mean(hjorth_activity)
    )

    feature["hjorth_mobility"] = float(
        np.mean(hjorth_mobility)
    )

    feature["hjorth_complexity"] = float(
        np.mean(hjorth_complexity)
    )

    feature["spectral_entropy"] = float(
        np.mean(entropy_values)
    )

    feature["spectral_centroid"] = float(
        np.mean(centroid_values)
    )

    feature["spectral_bandwidth"] = float(
        np.mean(bandwidth_values)
    )

    # ------------------------------------------------------------
    # Derived EEG ratios
    # ------------------------------------------------------------

    theta = feature["theta_relative"]
    alpha = feature["alpha_relative"]
    beta = feature["beta_relative"]

    feature["theta_beta_ratio"] = (
        theta /
        max(beta, 1e-8)
    )

    feature["alpha_beta_ratio"] = (
        alpha /
        max(beta, 1e-8)
    )

    # ------------------------------------------------------------
    # Additional composite EEG features
    # ------------------------------------------------------------

    # Low-frequency activity
    feature["slow_wave_index"] = (
        feature["delta_relative"] +
        feature["theta_relative"]
    )

    # Fast activity
    feature["fast_wave_index"] = (
        feature["beta_relative"] +
        feature["gamma_relative"]
    )

    # Alpha dominance
    feature["alpha_dominance"] = (
        feature["alpha_relative"] /
        max(
            feature["fast_wave_index"],
            1e-8
        )
    )

    # Activation-related spectral ratio
    feature["activation_ratio"] = (
        feature["fast_wave_index"] /
        max(
            feature["slow_wave_index"],
            1e-8
        )
    )

    # Alpha relative to high-frequency activity
    feature["relaxation_ratio"] = (
        feature["alpha_relative"] /
        max(
            feature["beta_relative"] +
            feature["gamma_relative"],
            1e-8
        )
    )

    return feature


# ================================================================
# 9. PROCESS ONE EEG RECORDING
# ================================================================

def process_recording(
    file,
    label=None
):
    """
    Complete pipeline for one EEG recording.
    """

    df, channels, fs = load_eeg_csv(
        file
    )

    filtered = preprocess_eeg(
        df,
        channels,
        fs
    )

    windows = create_windows(
        filtered,
        fs
    )

    print(
        f"Generated windows : "
        f"{len(windows)}"
    )

    if len(windows) == 0:
        raise ValueError(
            "The recording is shorter than "
            "the 4-second analysis window."
        )

    records = []

    for w in windows:

        features = extract_window_features(
            w["data"],
            fs,
            channels
        )

        record = {
            "window_id": w["id"],
            "start_time": w["start_time"],
            "end_time": w["end_time"],
        }

        record.update(
            features
        )

        if label is not None:
            record["label"] = label

        records.append(
            record
        )

    return pd.DataFrame(records)


# ================================================================
# 10. PERSONAL BASELINE
# ================================================================

class PersonalBaseline:
    """
    Stores the user's baseline EEG feature profile.

    Instead of asking:
        "Is this normal for humans?"

    we ask:
        "How different is this from THIS user's baseline?"
    """

    def __init__(self):

        self.feature_columns = None
        self.mean = None
        self.std = None

    def fit(self, df, feature_columns):

        self.feature_columns = list(
            feature_columns
        )

        values = df[
            self.feature_columns
        ].astype(float)

        self.mean = values.mean()

        self.std = values.std()

        # Avoid division by zero
        self.std = self.std.replace(
            0,
            1e-8
        )

        return self

    def transform(self, df):

        values = df[
            self.feature_columns
        ].astype(float)

        return (
            values -
            self.mean
        ) / self.std

    def transform_dataframe(self, df):

        result = df.copy()

        z = self.transform(df)

        for col in self.feature_columns:

            result[
                f"{col}_personal_z"
            ] = z[col].values

        return result


# ================================================================
# 11. DEFINE FEATURE SET
# ================================================================

FEATURE_COLUMNS = [
    # Relative band powers
    "delta_relative",
    "theta_relative",
    "alpha_relative",
    "beta_relative",
    "gamma_relative",

    # Absolute powers
    "delta_power",
    "theta_power",
    "alpha_power",
    "beta_power",
    "gamma_power",

    # Ratios
    "theta_beta_ratio",
    "alpha_beta_ratio",

    # Spectral
    "spectral_entropy",
    "spectral_centroid",
    "spectral_bandwidth",

    # Hjorth
    "hjorth_activity",
    "hjorth_mobility",
    "hjorth_complexity",

    # Time domain
    "signal_std",
    "signal_rms",
    "signal_ptp",

    # Composite features
    "slow_wave_index",
    "fast_wave_index",
    "alpha_dominance",
    "activation_ratio",
    "relaxation_ratio",
]


# ================================================================
# 12. PERSONALIZED BRAIN-STATE MODEL
# ================================================================

class NeuroSyncPersonalModel:

    def __init__(self):

        self.baseline = (
            PersonalBaseline()
        )

        self.model = None

        self.anomaly_detector = None

        self.labels = None

        self.feature_columns = (
            FEATURE_COLUMNS
        )

    def fit(self, calibration_df):

        if "label" not in calibration_df.columns:
            raise ValueError(
                "Calibration dataset must "
                "contain a label column."
            )

        # --------------------------------------------------------
        # Remove invalid rows
        # --------------------------------------------------------

        calibration_df = (
            calibration_df
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .dropna(
                subset=self.feature_columns +
                ["label"]
            )
            .copy()
        )

        # --------------------------------------------------------
        # Check labels
        # --------------------------------------------------------

        labels = (
            calibration_df["label"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        unique_labels = sorted(
            labels.unique()
        )

        if len(unique_labels) < 2:

            raise ValueError(
                "At least TWO different "
                "calibration states are required."
            )

        self.labels = unique_labels

        # --------------------------------------------------------
        # Personal baseline
        #
        # We use the 'normal' state if it exists.
        # Otherwise we use all calibration data.
        # --------------------------------------------------------

        normal_mask = (
            labels == "normal"
        )

        if normal_mask.sum() >= 3:

            baseline_df = (
                calibration_df[
                    normal_mask
                ]
            )

        else:

            baseline_df = calibration_df

        self.baseline.fit(
            baseline_df,
            self.feature_columns
        )

        # --------------------------------------------------------
        # Transform into personal z-scores
        # --------------------------------------------------------

        X = self.baseline.transform(
            calibration_df
        )

        y = labels.values

        # --------------------------------------------------------
        # Personalized classifier
        #
        # RBF SVM is useful here because EEG state boundaries
        # are unlikely to be purely linear.
        # --------------------------------------------------------

        self.model = Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    probability=True,
                    class_weight="balanced",
                    random_state=RANDOM_STATE
                )
            )
        ])

        self.model.fit(
            X,
            y
        )

        # Train anomaly detector separately on NORMAL calibration
        # windows. It is intentionally independent of the SVM.
        self.anomaly_detector = (
            PersonalEEGAnomalyDetector()
        )

        self.anomaly_detector.fit(
            calibration_df
        )

        return self

    def predict(self, df):

        if self.model is None:

            raise RuntimeError(
                "Model has not been calibrated."
            )

        X = self.baseline.transform(
            df
        )

        predictions = (
            self.model.predict(X)
        )

        probabilities = (
            self.model.predict_proba(X)
        )

        classes = (
            self.model
            .named_steps["classifier"]
            .classes_
        )

        confidence = (
            np.max(
                probabilities,
                axis=1
            )
        )

        result = df.copy()

        result[
            "predicted_state"
        ] = predictions

        result[
            "state_confidence"
        ] = confidence

        # Store probability for every state
        for i, cls in enumerate(classes):

            result[
                f"prob_{cls}"
            ] = probabilities[:, i]

        return result



# ================================================================
# 13. PERSONALIZED EEG DEVIATION / ANOMALY DETECTION
# ================================================================
#
# State classifier:
#     What calibrated state does this EEG resemble?
#
# Anomaly detector:
#     How unusual is this EEG compared with this user's NORMAL EEG?
#
# Isolation Forest is trained only on NORMAL calibration windows.
#
# IMPORTANT:
# This is a personalized deviation detector, NOT a medical
# abnormality detector or diagnostic system.
# ================================================================

class PersonalEEGAnomalyDetector:

    def __init__(
        self,
        contamination=ANOMALY_CONTAMINATION,
        n_estimators=ANOMALY_N_ESTIMATORS,
        random_state=RANDOM_STATE
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state

        self.baseline = PersonalBaseline()
        self.model = None

        self.feature_columns = list(
            FEATURE_COLUMNS
        )

        self.train_min_score = None
        self.train_max_score = None

    def fit(self, calibration_df):

        if "label" not in calibration_df.columns:
            raise ValueError(
                "Calibration dataset must contain a label column."
            )

        clean = (
            calibration_df
            .replace([np.inf, -np.inf], np.nan)
            .dropna(
                subset=self.feature_columns + ["label"]
            )
            .copy()
        )

        labels = (
            clean["label"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        normal_df = clean[
            labels == "normal"
        ].copy()

        if len(normal_df) < 10:
            raise ValueError(
                "At least 10 NORMAL calibration windows are required "
                "for anomaly detection.\n"
                f"Only {len(normal_df)} normal windows were found.\n\n"
                "Recommended: collect 2-5 minutes of normal EEG."
            )

        # Personal normal reference
        self.baseline.fit(
            normal_df,
            self.feature_columns
        )

        X_normal = self.baseline.transform(
            normal_df
        ).replace(
            [np.inf, -np.inf],
            np.nan
        ).fillna(0.0)

        # Isolation Forest learns the normal feature distribution.
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1
        )

        self.model.fit(X_normal)

        train_scores = self.model.decision_function(
            X_normal
        )

        self.train_min_score = float(
            np.min(train_scores)
        )

        self.train_max_score = float(
            np.max(train_scores)
        )

        return self

    def _normalize_deviation_score(self, raw_scores):

        # Isolation Forest decision_function:
        # higher = more normal.
        # Therefore invert it so higher = more deviation.
        inverted = -np.asarray(
            raw_scores,
            dtype=float
        )

        low = -self.train_max_score
        high = -self.train_min_score

        denominator = max(
            high - low,
            1e-8
        )

        score = (
            (inverted - low) /
            denominator
        )

        return np.clip(
            score,
            0.0,
            1.0
        )

    def predict(
        self,
        df,
        score_threshold=ANOMALY_SCORE_THRESHOLD
    ):

        if self.model is None:
            raise RuntimeError(
                "Anomaly detector has not been trained."
            )

        X = self.baseline.transform(
            df
        ).replace(
            [np.inf, -np.inf],
            np.nan
        ).fillna(0.0)

        raw_scores = self.model.decision_function(X)

        forest_predictions = self.model.predict(X)

        anomaly_score = (
            self._normalize_deviation_score(
                raw_scores
            )
        )

        anomaly_flag = (
            (forest_predictions == -1) &
            (anomaly_score >= score_threshold)
        )

        result = df.copy()

        result["anomaly_raw_score"] = raw_scores

        # 0-1 display score; NOT a probability.
        result["anomaly_score"] = anomaly_score

        result["anomaly_prediction"] = np.where(
            anomaly_flag,
            "ANOMALY",
            "NORMAL"
        )

        return result


def apply_anomaly_persistence(
    result,
    min_consecutive=ANOMALY_MIN_CONSECUTIVE_WINDOWS,
    score_threshold=ANOMALY_SCORE_THRESHOLD
):
    """
    A single unusual EEG window is not immediately treated as a
    persistent deviation.

    With a 2-second step and min_consecutive=3, persistence begins
    after approximately 6 seconds of consecutive candidate windows.
    """

    result = result.copy()

    candidate = (
        result["anomaly_prediction"]
        .astype(str)
        .eq("ANOMALY")
        &
        (
            result["anomaly_score"]
            >= score_threshold
        )
    )

    persistent = np.zeros(
        len(result),
        dtype=bool
    )

    run_length = 0

    for i, flag in enumerate(candidate.values):

        if flag:
            run_length += 1
        else:
            run_length = 0

        if run_length >= min_consecutive:
            start = i - min_consecutive + 1
            persistent[start:i + 1] = True

    result["persistent_anomaly"] = persistent

    result["deviation_status"] = np.where(
        persistent,
        "PERSISTENT DEVIATION",
        np.where(
            candidate,
            "CANDIDATE DEVIATION",
            "NORMAL"
        )
    )

    return result


def calculate_session_anomaly_summary(
    result,
    step_seconds=STEP_SECONDS
):
    """
    Session-level deviation summary.

    Prototype thresholds:
        < 5%   persistent deviation -> STABLE
        < 15%  persistent deviation -> MILD DEVIATION
        >=15%  persistent deviation -> SIGNIFICANT DEVIATION

    These are engineering thresholds, not clinical thresholds.
    """

    total_windows = len(result)

    if total_windows == 0:
        return {
            "total_windows": 0,
            "candidate_anomalous_windows": 0,
            "candidate_deviation_percent": 0.0,
            "persistent_anomalous_windows": 0,
            "persistent_deviation_percent": 0.0,
            "persistent_episodes": 0,
            "longest_episode_seconds": 0.0,
            "peak_anomaly_score": 0.0,
            "mean_anomaly_score": 0.0,
            "session_anomaly_status": "NO DATA",
        }

    persistent_mask = (
        result["persistent_anomaly"]
        .astype(bool)
        .values
    )

    candidate_mask = (
        result["anomaly_prediction"]
        .astype(str)
        .eq("ANOMALY")
        .values
    )

    persistent_count = int(
        np.sum(persistent_mask)
    )

    candidate_count = int(
        np.sum(candidate_mask)
    )

    persistent_percent = (
        100.0 * persistent_count / total_windows
    )

    candidate_percent = (
        100.0 * candidate_count / total_windows
    )

    episode_count = 0
    longest_episode_windows = 0
    current_run = 0

    for flag in persistent_mask:

        if flag:
            current_run += 1

            if current_run == 1:
                episode_count += 1

            longest_episode_windows = max(
                longest_episode_windows,
                current_run
            )

        else:
            current_run = 0

    longest_episode_seconds = (
        longest_episode_windows *
        step_seconds
    )

    peak_anomaly_score = float(
        np.max(result["anomaly_score"])
    )

    mean_anomaly_score = float(
        np.mean(result["anomaly_score"])
    )

    if persistent_percent < SESSION_MILD_DEVIATION_PERCENT:
        session_status = "STABLE"
    elif persistent_percent < SESSION_SIGNIFICANT_DEVIATION_PERCENT:
        session_status = "MILD DEVIATION"
    else:
        session_status = "SIGNIFICANT DEVIATION"

    return {
        "total_windows": total_windows,
        "candidate_anomalous_windows": candidate_count,
        "candidate_deviation_percent": candidate_percent,
        "persistent_anomalous_windows": persistent_count,
        "persistent_deviation_percent": persistent_percent,
        "persistent_episodes": episode_count,
        "longest_episode_seconds": longest_episode_seconds,
        "peak_anomaly_score": peak_anomaly_score,
        "mean_anomaly_score": mean_anomaly_score,
        "session_anomaly_status": session_status,
    }


def print_session_anomaly_summary(summary):

    print(
        "\n================================================"
    )

    print(
        "        NEUROSYNC EEG DEVIATION SUMMARY"
    )

    print(
        "================================================"
    )

    print(
        f"Total analysis windows       : "
        f"{summary['total_windows']}"
    )

    print(
        f"Candidate deviation windows : "
        f"{summary['candidate_anomalous_windows']}"
    )

    print(
        f"Candidate deviation         : "
        f"{summary['candidate_deviation_percent']:.2f}%"
    )

    print(
        f"Persistent deviation windows: "
        f"{summary['persistent_anomalous_windows']}"
    )

    print(
        f"Persistent deviation        : "
        f"{summary['persistent_deviation_percent']:.2f}%"
    )

    print(
        f"Persistent episodes         : "
        f"{summary['persistent_episodes']}"
    )

    print(
        f"Longest deviation episode  : "
        f"{summary['longest_episode_seconds']:.1f} sec"
    )

    print(
        f"Peak deviation score       : "
        f"{summary['peak_anomaly_score']:.3f}"
    )

    print(
        f"Mean deviation score       : "
        f"{summary['mean_anomaly_score']:.3f}"
    )

    print(
        "\nSESSION STATUS             : "
        f"{summary['session_anomaly_status']}"
    )

    print(
        "\nNOTE: This status represents deviation from the "
        "user's calibrated NORMAL EEG baseline."
    )

    print(
        "It is not a medical abnormality or diagnosis."
    )

    print(
        "================================================"
    )



# ================================================================
# 13. CONVERT STATE INTO POSITIVE / NEUTRAL / NEGATIVE
# ================================================================

POSITIVE_STATES = {
    "happy",
    "positive",
    "relaxed",
    "calm"
}

NEUTRAL_STATES = {
    "normal",
    "neutral",
    "baseline"
}

NEGATIVE_STATES = {
    "sad",
    "negative",
    "angry",
    "stressed",
    "stress",
}


def classify_polarity(state):

    state = str(
        state
    ).lower().strip()

    if state in POSITIVE_STATES:
        return "POSITIVE"

    if state in NEGATIVE_STATES:
        return "NEGATIVE"

    if state in NEUTRAL_STATES:
        return "NEUTRAL"

    # Unknown states are treated as neutral
    # rather than making a dangerous assumption.
    return "NEUTRAL"


# ================================================================
# 14. DERIVED PERSONAL INDICATORS
# ================================================================

def calculate_personal_indicators(
    result_df,
    baseline
):
    """
    Calculate interpretable indicators relative to
    the user's personal baseline.

    IMPORTANT:
    These are EEG-derived proxies.
    They are NOT clinical measurements.
    """

    result = result_df.copy()

    z = baseline.transform(
        result
    )

    # ------------------------------------------------------------
    # Cognitive load proxy
    # ------------------------------------------------------------
    #
    # Uses:
    #   beta
    #   theta/beta
    #   fast-wave activity
    #
    # This is a heuristic indicator, NOT a medical diagnosis.
    # ------------------------------------------------------------

    cognitive_raw = (
        0.50 * z["beta_relative"] +
        0.30 * z["activation_ratio"] -
        0.20 * z["theta_beta_ratio"]
    )

    result[
        "cognitive_load_score"
    ] = np.clip(
        50 +
        15 * cognitive_raw,
        0,
        100
    )

    # ------------------------------------------------------------
    # Relaxation proxy
    # ------------------------------------------------------------

    relaxation_raw = (
        0.55 * z["alpha_relative"] +
        0.30 * z["relaxation_ratio"] -
        0.15 * z["activation_ratio"]
    )

    result[
        "relaxation_score"
    ] = np.clip(
        50 +
        15 * relaxation_raw,
        0,
        100
    )

    # ------------------------------------------------------------
    # Activation proxy
    # ------------------------------------------------------------

    activation_raw = (
        0.55 * z["beta_relative"] +
        0.25 * z["gamma_relative"] +
        0.20 * z["fast_wave_index"]
    )

    result[
        "activation_score"
    ] = np.clip(
        50 +
        15 * activation_raw,
        0,
        100
    )

    # ------------------------------------------------------------
    # Fatigue-related proxy
    # ------------------------------------------------------------

    fatigue_raw = (
        0.45 * z["theta_relative"] +
        0.35 * z["slow_wave_index"] -
        0.20 * z["alpha_relative"]
    )

    result[
        "fatigue_indicator"
    ] = np.clip(
        50 +
        15 * fatigue_raw,
        0,
        100
    )

    # ------------------------------------------------------------
    # EEG stability
    #
    # Higher absolute deviation from baseline means lower
    # stability.
    # ------------------------------------------------------------

    mean_abs_z = (
        np.mean(
            np.abs(z),
            axis=1
        )
    )

    stability = (
        100 -
        20 * mean_abs_z
    )

    result[
        "eeg_stability_score"
    ] = np.clip(
        stability,
        0,
        100
    )

    return result


# ================================================================
# 15. FINAL BRAIN STATE SCORE
# ================================================================

def calculate_brain_state_score(result):

    """
    Produces a single 0-100 brain-state score.

    This is NOT BBSI.

    It represents the EEG-side state only.

    Positive state:
        higher score

    Neutral:
        middle score

    Negative:
        lower score
    """

    polarity = (
        result[
            "predicted_polarity"
        ]
        .astype(str)
        .str.upper()
    )

    score = np.zeros(
        len(result)
    )

    for i, state in enumerate(polarity):

        if state == "POSITIVE":
            score[i] = 80

        elif state == "NEGATIVE":
            score[i] = 20

        else:
            score[i] = 50

    # Modify according to indicators
    #
    # This prevents the final score from being purely
    # dependent on the classifier label.

    score += (
        0.10 *
        (
            result[
                "relaxation_score"
            ].values -
            50
        )
    )

    score -= (
        0.10 *
        (
            result[
                "cognitive_load_score"
            ].values -
            50
        )
    )

    score += (
        0.05 *
        (
            result[
                "eeg_stability_score"
            ].values -
            50
        )
    )

    result[
        "brain_state_score"
    ] = np.clip(
        score,
        0,
        100
    )

    return result


# ================================================================
# 16. HUMAN-READABLE INTERPRETATION
# ================================================================

def level_from_score(score):

    if score < 35:
        return "LOW"

    if score < 65:
        return "MODERATE"

    return "HIGH"


def add_interpretations(result):

    result = result.copy()

    result[
        "cognitive_load_level"
    ] = result[
        "cognitive_load_score"
    ].apply(level_from_score)

    result[
        "relaxation_level"
    ] = result[
        "relaxation_score"
    ].apply(level_from_score)

    result[
        "activation_level"
    ] = result[
        "activation_score"
    ].apply(level_from_score)

    result[
        "fatigue_level"
    ] = result[
        "fatigue_indicator"
    ].apply(level_from_score)

    result[
        "stability_level"
    ] = result[
        "eeg_stability_score"
    ].apply(level_from_score)

    return result


# ================================================================
# 17. COMPLETE CALIBRATION DATASET BUILDER
# ================================================================

def build_calibration_dataset(
    recordings
):
    """
    recordings format:

    [
        ("happy.csv", "happy"),
        ("normal.csv", "normal"),
        ("sad.csv", "sad"),
        ("angry.csv", "angry"),
    ]

    Each recording is converted into multiple 4-second
    windows.
    """

    all_data = []

    print(
        "\n========================================"
    )
    print(
        "BUILDING PERSONAL CALIBRATION DATASET"
    )
    print(
        "========================================"
    )

    for file, label in recordings:

        print(
            f"\nProcessing: {file}"
        )

        data = process_recording(
            file,
            label=label
        )

        count = len(data)

        print(
            f"State      : {label}"
        )

        print(
            f"Windows    : {count}"
        )

        if count < MIN_CALIBRATION_WINDOWS:

            print(
                f"WARNING: only {count} windows "
                f"available for {label}."
            )

        all_data.append(
            data
        )

    calibration = pd.concat(
        all_data,
        ignore_index=True
    )

    print(
        "\n========================================"
    )
    print(
        "CALIBRATION DATASET SUMMARY"
    )
    print(
        "========================================"
    )

    print(
        calibration["label"]
        .value_counts()
    )

    print(
        f"\nTotal calibration windows: "
        f"{len(calibration)}"
    )

    return calibration


# ================================================================
# 18. TRAIN PERSONALIZED MODEL
# ================================================================

def train_personalized_model(
    calibration_df
):

    print(
        "\n========================================"
    )
    print(
        "PERSONALIZED MODEL TRAINING"
    )
    print(
        "========================================"
    )

    model = (
        NeuroSyncPersonalModel()
    )

    model.fit(
        calibration_df
    )

    print(
        "\nCalibration complete."
    )

    print(
        "Personal states learned:"
    )

    for label in model.labels:
        print(
            f"  - {label}"
        )

    return model


# ================================================================
# 19. ANALYZE NEW / LIVE EEG
# ================================================================

def analyze_new_recording(
    file,
    model
):

    print(
        "\n========================================"
    )
    print(
        "ANALYZING NEW EEG RECORDING"
    )
    print(
        "========================================"
    )

    data = process_recording(
        file
    )

    predictions = model.predict(
        data
    )

    # Personalized EEG deviation detection
    predictions = (
        model.anomaly_detector.predict(
            predictions
        )
    )

    # Persistence filter prevents a single unusual window from
    # immediately becoming a persistent deviation.
    predictions = (
        apply_anomaly_persistence(
            predictions
        )
    )

    predictions[
        "predicted_polarity"
    ] = predictions[
        "predicted_state"
    ].apply(
        classify_polarity
    )

    predictions = (
        calculate_personal_indicators(
            predictions,
            model.baseline
        )
    )

    predictions = (
        calculate_brain_state_score(
            predictions
        )
    )

    predictions = (
        add_interpretations(
            predictions
        )
    )

    return predictions


# ================================================================
# 20. CURRENT BRAIN STATE
# ================================================================

def print_current_brain_state(
    result
):

    latest = result.iloc[-1]

    print(
        "\n\n"
        "================================================"
    )

    print(
        "           NEUROSYNC BRAIN STATE"
    )

    print(
        "================================================"
    )

    print(
        f"Time                  : "
        f"{latest['start_time']:.1f} - "
        f"{latest['end_time']:.1f} sec"
    )

    print(
        f"Brain state           : "
        f"{str(latest['predicted_state']).upper()}"
    )

    print(
        f"State category        : "
        f"{latest['predicted_polarity']}"
    )

    print(
        f"Confidence            : "
        f"{latest['state_confidence'] * 100:.1f}%"
    )

    print(
        f"Brain-state score     : "
        f"{latest['brain_state_score']:.1f}/100"
    )

    print(
        "\n--------------- INDICATORS ----------------"
    )

    print(
        f"Cognitive load        : "
        f"{latest['cognitive_load_level']} "
        f"({latest['cognitive_load_score']:.1f})"
    )

    print(
        f"Relaxation            : "
        f"{latest['relaxation_level']} "
        f"({latest['relaxation_score']:.1f})"
    )

    print(
        f"Activation            : "
        f"{latest['activation_level']} "
        f"({latest['activation_score']:.1f})"
    )

    print(
        f"Fatigue-related       : "
        f"{latest['fatigue_level']} "
        f"({latest['fatigue_indicator']:.1f})"
    )

    print(
        f"EEG stability         : "
        f"{latest['stability_level']} "
        f"({latest['eeg_stability_score']:.1f})"
    )

    print(
        "\n----------- EEG DEVIATION STATUS -----------"
    )

    print(
        f"Deviation status      : "
        f"{latest['deviation_status']}"
    )

    print(
        f"Deviation score       : "
        f"{latest['anomaly_score']:.3f}"
    )

    print(
        f"Anomaly flag          : "
        f"{latest['anomaly_prediction']}"
    )

    print(
        "\n--------------- EEG BANDS -----------------"
    )

    print(
        f"Delta                 : "
        f"{latest['delta_relative'] * 100:.2f}%"
    )

    print(
        f"Theta                 : "
        f"{latest['theta_relative'] * 100:.2f}%"
    )

    print(
        f"Alpha                 : "
        f"{latest['alpha_relative'] * 100:.2f}%"
    )

    print(
        f"Beta                  : "
        f"{latest['beta_relative'] * 100:.2f}%"
    )

    print(
        f"Gamma                 : "
        f"{latest['gamma_relative'] * 100:.2f}%"
    )

    print(
        "================================================"
    )


# ================================================================
# 21. PLOT EEG BAND ACTIVITY
# ================================================================

def plot_band_activity(result):

    plt.figure(
        figsize=(12, 5)
    )

    x = result[
        "start_time"
    ]

    for band in [
        "delta",
        "theta",
        "alpha",
        "beta",
        "gamma"
    ]:

        plt.plot(
            x,
            result[
                f"{band}_relative"
            ],
            label=band.capitalize()
        )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Relative Power"
    )

    plt.title(
        "NeuroSync EEG Band Activity"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.show()


# ================================================================
# 22. PLOT BRAIN STATE
# ================================================================

def plot_brain_state(result):

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        result["start_time"],
        result["brain_state_score"],
        marker="o"
    )

    plt.axhline(
        50,
        linestyle="--"
    )

    plt.ylim(
        0,
        100
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Brain-state score"
    )

    plt.title(
        "NeuroSync Personalized Brain-State Score"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.show()


# ================================================================
# 23. PLOT MENTAL-STATE INDICATORS
# ================================================================

def plot_indicators(result):

    plt.figure(
        figsize=(12, 5)
    )

    x = result[
        "start_time"
    ]

    plt.plot(
        x,
        result[
            "cognitive_load_score"
        ],
        label="Cognitive Load"
    )

    plt.plot(
        x,
        result[
            "relaxation_score"
        ],
        label="Relaxation"
    )

    plt.plot(
        x,
        result[
            "activation_score"
        ],
        label="Activation"
    )

    plt.plot(
        x,
        result[
            "fatigue_indicator"
        ],
        label="Fatigue-related"
    )

    plt.plot(
        x,
        result[
            "eeg_stability_score"
        ],
        label="EEG Stability"
    )

    plt.ylim(
        0,
        100
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Score"
    )

    plt.title(
        "NeuroSync EEG-Derived Indicators"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.show()



# ================================================================
# 24. PLOT PERSONALIZED EEG DEVIATION
# ================================================================

def plot_anomaly_activity(result):

    plt.figure(
        figsize=(12, 5)
    )

    x = result["start_time"]

    plt.plot(
        x,
        result["anomaly_score"],
        label="Personalized Deviation Score"
    )

    plt.axhline(
        ANOMALY_SCORE_THRESHOLD,
        linestyle="--",
        label="Deviation Threshold"
    )

    persistent = (
        result["persistent_anomaly"]
        .astype(bool)
    )

    if persistent.any():

        plt.scatter(
            x[persistent],
            result.loc[
                persistent,
                "anomaly_score"
            ],
            marker="x",
            label="Persistent Deviation"
        )

    plt.ylim(
        0,
        1
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Deviation score"
    )

    plt.title(
        "NeuroSync Personalized EEG Deviation"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.show()


# ================================================================
# 24. EXPORT RESULTS
# ================================================================

def export_results(
    result,
    output_file="neurosync_brain_state_results.csv"
):

    result.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nResults saved to:"
        f"\n{output_file}"
    )


# ================================================================
# 25. GOOGLE COLAB UPLOAD HELPERS
# ================================================================

def upload_files():

    from google.colab import files

    uploaded = files.upload()

    return uploaded


