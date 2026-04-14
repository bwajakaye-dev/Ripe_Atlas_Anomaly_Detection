"""
anomaly_detection.py
Four complementary anomaly detection methods applied to RIPE Atlas network metrics:

  1. Z-Score          – statistical, per-probe global baseline
  2. IQR              – robust statistical, per-probe (resistant to outliers)
  3. Isolation Forest – ensemble ML, multivariate
  4. Local Outlier Factor (LOF) – density-based ML, multivariate
  5. Rolling Z-Score  – time-series specific, detects contextual anomalies

Each detector adds a boolean column  <method>_anomaly  and a score column
<method>_score  to the DataFrame. A final  ensemble_anomaly  column flags
rows marked anomalous by at least N of the 5 detectors.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


# ── 1. Z-Score ────────────────────────────────────────────────────────────────

def zscore_detector(df: pd.DataFrame, col: str = "avg_rtt",
                    threshold: float = 3.0) -> pd.DataFrame:
    """
    Flag rows where the per-probe z-score of `col` exceeds `threshold`.
    Uses the pre-computed z_score_rtt column if available and col == avg_rtt.
    """
    df = df.copy()

    if col == "avg_rtt" and "z_score_rtt" in df.columns:
        scores = df["z_score_rtt"].abs()
    else:
        grp = df.groupby("probe_id")[col]
        scores = ((df[col] - grp.transform("mean")) / grp.transform("std")).abs()

    df["zscore_score"]   = scores
    df["zscore_anomaly"] = scores > threshold
    return df


# ── 2. IQR ────────────────────────────────────────────────────────────────────

def iqr_detector(df: pd.DataFrame, col: str = "avg_rtt",
                 multiplier: float = 1.5) -> pd.DataFrame:
    """
    Flag rows where `col` falls outside  [Q1 - k*IQR, Q3 + k*IQR]
    computed per probe (robust to extreme outliers compared to z-score).
    """
    df = df.copy()

    def _iqr_flag(series: pd.Series) -> pd.Series:
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
        # Score = distance outside the fence (0 if inside)
        score = np.maximum(lower - series, series - upper).clip(lower=0)
        return score

    scores = df.groupby("probe_id")[col].transform(_iqr_flag)
    df["iqr_score"]   = scores
    df["iqr_anomaly"] = scores > 0
    return df


# ── 3. Isolation Forest ────────────────────────────────────────────────────────

def isolation_forest_detector(df: pd.DataFrame,
                               feature_cols: list[str] | None = None,
                               contamination: float = 0.05,
                               random_state: int = 42) -> pd.DataFrame:
    """
    Multivariate anomaly detection using Isolation Forest.
    contamination = expected fraction of anomalies in the data.
    """
    df = df.copy()

    if feature_cols is None:
        feature_cols = [c for c in
                        ["avg_rtt", "jitter", "packet_loss_pct",
                         "rolling_std_rtt", "hour_of_day"]
                        if c in df.columns]

    valid_mask = df[feature_cols].notna().all(axis=1)
    X = df.loc[valid_mask, feature_cols].values

    if len(X) < 10:
        print(f"  Skipping Isolation Forest: only {len(X)} usable samples (need ≥ 10)")
        df["iforest_score"]   = np.nan
        df["iforest_anomaly"] = False
        return df

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = IsolationForest(contamination=contamination,
                          random_state=random_state, n_jobs=-1)
    clf.fit(X_scaled)

    # score_samples returns negative anomaly scores: more negative = more anomalous
    raw_scores = clf.score_samples(X_scaled)
    predictions = clf.predict(X_scaled)   # -1 = anomaly, 1 = normal

    df["iforest_score"]   = np.nan
    df["iforest_anomaly"] = False
    df.loc[valid_mask, "iforest_score"]   = -raw_scores   # flip so higher = more anomalous
    df.loc[valid_mask, "iforest_anomaly"] = predictions == -1

    return df


# ── 4. Local Outlier Factor ───────────────────────────────────────────────────

def lof_detector(df: pd.DataFrame,
                 feature_cols: list[str] | None = None,
                 n_neighbors: int = 20,
                 contamination: float = 0.05) -> pd.DataFrame:
    """
    Density-based anomaly detection using Local Outlier Factor.
    Points in low-density neighbourhoods relative to their neighbours are flagged.
    """
    df = df.copy()

    if feature_cols is None:
        feature_cols = [c for c in
                        ["avg_rtt", "jitter", "packet_loss_pct",
                         "rolling_std_rtt", "hour_of_day"]
                        if c in df.columns]

    valid_mask = df[feature_cols].notna().all(axis=1)
    X = df.loc[valid_mask, feature_cols].values

    if len(X) < 4:
        print(f"  Skipping LOF: only {len(X)} usable samples (need ≥ 4)")
        df["lof_score"]   = np.nan
        df["lof_anomaly"] = False
        return df

    # n_neighbors must be < number of samples
    n_neighbors = min(n_neighbors, len(X) - 1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # LOF must be fit_predict at once (no separate predict after fit)
    clf = LocalOutlierFactor(n_neighbors=n_neighbors,
                             contamination=contamination,
                             n_jobs=-1)
    predictions = clf.fit_predict(X_scaled)   # -1 = anomaly, 1 = normal
    scores      = -clf.negative_outlier_factor_   # higher = more anomalous

    df["lof_score"]   = np.nan
    df["lof_anomaly"] = False
    df.loc[valid_mask, "lof_score"]   = scores
    df.loc[valid_mask, "lof_anomaly"] = predictions == -1

    return df


# ── 5. Rolling Z-Score (time-series contextual) ───────────────────────────────

def rolling_zscore_detector(df: pd.DataFrame,
                             col: str = "avg_rtt",
                             window: int = 10,
                             threshold: float = 3.0) -> pd.DataFrame:
    """
    Detects contextual anomalies: points that deviate from their local
    rolling window mean by more than `threshold` standard deviations.
    Unlike the global z-score this catches sudden spikes within a probe's
    otherwise stable baseline.
    """
    df = df.copy().sort_values(["probe_id", "datetime"])

    def _rolling_z(series: pd.Series) -> pd.Series:
        roll_mean = series.rolling(window, min_periods=3).mean()
        roll_std  = series.rolling(window, min_periods=3).std()
        return ((series - roll_mean) / roll_std.replace(0, np.nan)).abs()

    scores = df.groupby("probe_id")[col].transform(_rolling_z)
    df["rolling_z_score"]   = scores
    df["rolling_z_anomaly"] = scores > threshold
    return df


# ── Ensemble ──────────────────────────────────────────────────────────────────

def ensemble_anomalies(df: pd.DataFrame, min_votes: int = 2) -> pd.DataFrame:
    """
    Add an  ensemble_anomaly  column: True when at least `min_votes`
    individual detectors agree on a row being anomalous.
    """
    vote_cols = [c for c in
                 ["zscore_anomaly", "iqr_anomaly", "iforest_anomaly",
                  "lof_anomaly", "rolling_z_anomaly"]
                 if c in df.columns]

    df["vote_count"]      = df[vote_cols].sum(axis=1)
    df["ensemble_anomaly"] = df["vote_count"] >= min_votes
    return df


# ── Run all detectors ─────────────────────────────────────────────────────────

def run_all(df: pd.DataFrame,
            contamination: float = 0.05,
            min_votes: int = 2) -> pd.DataFrame:
    """Apply all five detectors and the ensemble, return annotated DataFrame."""
    df = zscore_detector(df)
    df = iqr_detector(df)
    df = isolation_forest_detector(df, contamination=contamination)
    df = lof_detector(df, contamination=contamination)
    df = rolling_zscore_detector(df)
    df = ensemble_anomalies(df, min_votes=min_votes)

    anomaly_count = df["ensemble_anomaly"].sum()
    total         = len(df)
    print(f"Ensemble anomalies: {anomaly_count} / {total} "
          f"({anomaly_count/total*100:.1f}%)")
    return df


# ── Summary ───────────────────────────────────────────────────────────────────

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Print per-detector anomaly counts and return a summary DataFrame."""
    detector_cols = {
        "Z-Score":          "zscore_anomaly",
        "IQR":              "iqr_anomaly",
        "Isolation Forest": "iforest_anomaly",
        "LOF":              "lof_anomaly",
        "Rolling Z-Score":  "rolling_z_anomaly",
        "Ensemble (≥2)":    "ensemble_anomaly",
    }
    rows = []
    for name, col in detector_cols.items():
        if col in df.columns:
            n = int(df[col].sum())
            pct = n / len(df) * 100
            rows.append({"Detector": name, "Anomalies": n, "Pct": f"{pct:.1f}%"})

    summary = pd.DataFrame(rows)
    print("\n── Anomaly Detection Summary ──")
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    from data_preprocessing import load_and_clean, engineer_features

    df = load_and_clean("measurement_data.csv")
    df = engineer_features(df)
    df = run_all(df)
    summarize(df)
    df.to_csv("measurement_anomalies.csv", index=False)
    print("\nResults saved → measurement_anomalies.csv")
