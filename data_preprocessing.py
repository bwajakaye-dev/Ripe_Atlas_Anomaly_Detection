"""
data_preprocessing.py
Loads the raw CSV produced by data_collection.py, cleans it,
and engineers features used by the anomaly detection step.
"""

import pandas as pd
import numpy as np


NUMERIC_COLS = ["min_rtt", "avg_rtt", "max_rtt", "jitter", "packet_loss_pct"]


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """
    Load the raw measurement CSV, coerce types, and drop unusable rows.
    Returns a cleaned DataFrame sorted by probe and time.
    """
    df = pd.read_csv(csv_path)

    # Parse timestamps
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

    # Coerce numeric measurement columns
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Show null counts before dropping so problems are visible
    if "avg_rtt" in df.columns:
        null_avg = df["avg_rtt"].isna().sum()
        if null_avg:
            print(f"  Warning: {null_avg}/{len(df)} rows have null avg_rtt and will be dropped")

    # Drop rows where the key metric (avg_rtt) is missing
    before = len(df)
    df = df.dropna(subset=["avg_rtt", "datetime"])
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} rows (null avg_rtt or datetime)")

    # Negative RTT values are sensor errors — remove them
    for col in ["min_rtt", "avg_rtt", "max_rtt", "jitter"]:
        if col in df.columns:
            df = df[df[col].isna() | (df[col] >= 0)]

    df = df.sort_values(["probe_id", "datetime"]).reset_index(drop=True)
    print(f"Loaded {len(df)} clean rows from {csv_path}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features on top of the cleaned DataFrame.

    New columns
    -----------
    jitter          : max_rtt - min_rtt  (if not already present)
    packet_loss_pct : derived from sent/received (if not already present)
    hour_of_day     : 0–23
    day_of_week     : 0 = Monday … 6 = Sunday
    is_weekend      : bool
    rtt_range       : max_rtt - min_rtt (alias kept for clarity)
    rolling_avg_rtt : 5-measurement rolling mean per probe (trend baseline)
    rolling_std_rtt : 5-measurement rolling std per probe (local variability)
    z_score_rtt     : per-probe z-score of avg_rtt (global baseline)
    """
    df = df.copy()

    # Jitter — recalculate if missing
    if "jitter" not in df.columns or df["jitter"].isna().all():
        df["jitter"] = df["max_rtt"] - df["min_rtt"]

    # Packet loss — recalculate if missing
    if "packet_loss_pct" not in df.columns or df["packet_loss_pct"].isna().all():
        sent = pd.to_numeric(df.get("packets_sent"), errors="coerce")
        rcvd = pd.to_numeric(df.get("packets_received"), errors="coerce")
        df["packet_loss_pct"] = np.where(
            sent > 0, (sent - rcvd.fillna(0)) / sent * 100, np.nan
        )

    # Calendar features
    df["hour_of_day"]  = df["datetime"].dt.hour
    df["day_of_week"]  = df["datetime"].dt.dayofweek
    df["is_weekend"]   = df["day_of_week"].isin([5, 6]).astype(int)

    # Per-probe rolling statistics (window = 5 observations)
    df = df.sort_values(["probe_id", "datetime"])
    df["rolling_avg_rtt"] = (
        df.groupby("probe_id")["avg_rtt"]
          .transform(lambda s: s.rolling(5, min_periods=1).mean())
    )
    df["rolling_std_rtt"] = (
        df.groupby("probe_id")["avg_rtt"]
          .transform(lambda s: s.rolling(5, min_periods=2).std())
    )

    # Per-probe global z-score
    grp = df.groupby("probe_id")["avg_rtt"]
    df["z_score_rtt"] = (df["avg_rtt"] - grp.transform("mean")) / grp.transform("std")

    return df.reset_index(drop=True)


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return only the numeric feature columns used as input for ML models.
    Rows with any NaN in these columns are dropped.
    """
    feature_cols = [
        "avg_rtt", "min_rtt", "max_rtt", "jitter",
        "packet_loss_pct", "rolling_avg_rtt", "rolling_std_rtt",
        "hour_of_day", "day_of_week",
    ]
    available = [c for c in feature_cols if c in df.columns]
    return df[available].dropna()


if __name__ == "__main__":
    df = load_and_clean("measurement_data.csv")
    df = engineer_features(df)
    df.to_csv("measurement_features.csv", index=False)
    print(df[["datetime", "probe_id", "avg_rtt", "jitter",
              "packet_loss_pct", "z_score_rtt"]].head(10).to_string())
