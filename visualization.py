"""
visualization.py
Plots for exploring RIPE Atlas measurement data and anomaly detection results.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np


# ── Helpers ────────────────────────────────────────────────────────────────────

def _savefig(fig: plt.Figure, path: str | None, title: str) -> None:
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"Saved → {path}")
    else:
        plt.show()
    plt.close(fig)


# ── 1. Time-series RTT with anomalies ─────────────────────────────────────────

def plot_rtt_timeseries(df: pd.DataFrame, probe_id=None,
                        anomaly_col: str = "ensemble_anomaly",
                        save_path: str | None = None) -> None:
    """
    Plot avg_rtt over time for one probe (or all probes averaged).
    Anomalous points are marked in red.
    """
    if probe_id is not None:
        data = df[df["probe_id"] == probe_id].copy()
        title = f"RTT Time Series — Probe {probe_id}"
    else:
        # Aggregate across all probes per timestamp
        data = (df.groupby("datetime")
                  .agg(avg_rtt=("avg_rtt", "mean"),
                       anomaly=(anomaly_col, "any"))
                  .reset_index()
                  .rename(columns={"anomaly": anomaly_col}))
        title = "RTT Time Series — All Probes (mean)"

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(data["datetime"], data["avg_rtt"],
            color="steelblue", linewidth=0.8, label="avg RTT (ms)")

    if anomaly_col in data.columns:
        anom = data[data[anomaly_col]]
        ax.scatter(anom["datetime"], anom["avg_rtt"],
                   color="red", s=20, zorder=5, label="anomaly")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate(rotation=30)
    ax.set_ylabel("avg RTT (ms)")
    ax.set_xlabel("Time (UTC)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    _savefig(fig, save_path, title)


# ── 2. Detector comparison bar chart ──────────────────────────────────────────

def plot_detector_comparison(summary_df: pd.DataFrame,
                              save_path: str | None = None) -> None:
    """
    Horizontal bar chart comparing anomaly counts across detectors.
    `summary_df` is the DataFrame returned by anomaly_detection.summarize().
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(summary_df["Detector"], summary_df["Anomalies"],
                   color="steelblue", edgecolor="white")
    ax.bar_label(bars, padding=4, fontsize=9)
    ax.set_xlabel("Number of anomalies flagged")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    _savefig(fig, save_path, "Anomaly Counts per Detector")


# ── 3. RTT distribution by probe ──────────────────────────────────────────────

def plot_rtt_distribution(df: pd.DataFrame,
                          top_n_probes: int = 8,
                          save_path: str | None = None) -> None:
    """
    Box plots of avg_rtt per probe for the N probes with most data points.
    """
    top_probes = (df.groupby("probe_id").size()
                    .nlargest(top_n_probes).index.tolist())
    subset = df[df["probe_id"].isin(top_probes)]

    fig, ax = plt.subplots(figsize=(10, 5))
    groups = [g["avg_rtt"].dropna().values
              for _, g in subset.groupby("probe_id")]
    labels = [str(p) for p in top_probes]

    bp = ax.boxplot(groups, labels=labels, patch_artist=True,
                    medianprops={"color": "red", "linewidth": 1.5})
    for patch in bp["boxes"]:
        patch.set_facecolor("lightsteelblue")

    ax.set_xlabel("Probe ID")
    ax.set_ylabel("avg RTT (ms)")
    ax.grid(axis="y", alpha=0.3)
    _savefig(fig, save_path, f"RTT Distribution — Top {top_n_probes} Probes")


# ── 4. Score scatter (Isolation Forest vs LOF) ────────────────────────────────

def plot_score_scatter(df: pd.DataFrame, save_path: str | None = None) -> None:
    """
    Scatter plot of Isolation Forest score vs LOF score,
    coloured by ensemble anomaly label.
    """
    required = {"iforest_score", "lof_score", "ensemble_anomaly"}
    if not required.issubset(df.columns):
        print("Skipping score scatter — run all detectors first.")
        return

    plot_df = df.dropna(subset=["iforest_score", "lof_score"])
    colors  = plot_df["ensemble_anomaly"].map({True: "red", False: "steelblue"})

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(plot_df["iforest_score"], plot_df["lof_score"],
               c=colors, s=10, alpha=0.5)

    # Legend proxy
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue",
               markersize=8, label="Normal"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
               markersize=8, label="Anomaly"),
    ]
    ax.legend(handles=handles, fontsize=9)
    ax.set_xlabel("Isolation Forest Score (higher = more anomalous)")
    ax.set_ylabel("LOF Score (higher = more anomalous)")
    ax.grid(alpha=0.3)
    _savefig(fig, save_path, "Isolation Forest vs LOF Scores")


# ── 5. Heatmap: avg RTT by hour of day and day of week ────────────────────────

def plot_heatmap_rtt(df: pd.DataFrame, save_path: str | None = None) -> None:
    """
    Heatmap showing mean avg_rtt for each (day-of-week × hour-of-day) cell.
    Useful for identifying diurnal / weekly patterns.
    """
    if "hour_of_day" not in df.columns or "day_of_week" not in df.columns:
        print("Skipping heatmap — run engineer_features first.")
        return

    pivot = (df.groupby(["day_of_week", "hour_of_day"])["avg_rtt"]
               .mean()
               .unstack(fill_value=np.nan))

    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", origin="upper")
    plt.colorbar(im, ax=ax, label="Mean avg RTT (ms)")

    ax.set_xticks(range(24))
    ax.set_xticklabels(range(24), fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([day_labels[i] for i in pivot.index])
    ax.set_xlabel("Hour of Day (UTC)")
    ax.set_ylabel("Day of Week")
    _savefig(fig, save_path, "Mean RTT Heatmap — Hour × Day of Week")


# ── 6. Packet loss over time ───────────────────────────────────────────────────

def plot_packet_loss(df: pd.DataFrame, save_path: str | None = None) -> None:
    """Bar chart of mean packet loss rate per hour."""
    if "packet_loss_pct" not in df.columns:
        print("Skipping packet loss plot — column not available.")
        return

    hourly = (df.groupby(df["datetime"].dt.floor("h"))["packet_loss_pct"]
                .mean()
                .reset_index())

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.bar(hourly["datetime"], hourly["packet_loss_pct"],
           width=0.03, color="tomato", alpha=0.8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate(rotation=30)
    ax.set_ylabel("Mean Packet Loss (%)")
    ax.set_xlabel("Time (UTC)")
    ax.grid(axis="y", alpha=0.3)
    _savefig(fig, save_path, "Hourly Mean Packet Loss")


if __name__ == "__main__":
    from data_preprocessing import load_and_clean, engineer_features
    from anomaly_detection import run_all, summarize

    df = load_and_clean("measurement_data.csv")
    df = engineer_features(df)
    df = run_all(df)
    summary = summarize(df)

    plot_rtt_timeseries(df, save_path="plot_rtt_timeseries.png")
    plot_detector_comparison(summary, save_path="plot_detector_comparison.png")
    plot_rtt_distribution(df, save_path="plot_rtt_distribution.png")
    plot_score_scatter(df, save_path="plot_score_scatter.png")
    plot_heatmap_rtt(df, save_path="plot_heatmap_rtt.png")
    plot_packet_loss(df, save_path="plot_packet_loss.png")
