"""
main.py
End-to-end pipeline:
  1. Collect RIPE Atlas measurement data
  2. Preprocess and engineer features
  3. Run anomaly detection
  4. Generate all plots
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from data_collection  import collect_measurements, MEASUREMENT_IDS, PROBE_IDS
from data_preprocessing import load_and_clean, engineer_features
from anomaly_detection  import run_all, summarize
from visualization      import (plot_rtt_timeseries, plot_detector_comparison,
                                 plot_rtt_distribution, plot_score_scatter,
                                 plot_heatmap_rtt, plot_packet_loss)


def parse_args():
    p = argparse.ArgumentParser(description="RIPE Atlas Anomaly Detection Pipeline")
    p.add_argument("--collect",      action="store_true",
                   help="Fetch data from RIPE Atlas API (requires network access)")
    p.add_argument("--csv",          default="measurement_data.csv",
                   help="Path to raw CSV (input if --collect not used, output if it is)")
    p.add_argument("--output-dir",   default=None,
                   help="Directory for output files (CSV + plots). "
                        "Defaults to ripe/output/ next to this script.")
    p.add_argument("--start",        default="2024-01-01",
                   help="Collection start date YYYY-MM-DD (used with --collect)")
    p.add_argument("--end",          default="2024-01-07",
                   help="Collection end date YYYY-MM-DD (used with --collect)")
    p.add_argument("--contamination", type=float, default=0.05,
                   help="Expected anomaly fraction for ML detectors (0–0.5)")
    p.add_argument("--min-votes",    type=int, default=2,
                   help="Min detector votes to flag ensemble anomaly")
    return p.parse_args()


def main():
    args = parse_args()
    out  = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "output"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out.resolve()}")

    raw_csv       = out / args.csv
    features_csv  = out / "measurement_features.csv"
    anomalies_csv = out / "measurement_anomalies.csv"

    # ── Step 1: Data Collection ───────────────────────────────────────────────
    if args.collect:
        start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end   = datetime.strptime(args.end,   "%Y-%m-%d").replace(tzinfo=timezone.utc)
        collect_measurements(MEASUREMENT_IDS, start, end,
                             probe_ids=PROBE_IDS, output_file=str(raw_csv))
    else:
        if not raw_csv.exists():
            print(f"ERROR: {raw_csv} not found. Run with --collect to fetch data first.")
            return

    # ── Step 2: Preprocessing ─────────────────────────────────────────────────
    print("\n── Preprocessing ──")
    df = load_and_clean(str(raw_csv))

    if len(df) == 0:
        print(
            "\nERROR: No usable rows after cleaning.\n"
            "Possible causes:\n"
            "  • The measurement IDs returned no data for the given date range\n"
            "  • The measurements are not ping-type (check on atlas.ripe.net/measurements/<id>)\n"
            "  • All rows had null avg_rtt — re-run --collect to refresh the CSV\n"
        )
        return

    df = engineer_features(df)
    df.to_csv(features_csv, index=False)
    print(f"Features saved → {features_csv}")

    # ── Step 3: Anomaly Detection ─────────────────────────────────────────────
    print("\n── Anomaly Detection ──")
    df      = run_all(df, contamination=args.contamination, min_votes=args.min_votes)
    summary = summarize(df)
    df.to_csv(anomalies_csv, index=False)
    print(f"Annotated results saved → {anomalies_csv}")

    # ── Step 4: Visualisation ─────────────────────────────────────────────────
    print("\n── Generating Plots ──")
    plot_rtt_timeseries(    df, save_path=str(out / "plot_rtt_timeseries.png"))
    plot_detector_comparison(summary, save_path=str(out / "plot_detector_comparison.png"))
    plot_rtt_distribution(  df, save_path=str(out / "plot_rtt_distribution.png"))
    plot_score_scatter(     df, save_path=str(out / "plot_score_scatter.png"))
    plot_heatmap_rtt(       df, save_path=str(out / "plot_heatmap_rtt.png"))
    plot_packet_loss(       df, save_path=str(out / "plot_packet_loss.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()
