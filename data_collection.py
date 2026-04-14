"""
data_collection.py
Fetches RIPE Atlas ping measurement results for a time range and saves them to CSV.
Uses AtlasResultsRequest (not AtlasLatestRequest) to support historical queries.
"""

from ripe.atlas.cousteau import AtlasResultsRequest
import csv
import os
from datetime import datetime, timezone
from pathlib import Path

# API key: set RIPE_ATLAS_API_KEY env var, or paste key here as fallback
API_KEY = os.environ.get("RIPE_ATLAS_API_KEY", "8a876b9c-bd77-44f3-b81c-c90301340047")

# Built-in measurement 1001: continuous global ping to k.root-servers.net.
# Guaranteed to have data for any date since 2010. No API key needed.
MEASUREMENT_IDS = [1001]

START_TIME = datetime(2024, 1, 1)
END_TIME   = datetime(2024, 1, 14)   # 2 weeks

# Limit to 30 geographically diverse probes so the fetch takes ~2–3 min
# instead of hours. Covers NA, EU, Asia-Pacific, South America, Africa.
PROBE_IDS = [
    # Europe
    1, 2, 6, 11, 20, 97, 165, 270, 681,
    # North America
    99, 141, 196, 579, 1216, 3557,
    # Asia-Pacific
    202, 564, 1130, 2561, 6126,
    # South America
    742, 1033, 5050,
    # Africa / Middle East
    630, 1136, 6096, 6352,
    # Oceania
    569, 3554,
]

OUTPUT_DIR  = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = str(OUTPUT_DIR / "measurement_data.csv")

FIELDNAMES = [
    "measurement_id", "probe_id", "timestamp", "datetime",
    "min_rtt", "avg_rtt", "max_rtt", "jitter",
    "packets_sent", "packets_received", "packet_loss_pct",
    "src_addr", "dst_addr", "dst_name",
]


def _parse_ping_result(result, msm_id):
    """Extract flat fields from a single RIPE Atlas ping result dict."""
    ts = result.get("timestamp")
    sent = result.get("sent")
    rcvd = result.get("rcvd")

    min_rtt = result.get("min")
    max_rtt = result.get("max")
    avg_rtt = result.get("avg")

    # RIPE Atlas returns avg/min/max = -1 when ALL packets were lost (100% loss).
    # Treat -1 as "no valid RTT" so these rows don't skew RTT statistics.
    # Packet loss will still be recorded correctly via sent/rcvd.
    if avg_rtt == -1:
        avg_rtt = None
    if min_rtt == -1:
        min_rtt = None
    if max_rtt == -1:
        max_rtt = None

    # Jitter = spread between best and worst RTT in this packet burst
    jitter = None
    if min_rtt is not None and max_rtt is not None:
        jitter = round(max_rtt - min_rtt, 4)

    # Packet loss percentage
    packet_loss_pct = None
    if sent and sent > 0:
        lost = sent - (rcvd or 0)
        packet_loss_pct = round(lost / sent * 100, 2)

    return {
        "measurement_id":    result.get("msm_id", msm_id),
        "probe_id":          result.get("prb_id"),
        "timestamp":         ts,
        "datetime":          datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else None,
        "min_rtt":           min_rtt,
        "avg_rtt":           avg_rtt,
        "max_rtt":           max_rtt,
        "jitter":            jitter,
        "packets_sent":      sent,
        "packets_received":  rcvd,
        "packet_loss_pct":   packet_loss_pct,
        "src_addr":          result.get("from"),
        "dst_addr":          result.get("dst_addr"),
        "dst_name":          result.get("dst_name"),
    }


def collect_measurements(measurement_ids, start_time, end_time,
                         probe_ids=None, output_file=OUTPUT_FILE):
    """
    Query RIPE Atlas for ping results over the given time window,
    parse each probe result, and save to a CSV.

    probe_ids : optional list of probe IDs to restrict the query.
                Pass PROBE_IDS to keep the dataset small and fast to fetch.
    Returns a list of dicts (one per probe measurement).
    """
    all_rows = []

    for msm_id in measurement_ids:
        print(f"Fetching measurement {msm_id}  "
              f"({start_time.date()} → {end_time.date()}) "
              f"[{len(probe_ids) if probe_ids else 'all'} probes] ...")

        kwargs = {
            "msm_id": msm_id,
            "start":  int(start_time.timestamp()),
            "stop":   int(end_time.timestamp()),
        }
        if probe_ids:
            kwargs["probe_ids"] = probe_ids
        # Only pass the key if we have one
        if API_KEY:
            kwargs["key"] = API_KEY

        is_success, results = AtlasResultsRequest(**kwargs).create()

        if not is_success:
            print(f"  ERROR: failed to retrieve measurement {msm_id}")
            continue

        count = 0
        for result in results:
            row = _parse_ping_result(result, msm_id)
            all_rows.append(row)
            count += 1

        print(f"  Retrieved {count} probe results.")

    if not all_rows:
        print("No data collected.")
        return all_rows

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved {len(all_rows)} rows → {output_file}")
    return all_rows


if __name__ == "__main__":
    collect_measurements(MEASUREMENT_IDS, START_TIME, END_TIME, probe_ids=PROBE_IDS)
