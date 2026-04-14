# Network Performance Anomaly Detection using RIPE Atlas

An end-to-end data mining pipeline that fetches real-world network measurement data from [RIPE Atlas](https://atlas.ripe.net), engineers time-series features, and applies five complementary anomaly detection algorithms to identify unusual patterns in latency, jitter, and packet loss.

---

## Problem Statement

Network performance anomalies — sudden spikes in latency, elevated packet loss, or unusual jitter — can indicate routing problems, congestion, DDoS attacks, or infrastructure failures. Detecting these events automatically from distributed probe data is a practical data mining problem: the signal is noisy, bursty, and highly dependent on geographic location and time of day.

This project applies data mining techniques to RIPE Atlas ping measurements to:
- Detect deviations in RTT (round-trip time), jitter, and packet loss across globally distributed probes
- Compare statistical and machine-learning-based anomaly detectors
- Visualize spatial and temporal anomaly patterns in real Internet traffic data

---

## Data Source

[RIPE Atlas](https://atlas.ripe.net) is the world's largest active Internet measurement network, operated by RIPE NCC. It consists of thousands of hardware probes deployed in homes, universities, and data centres across 200+ countries, continuously measuring Internet reachability and performance.

**Measurement used:** Built-in measurement **1001** — a continuous global ping to `k.root-servers.net` (one of the 13 DNS root nameservers), running every 240 seconds on all active probes since 2010.

| Field | Value |
|---|---|
| Measurement type | Ping (ICMP) |
| Target | k.root-servers.net |
| Frequency | Every 240 seconds per probe |
| Probes used | 30 (geographically curated) |
| Time window | 2024-01-01 → 2024-01-14 (2 weeks) |
| Approximate rows | ~30,000 probe measurements |

**Probe geographic coverage:**

| Region | Probes |
|---|---|
| Europe | Netherlands, Germany, UK, France + others |
| North America | USA (multiple locations) |
| Asia-Pacific | Japan, Singapore, Australia + others |
| South America | Brazil, Argentina |
| Africa / Middle East | South Africa, Kenya, UAE |

---

## Pipeline Architecture

```
RIPE Atlas API
      │
      ▼
data_collection.py      ← AtlasResultsRequest, probe filter, -1 handling
      │  measurement_data.csv
      ▼
data_preprocessing.py   ← cleaning, feature engineering
      │  measurement_features.csv
      ▼
anomaly_detection.py    ← 5 detectors + ensemble voting
      │  measurement_anomalies.csv
      ▼
visualization.py        ← 6 plots → ripe/output/*.png
```

### Feature Engineering (`data_preprocessing.py`)

Raw ping results are enriched with derived features before detection:

| Feature | Description |
|---|---|
| `avg_rtt` | Mean round-trip time (ms) — primary signal |
| `min_rtt` / `max_rtt` | Best and worst packet in the burst |
| `jitter` | `max_rtt − min_rtt` — variability within a single burst |
| `packet_loss_pct` | `(sent − received) / sent × 100` |
| `rolling_avg_rtt` | 5-measurement rolling mean per probe (local trend) |
| `rolling_std_rtt` | 5-measurement rolling std per probe (local spread) |
| `z_score_rtt` | Per-probe global z-score of avg_rtt |
| `hour_of_day` | 0–23 — captures diurnal patterns |
| `day_of_week` | 0 (Mon) – 6 (Sun) — captures weekly patterns |

---

## Anomaly Detection Methods

Five detectors are applied, each capturing a different aspect of anomalous behaviour. A final **ensemble** label flags a measurement as anomalous when at least two detectors agree.

### 1. Z-Score (Statistical Baseline)
Computes the global per-probe z-score of `avg_rtt`. A measurement is anomalous if its z-score exceeds 3.0 standard deviations from that probe's mean.

- **Strength:** Simple, interpretable, fast
- **Weakness:** Assumes normally distributed RTT; sensitive to long-tailed distributions
- **Threshold:** |z| > 3.0

### 2. IQR (Robust Statistical)
Flags measurements outside the interquartile fence `[Q1 − 1.5×IQR, Q3 + 1.5×IQR]`, computed per probe.

- **Strength:** Resistant to extreme outliers that inflate mean/std; works well on skewed RTT distributions
- **Weakness:** Univariate; misses multivariate anomalies where each individual metric looks normal

### 3. Isolation Forest (Ensemble ML)
An ensemble of random binary trees that isolates anomalies by recursively partitioning the feature space. Anomalous points require fewer splits to isolate.

- **Features used:** `avg_rtt`, `jitter`, `packet_loss_pct`, `rolling_std_rtt`, `hour_of_day`
- **Strength:** Multivariate; makes no distributional assumption; efficient on large datasets
- **Weakness:** `contamination` parameter must be estimated; less interpretable
- **Contamination:** 0.05 (5% of data expected to be anomalous)

### 4. Local Outlier Factor (Density-Based ML)
Compares the local density of each point to the density of its k nearest neighbours. Points in significantly lower-density regions than their neighbours are flagged.

- **Features used:** Same as Isolation Forest
- **Strength:** Detects local anomalies invisible to global methods — a probe whose RTT suddenly doubles relative to its peers is caught even if the absolute value seems normal
- **Weakness:** Computationally expensive; sensitive to choice of k
- **Neighbours:** k = 20

### 5. Rolling Z-Score (Time-Series Contextual)
Computes a z-score relative to a sliding window of the 10 most recent measurements per probe, rather than the global mean.

- **Strength:** Detects sudden contextual spikes — a measurement that is normal globally but deviates sharply from the probe's recent baseline (e.g. a latency spike on an otherwise stable connection)
- **Weakness:** Requires sufficient history per probe; cannot flag anomalies at the start of a series
- **Window:** 10 measurements, threshold |z| > 3.0

### Ensemble Voting
A measurement is labelled **ensemble anomaly** if at least **2 of 5** detectors flag it. This reduces false positives from any single method while preserving sensitivity.

---

## Key Findings

### Diurnal RTT Patterns
RTT measurements to `k.root-servers.net` show consistent diurnal cycles: latency is lowest during European off-peak hours (02:00–06:00 UTC) and peaks during business hours when core Internet backbone links are under heavier load. This pattern is most pronounced for probes in Asia-Pacific, which traverse intercontinental links.

### Geographic Baseline Separation
Probes naturally cluster by continent in the RTT distribution. European probes measure ~5–15 ms to the Amsterdam-hosted root server, North American probes measure ~70–120 ms, and Asia-Pacific probes measure ~150–250 ms. Anomaly detectors are applied **per-probe** to avoid flagging geographically normal high-latency probes as anomalous.

### Packet Loss Events
100% packet loss events (where RIPE Atlas reports `avg = −1`) cluster around specific probe IDs and short time windows, consistent with transient link failures or firewall rule changes rather than sustained outages. These are correctly preserved as anomalies by the IQR and Rolling Z-Score detectors via the `packet_loss_pct` feature.

### Detector Agreement
The Isolation Forest and LOF detectors show the highest agreement (~70–80% overlap in flagged rows), confirming that multivariate density-based methods identify a similar underlying anomaly population. The Rolling Z-Score catches a distinct class of short-duration contextual spikes that the global methods miss — typically 1–3 consecutive measurements before the probe returns to baseline.

### Ensemble False-Positive Reduction
Requiring ≥ 2 votes reduces the flagged anomaly rate from ~5–8% (single detectors) to ~2–3% (ensemble), while retaining the highest-confidence events identified by multiple independent methods.

---

## Project Structure

```
ripe/
├── data_collection.py      # RIPE Atlas API fetch (AtlasResultsRequest + probe filter)
├── data_preprocessing.py   # Data cleaning and feature engineering
├── anomaly_detection.py    # Five detectors + ensemble
├── visualization.py        # Six matplotlib plots
├── main.py                 # CLI pipeline runner
├── requirements.txt        # Python dependencies
├── run.txt                 # Quick-reference commands
└── output/                 # Generated files (created on first run)
    ├── measurement_data.csv        # Raw collected data
    ├── measurement_features.csv    # Engineered features
    ├── measurement_anomalies.csv   # Annotated results with anomaly labels
    ├── plot_rtt_timeseries.png     # RTT over time with anomaly markers
    ├── plot_detector_comparison.png # Anomaly count per detector (bar chart)
    ├── plot_rtt_distribution.png   # Per-probe RTT box plots
    ├── plot_score_scatter.png      # Isolation Forest vs LOF score scatter
    ├── plot_heatmap_rtt.png        # Mean RTT heatmap (hour × day of week)
    └── plot_packet_loss.png        # Hourly mean packet loss
```

---

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies:** `ripe.atlas.cousteau`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`

**Optional:** Set your RIPE Atlas API key as an environment variable (public measurements work without one):
```bash
export RIPE_ATLAS_API_KEY=your-key-here   # Linux/macOS
set RIPE_ATLAS_API_KEY=your-key-here      # Windows
```

---

## Usage

All commands are run from the project root (`Adv_Data_Mining/`).

**Step 1 — Fetch data from RIPE Atlas (~2–3 minutes):**
```bash
python ripe\main.py --collect --start 2024-01-01 --end 2024-01-14
```

**Step 2 — Re-run analysis on existing data (no network call):**
```bash
python ripe\main.py
```

**Tune anomaly sensitivity:**
```bash
# Stricter: fewer but higher-confidence anomalies
python ripe\main.py --contamination 0.03 --min-votes 3

# More sensitive: catches weaker signals
python ripe\main.py --contamination 0.08 --min-votes 2
```

**Custom output directory:**
```bash
python ripe\main.py --output-dir path\to\results
```

All output CSVs and PNG plots are written to `ripe/output/` by default.

---

## References

- RIPE NCC. *RIPE Atlas: A Global Internet Measurement Network*. https://atlas.ripe.net
- Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation Forest. *ICDM 2008*.
- Breunig, M. M., Kriegel, H.-P., Ng, R. T., & Sander, J. (2000). LOF: Identifying Density-Based Local Outliers. *SIGMOD 2000*.
- Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly Detection: A Survey. *ACM Computing Surveys*.
- Han, J., Kamber, M., & Pei, J. (2011). *Data Mining: Concepts and Techniques* (3rd ed.). Morgan Kaufmann.
