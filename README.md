# 🚨 GridLock IQ
### Predictive Parking Intelligence & Enforcement Optimization System

> **Flipkart GridLock 2.0 — Problem Statement: Poor Visibility on Parking-Induced Congestion**  
> Bengaluru Traffic Police · H3 Spatial Indexing (Resolution 7) · LightGBM Delta Model · Streamlit Dashboard

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [What GridLock IQ Does](#what-gridlock-iq-does)
3. [Key Results](#key-results)
4. [System Architecture](#system-architecture)
5. [Tech Stack](#tech-stack)
6. [Project Structure](#project-structure)
7. [Dataset](#dataset)
8. [Installation](#installation)
9. [Running the Pipeline](#running-the-pipeline)
10. [Running the Dashboard](#running-the-dashboard)
11. [IEU Formula](#ieu-formula)
12. [Model Details](#model-details)
13. [Dashboard Pages](#dashboard-pages)
14. [Audit Trail](#audit-trail)
15. [Honest Limitations](#honest-limitations)

---

## Problem Statement

On-street illegal parking and spillover parking near commercial areas, metro stations, and event venues choke carriageways and intersections across Bengaluru. Enforcement today is:

- **Patrol-based and reactive** — officers respond after congestion forms, not before
- **No spatial heatmap** of parking violations vs. congestion impact
- **Difficult to prioritize** enforcement zones across 178 H3 cells citywide

**How can AI-driven parking intelligence detect illegal parking hotspots, quantify their congestion impact, and enable targeted, proactive enforcement?**

---

## What GridLock IQ Does

GridLock IQ transforms raw Bengaluru traffic violation citation data into a predictive enforcement intelligence system with five operational layers:

| Layer | What It Answers |
|---|---|
| **Command Overview** | What is the city-wide risk level right now? |
| **Hotspot Intelligence Map** | Where are the active congestion zones? |
| **Prediction Engine** | Where will congestion spike in 1–4 hours? |
| **Enforcement Optimizer** | Which zones need units, and how many? |
| **What-If Simulator** | How does AI dispatch compare to historical patrol patterns? |

---

## Key Results

| Metric | Value |
|---|---|
| Critical Zone Recall (T+1h) | **87.1%** — catches 87 of every 100 critical events |
| Precision | **71.1%** — 29% false alert rate |
| False Positive Rate | **0.47%** — near-zero false alarms on safe zones |
| F2 Score | **0.834** |
| Active Cell R² | **0.724** |
| Active Cell MAE | **8.01 IEU points** |
| Beats Naive Persistence | **+1.65 MAE points** on active zones |
| Leakage Audits Passed | **5 / 5** — zero data contamination confirmed |

> **Critical zone** = IEU ≥ 75 on a 0–100 scale. **Active cell** = H3 cell with at least one violation in the hour window.

---

## System Architecture

```
Raw Violation Data (CSV)
        │
        ▼
┌───────────────────┐
│  data_pipeline.py │  → Clean, type-map, H3 assign, timestamp parse,
│                   │     duration compute, vehicle mass mapping
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  spatial_engine.py│  → Compute IEU per violation record
│                   │     IEU_raw = J × (Σ mass, 3h window)^0.8
│                   │     IEU = 100 × (1 − e^(−raw/k))
│                   │     HDBSCAN corridor clustering
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  forecast_engine  │  → Build hourly spatio-temporal feature matrix
│  .py              │     Lag features, rolling stats, spatial neighbor
│                   │     features, cyclical time encoding
│                   │     Train LightGBM Delta Model (Huber loss)
│                   │     Target: IEU(t+h) − IEU(t)
│                   │     Reconstruct at inference: pred + IEU(t)
└───────────────────┘
        │
        ▼
┌───────────────────────────┐
│  forecast_results.parquet │  → Pre-computed T+1h, T+2h, T+4h forecasts
│                           │     for all 178 cells across entire date range
└───────────────────────────┘
        │
        ▼
┌───────────────────┐
│  Streamlit App    │  → 5-page operational dashboard
│  (app/main.py)    │     Leaflet maps, Plotly charts, Dispatch tables
└───────────────────┘
```

---

## Tech Stack

| Category | Tools |
|---|---|
| **Data Processing** | Python 3.10, Pandas, NumPy |
| **Spatial Indexing** | H3-py (Uber H3, Resolution **7**, ~1.2km edge) |
| **Clustering** | HDBSCAN (corridor hotspot grouping) |
| **ML Model** | LightGBM (Huber loss, delta target) |
| **Dashboard** | Streamlit |
| **Maps** | Leaflet.js + Carto Dark tiles (free, no API key) |
| **Charting** | Plotly Graph Objects |
| **Data Storage** | Parquet (Apache Arrow / PyArrow) |

> **Zero external data constraint strictly upheld.** All features are derived exclusively from the provided violation dataset. No external APIs, map data feeds, or third-party datasets are used.

---

## Project Structure

```
gridlock-iq/
│
├── data/
│   ├── raw/                              # Original HackerEarth violation CSV
│   └── processed/
│       ├── cleaned_violations.parquet    # Cleaned records with IEU, H3 cells
│       ├── forecast_results.parquet      # Model predictions (T+1h, T+2h, T+4h)
│       ├── h3_clustered_zones.parquet    # HDBSCAN corridor clusters
│       ├── repeat_offenders.parquet      # Vehicle repeat-risk register
│       └── cell_poi_tags.json            # H3 cell → junction label (from dataset)
│
├── src/
│   ├── data_pipeline.py                  # Raw cleaning, duration parse, mass map
│   ├── spatial_engine.py                 # H3 indexing, IEU formula, HDBSCAN
│   ├── forecast_engine.py                # Feature engineering + LightGBM training
│   ├── patrol_optimizer.py               # Greedy dispatch & What-If simulation
│   └── repeat_offender.py                # Vehicle repeat-risk scoring
│
├── models/
│   ├── lgbm_t1.pkl                       # T+1h LightGBM regressor
│   ├── lgbm_t2.pkl                       # T+2h LightGBM regressor
│   └── lgbm_t4.pkl                       # T+4h LightGBM regressor
│
├── api/
│   └── main.py                           # FastAPI REST server (optional)
│
├── app/
│   ├── main.py                           # Streamlit multipage router
│   ├── data_state.py                     # Session state & data loader
│   ├── components/
│   │   ├── alert_utils.py                # Alert card rendering
│   │   └── chart_utils.py                # Plotly chart builders
│   └── pages/
│       ├── command_Overview.py
│       ├── Hotspot_Map.py
│       ├── Prediction_Engine.py
│       ├── Enforcement_Optimizer.py
│       └── Whatif_Simulator.py
│
├── audits/
│   ├── leakage_audit.py                  # ✅ No-leakage proof (5 checks)
│   ├── threshold_analysis.py             # ✅ F2-optimal threshold = 65 IEU
│   ├── model_accuracy.py                 # ✅ Stratified accuracy report
│   ├── retrain.py                        # Utility: retrain all LightGBM models
│   └── build_poi_tags.py                 # Utility: build cell_poi_tags.json
│
├── config.py                             # Paths, mass map, AOI/IEU constants
├── requirements.txt                      # Package dependencies
├── .gitignore
└── README.md
```

---

## Dataset

**Source:** Bengaluru Traffic Violation citation dataset provided by HackerEarth (Flipkart GridLock 2.0).

**Important context:** Each row is a violation citation issued by a traffic officer — not a ground-truth sensor measurement. This means:
- Coverage depends on patrol routes and officer presence
- ~73% of H3 cell-hours have zero violations (zero-inflation)
- Citation density is a behavioral proxy for congestion, not a direct measurement

**Key columns used:**

| Column | Description |
|---|---|
| `latitude`, `longitude` | Violation GPS coordinates |
| `created_datetime` | Citation timestamp |
| `modified_datetime` | Used for obstruction duration (fallback) |
| `clean_vehicle_type` | Standardized vehicle category |
| `junction_name` | Nearest landmark / junction (used for POI labels) |
| `police_station` | BTP station jurisdiction |
| `junction_flag` | Binary: major intersection (2× IEU weight) |

**Scope:** 5 months of citation data · 178 unique H3 cells (Resolution 7, ~1.2km edge length)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/charansai-1411/GridLock-IQ.git
cd GridLock-IQ

# Create virtual environment
python -m venv venv

# Activate — Windows
venv\Scripts\activate
# Activate — Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Pipeline

Run these scripts in order. Each saves output to `data/processed/` for the next step.

### Step 1 — Data Cleaning

```bash
python src/data_pipeline.py
```

- Parses timestamps (`created_datetime`, `modified_datetime`)
- Computes obstruction duration in minutes (median imputation for anomalous/missing)
- Maps `clean_vehicle_type` to vehicle mass weights
- Assigns each violation to an H3 cell (Resolution 7) using lat/lon
- Output: `data/processed/cleaned_violations.parquet`

### Step 2 — Spatial Engine (IEU + Clustering)

```bash
python src/spatial_engine.py
```

- Computes 3-hour rolling mass sum per H3 cell
- Applies IEU formula with smooth-squash normalization
- Runs HDBSCAN on high-risk cell centroids (haversine metric)
- Output: `data/processed/cleaned_violations.parquet` (enriched), `h3_clustered_zones.parquet`

### Step 3 — Train Models

```bash
python audits/retrain.py
```

- Builds hourly spatio-temporal feature matrix (all 178 cells × all hours)
- Trains three LightGBM models (T+1h, T+2h, T+4h) with delta targets
- Saves `models/lgbm_t1.pkl`, `lgbm_t2.pkl`, `lgbm_t4.pkl`
- Saves `data/processed/forecast_results.parquet` (pre-computed predictions)

### Step 4 — Repeat Offender Register

```bash
python src/repeat_offender.py
```

- Output: `data/processed/repeat_offenders.parquet`

### Step 5 — Build POI Labels

```bash
python audits/build_poi_tags.py
```

- Extracts landmark names from `junction_name`, `location`, `police_station` columns of the raw CSV (no external geocoding)
- Output: `data/processed/cell_poi_tags.json`

---

## Running the Dashboard

```bash
streamlit run app/main.py
```

Opens at `http://localhost:8501`

**Default demo timestamp:** `2024-03-12 08:30 AM IST` — a peak-hour window with 7 active critical zones and clear AI vs. historical dispatch differentiation.

---

## IEU Formula

**IEU (Instantaneous Enforcement Urgency)** is a physics-motivated congestion proxy computed entirely from the provided dataset — zero external data sources.

### Formula

```
IEU_raw(cell, t) = J × ( Σ Mᵢ for all violations in 3h rolling window )^0.8

k = p90(IEU_raw) / ln(4)        ← anchors p90 → IEU = 75

IEU(cell, t) = 100 × (1 − exp(−IEU_raw / k))
```

### Vehicle Mass Mapping

| Vehicle Type | Mass Weight |
|---|---|
| TANKER | 5.0 |
| BUS / PRIVATE BUS / SCHOOL VEHICLE | 4.5 |
| HGV / LORRY | 4.0 |
| LGV / MINI LORRY / TRACTOR | 3.5 |
| CAR / JEEP / MAXI-CAB / TEMPO | 3.0 |
| TAXI / VAN | 2.5 |
| GOODS AUTO / PASSENGER AUTO | 2.0 |
| MOTOR CYCLE | 1.5 |
| SCOOTER / MOPED | 1.0 |

### Junction Multiplier

| Junction Type | Multiplier |
|---|---|
| Named major junction / intersection | 2.0× |
| Mid-block / non-junction | 1.0× |

### Design Choices (Ablation Validated)

- ✅ **3-hour rolling window** — autocorrelation 0.884 vs 0.51 for 15-min window
- ✅ **α = 0.8** saturation power — sublinear (first vehicles block most; adding to an already blocked road yields less marginal harm)
- ✅ **Junction multiplier** — spillover at intersections compounds
- ✅ **Smooth-squash normalization** — eliminates the spike at exactly 100 that linear P90 clipping produces
- ✅ **Duration weighting retained** — obstruction duration captures how long a vehicle blocks the road

### ⚠️ Honest Note on Ground Truth

This dataset is a **pure citation-issuance log** — it records when a traffic officer wrote a ticket, not:
- When the vehicle was actually removed from the road
- Whether a traffic jam actually occurred at that location
- What road speed or vehicle throughput was at that moment

The `closed_datetime` column (which would indicate resolution) is **null for 100% of records**. As a result, every metric in this project — IEU score, risk tier, congestion prediction — is a **physics-motivated proxy** derived from the citation data itself.

**The model's validation target (`IEU at t+1`) is evaluated against our own derived formula, not against any external sensor or ground-truth measurement.**

This is not a limitation of the approach — it is the correct way to operate under the **zero-external-dataset constraint** of this problem statement. It does mean that operational deployment would require calibration against real traffic speed data (e.g., ATMS sensors or GPS probe data) before claims of accuracy can be made against ground truth.

---

## Model Details


### Architecture: LightGBM Delta Model

**Key design decision:** Instead of predicting absolute IEU(t+1), the model predicts the *change*: `delta = IEU(t+h) − IEU(t)`. A model predicting `delta = 0` everywhere exactly matches the naive persistence baseline — so any MAE improvement is genuine signal above persistence, not inflation from the 73% zero rows.

**Reconstruction at inference:**
```python
predicted_IEU = np.clip(model.predict(X) + current_IEU, 0, 100)
```

### Training Configuration

```python
model = LGBMRegressor(
    objective='huber',
    alpha=5.0,           # L1/L2 boundary at 5 IEU points — robust to rare spikes
    n_estimators=400,
    learning_rate=0.04,
    max_depth=7,
    num_leaves=63,
    min_child_samples=15,
    reg_alpha=0.05,
    reg_lambda=0.1,
    random_state=42,
)
```

**Split:** Chronological 80/20 — no shuffling. Train on first 80% of hours, validate on last 20%.

### Feature Set (19 features)

| Feature | Description |
|---|---|
| `AOI` | Current IEU — baseline for delta prediction |
| `AOI_lag_1h`, `AOI_lag_2h`, `AOI_lag_3h` | IEU 1/2/3 hours ago (per-cell shifted) |
| `AOI_roll_mean_7d`, `AOI_roll_max_7d`, `AOI_roll_std_7d` | 7-day rolling stats |
| `hour_sin`, `hour_cos` | Cyclical hour-of-day encoding |
| `day_sin`, `day_cos` | Cyclical day-of-week encoding |
| `historical_density` | Cell's long-run avg violations/hour at this hour |
| `neighbor_mean_lag1`, `neighbor_max_lag1` | Spatial neighbor mean/max IEU (k-ring-1) |
| `neighbor_any_critical` | Binary: any H3 neighbor currently critical |
| `latitude`, `longitude` | Cell centroid coordinates |
| `cell_vehicle_mass` | Dominant vehicle mass for the cell |
| `junction_flag` | Binary: major intersection |
| `junction_type_ord` | metro=4 / market+mall=3 / junction=2 / area=1 |

### Stratified Evaluation (T+1h)

| Segment | MAE | R² |
|---|---|---|
| All rows (incl. 73% zero cells) | ~4.66 | ~0.862 |
| **Active cells only (IEU > 0)** | **8.01** | **0.724** |
| Critical cells only (IEU ≥ 75) | ~9.08 | — |
| Naive persistence baseline | ~6.31 | — |

> All-rows R²=0.862 is inflated by the 73% zero-zero pairs. **Active-cell R²=0.724 is the operationally honest metric.**

### Decision Threshold

F2-optimized threshold: **IEU = 65** (recall weighted 2× over precision — missing a critical zone is worse than a false dispatch).

| Threshold | Recall | Precision | F2 |
|---|---|---|---|
| **65 (selected)** | **87.1%** | **71.1%** | **0.834** |
| 75 (naive) | 65.0% | 75.0% | 0.668 |

---

## Dashboard Pages

### 1. Command Overview
City-wide operational status. KPI cards (active critical zones, predicted peak window, city risk level, units required). Hourly IEU trend with T+1h–T+4h prediction shading. Active dispatch alert cards with landmark names and severity badges. Shift commander brief with download.

### 2. Hotspot Intelligence Map
2D Leaflet map with flat H3 hexagon overlays colored by current IEU tier. Hover tooltip showing landmark name, risk score, primary violator, jurisdiction. Zone inspect panel with T+1/T+2/T+4 forecast bars.

### 3. Prediction Engine
2D prediction map showing **predicted** IEU for selected horizon (T+1h / T+2h / T+4h). Top predicted zones table. Chronic / Episodic / Random zone classification based on recurrence patterns.

### 4. Enforcement Optimizer
Dispatch orders table ranked by predicted IEU — junction names, units assigned, severity, action label. Jurisdiction coverage chart. Patrol timeline grid (T+1h → T+4h coverage). Repeat offender watchlist (search + ticket counts). Shift commander brief with download.

### 5. What-If Simulator
Officer slider (5–60). Scenario A (historical density allocation) vs Scenario B (AI submodular greedy allocation) with map toggle. Zone coverage comparison table. Unit efficiency scatter plot (IEU covered vs units). Live recalculation on every slider change.

---

## Audit Trail

Five independent data integrity checks in `audits/leakage_audit.py`:

| Audit | Check | Result |
|---|---|---|
| 1 — Timestamp alignment | `AOI` feature = IEU(t), never IEU(t+1) | ✅ CLEAN |
| 2 — Manual row verification | 5 random rows hand-computed | ✅ CLEAN (5/5) |
| 3 — Feature ablation | R² drop when AOI removed: −0.004 | ✅ STABLE (not leaking) |
| 4 — Train/val split | Hour overlap between train and val sets | ✅ CLEAN (0 overlap) |
| 5 — Cross-cell lag contamination | `groupby('h3_cell').shift()` confirmed | ✅ CLEAN (178/178 cells) |

```python
# All lag features use per-cell groupby shift — zero cross-cell contamination
grid_df['AOI_lag_1h'] = grid_df.groupby('h3_cell')['AOI'].shift(1).fillna(0.0)
grid_df['AOI_lag_2h'] = grid_df.groupby('h3_cell')['AOI'].shift(2).fillna(0.0)
grid_df['AOI_lag_3h'] = grid_df.groupby('h3_cell')['AOI'].shift(3).fillna(0.0)

# Targets also use per-cell groupby shift
grid_df['target_t1'] = grid_df.groupby('h3_cell')['AOI'].shift(-1)
grid_df['target_t2'] = grid_df.groupby('h3_cell')['AOI'].shift(-2)
grid_df['target_t4'] = grid_df.groupby('h3_cell')['AOI'].shift(-4)
```

Run any audit:
```bash
python audits/leakage_audit.py
python audits/threshold_analysis.py
python audits/model_accuracy.py
```

---

## Honest Limitations

| Limitation | Impact |
|---|---|
| IEU is a derived formula, not ground-truth sensor data | Accuracy measured against our own proxy, not real road speed |
| 5-month dataset window | Limited seasonal learning; full year would improve T+4h significantly |
| 73% sparse cells (zero violations) | Active-cell R²=0.724 is the meaningful metric, not all-rows R²=0.862 |
| Citation behavior ≠ congestion directly | Officer patrol routes and enforcement quotas add noise |
| Single city (Bengaluru) | Generalizability to other cities not validated |

---

## Acknowledgments

- **Dataset:** Flipkart GridLock 2.0 / HackerEarth — Bengaluru Traffic Violation data
- **Spatial Indexing:** Uber H3 library (Resolution 7)
- **Map Tiles:** Leaflet.js + Carto Dark (free, no API key required)
- **Institution:** Chaitanya Bharathi Institute of Technology (CBIT), Hyderabad

---

## License

Built for Flipkart GridLock 2.0 hackathon. Dataset used under HackerEarth competition terms.

---

*Built by Y. Charan Sai — B.E. Artificial Intelligence & Data Science, CBIT Hyderabad*
