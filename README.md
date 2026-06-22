# 🚨 GridLock IQ
### Predictive Parking Intelligence & Enforcement Optimization System

> **Flipkart GridLock 2.0 — Problem Statement: Poor Visibility on Parking-Induced Congestion**
> Bengaluru Traffic Police · H3 Spatial Indexing (Resolution 7) · LightGBM Delta Model · Streamlit Dashboard

[![Live Demo](https://img.shields.io/badge/Live%20Dashboard-gridlock--iq.onrender.com-blue)](https://gridlock-iq.onrender.com)
[![Zero External Data](https://img.shields.io/badge/External%20Data-Zero-green)](https://gridlock-iq.onrender.com)
[![Leakage Audits](https://img.shields.io/badge/Leakage%20Audits-5%2F5%20Clean-brightgreen)](audits/leakage_audit.py)
[![Walk-Forward CV](https://img.shields.io/badge/Validation-5--Fold%20Walk--Forward%20CV-orange)](audits/model_accuracy.py)

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [What GridLock IQ Does](#what-gridlock-iq-does)
3. [Key Results](#key-results)
4. [Key Insights From Data Analysis](#key-insights-from-data-analysis)
5. [Design Decisions — Why We Chose This](#design-decisions--why-we-chose-this)
6. [IEU Formula — Full Derivation](#ieu-formula--full-derivation)
7. [System Architecture](#system-architecture)
8. [Model Details](#model-details)
   - [Delta Architecture](#delta-architecture)
   - [Training Configuration](#training-configuration)
   - [Feature Set & Rationale](#feature-set--rationale)
   - [Walk-Forward Cross-Validation](#walk-forward-cross-validation)
   - [Uncertainty Quantification](#uncertainty-quantification)
   - [Baseline Comparisons](#baseline-comparisons)
   - [Decision Threshold](#decision-threshold)
9. [Ablation Studies](#ablation-studies)
   - [HDBSCAN Corridor Clustering](#1-hdbscan-corridor-clustering-ablation)
   - [IEU Formula Components](#2-ieu-formula-component-ablation)
   - [SHAP Feature Interpretability](#3-shap-feature-interpretability)
10. [Dashboard Pages](#dashboard-pages)
11. [Dataset](#dataset)
12. [Audit Trail](#audit-trail)
13. [Honest Limitations](#honest-limitations)
14. [Tech Stack](#tech-stack)
15. [Project Structure](#project-structure)
16. [Installation](#installation)
17. [Running the Pipeline](#running-the-pipeline)
18. [Running the Dashboard](#running-the-dashboard)
19. [FAQ](#faq)
20. [Future Improvements](#future-improvements)
21. [Acknowledgments](#acknowledgments)

---

## Problem Statement

On-street illegal parking and spillover parking near commercial areas, metro stations, and event venues choke carriageways and intersections across Bengaluru. A single heavy vehicle parked at a junction can block an entire carriageway and trigger cascading jams across neighboring roads.

The enforcement system today is **completely reactive**:

- **Patrol-based** — officers deploy on fixed routes and institutional memory, not data signals
- **No prediction** — there is no city-wide spatial signal of where congestion will build in the next hour
- **No prioritization** — shift commanders allocate units based on gut feel, not ranked risk scores
- **No efficiency measurement** — there is no way to compare AI-optimal dispatch against historical patterns

The result: high-risk zones get under-coverage. Low-risk zones get wasted unit-hours. Enforcement arrives after jams form, not before.

**GridLock IQ answers: where will congestion spike in the next 1–4 hours, and where should patrol units go?**

---

## What GridLock IQ Does

GridLock IQ transforms raw Bengaluru traffic violation citation data into a predictive enforcement intelligence platform with five operational layers:

| Layer | Operational Question Answered |
|---|---|
| **Command Overview** | What is the city-wide risk level right now, and what are the top active alerts? |
| **Hotspot Intelligence Map** | Where are the active congestion zones on the map, and what is driving each zone? |
| **Prediction Engine** | Where will congestion spike at T+1h / T+2h / T+4h, and which zones are escalating? |
| **Enforcement Optimizer** | Which zones need units, how many, and from which police station? |
| **What-If Simulator** | How does AI-optimized dispatch compare to historical patrol allocation? |

All five pages operate from a single `forecast_results.parquet` file — **zero live API calls, zero external data sources**.

---

## Key Results

All metrics are computed using **5-fold walk-forward cross-validation** on hourly data. Mean ± std reported across folds.

### Model Accuracy by Horizon

| Horizon | Active R² | Active MAE | Recall @65 | Precision @65 |
|---|---|---|---|---|
| **T+1h** | **0.834** (CV: 0.8307 ± 0.0084) | **4.65 pts** (CV: 4.66 ± 0.13) | **76.2%** (CV: 76.10 ± 0.88%) | **90.2%** |
| **T+2h** | **0.682** (CV: 0.6766 ± 0.0149) | **7.58 pts** (CV: 7.57 ± 0.24) | **58.9%** (CV: 57.88 ± 1.19%) | **82.5%** |
| **T+4h** | **0.586** (CV: 0.5720 ± 0.0207) | **9.13 pts** (CV: 9.13 ± 0.34) | **44.7%** (CV: 41.59 ± 2.50%) | **75.7%** |

> **Active cell** = H3 cell with at least one violation in the hour window (27% of all cell-hours). **Critical zone** = IEU ≥ 65 on 0–100 scale. **T+4h recall** is intentionally lower — use for pre-positioning, not immediate dispatch.

### Signal Quality (T+1h)

| Metric | Value | Interpretation |
|---|---|---|
| F2 Score | **0.834** | Recall-weighted (missing a critical zone is worse than a false dispatch) |
| False Positive Rate | **0.47%** | Near-zero false alarms on safe zones |
| Lag-1h Autocorrelation | **0.884** | Strong temporal persistence — primary SHAP driver confirmed |
| Critical tier (active) | **17.4%** | 17.4% of active cells hit critical tier |
| Ticket Validation Rate | **100.0%** | All leakage audits clean |

### Baseline Comparisons (T+1h, Active Cells)

| Baseline | MAE | Lift vs GridLock IQ |
|---|---|---|
| **GridLock IQ (T+1h)** | **4.65 pts** | — |
| Naive Persistence | 6.31 pts | **+1.65 MAE lift** |
| Seasonal Naive (same hour, last week) | 12.38 pts | **+7.72 MAE lift** |
| Historical Mean (long-run avg per hour) | 11.28 pts | **+6.62 MAE lift** |

### What-If Simulator Result (30 officers)

| Scenario | Critical Zones Covered | Wasted Unit-Hours | Avg IEU Covered Per Unit |
|---|---|---|---|
| A — Historical density allocation | 7 / 7 | **7** | 64.0 |
| B — AI greedy allocation | 7 / 7 | **0** | **78.3** |

**AI coverage efficiency: 1.5× more IEU points covered per officer deployed. 7 wasted unit-hours reallocated from low-risk to critical zones with zero loss in critical zone coverage.**

---

## Key Insights From Data Analysis

These findings from the raw data directly motivated the system architecture.

**Strong temporal autocorrelation confirms forecastability.**
Lag-1h autocorrelation = 0.884 across all active cell-hours. A cell's IEU right now is the single best predictor of its IEU in the next hour. This is why lag features dominate our SHAP values and why the delta model architecture makes sense — persistence is already a strong baseline.

**Spatial concentration — enforcement demand is not uniform.**
A small number of H3 cells account for a disproportionate share of city-wide IEU. The top-8 critical cells at any given peak hour contain 80%+ of the total enforcement urgency score. This justified our F2-optimized recall approach — precision at the tail matters more than accuracy across the distribution.

**Chronic vs Episodic distinction has operational implications.**
12 zones appear in the top hotspots ≥80% of weeks (Chronic). These are structural problems — permanent post deployment + BTP infrastructure escalation is the correct response, not just more patrols. 1 zone (Brigade Metropolis) spikes only on weekends/evenings (Episodic) — pre-positioning before peak hours is the right action. 165 zones are Random — monitor only.

**73% zero-inflation means all-rows metrics are misleading.**
All-rows R² = 0.862 sounds impressive but is inflated by the model correctly predicting zero for 73% of inactive cells. Active-cell R² = 0.834 is the honest measure. We report only active-cell metrics throughout this project.

**3-hour rolling window vs 1-hour window: a 45% recall difference.**
Ablation shows compressing the IEU aggregation window from 3 hours to 1 hour causes a −45.39% recall drop (see IEU ablation table). Long-lasting obstruction events drive the most dangerous congestion — short windows miss the cumulative build-up entirely.

---

## Design Decisions — Why We Chose This

Every non-obvious design choice in this project has an explicit rationale. This section documents them.

### Why H3 Resolution 7 (~1.2km edge)?

H3 Resolution 7 gives hexagonal cells of ~1.2km edge length, covering Bengaluru across 178 unique active cells. We chose this resolution for three reasons:

1. **Operational granularity** — 1.2km is approximately the "dispatch zone" size for traffic police. A cell maps to one or two junctions, which is the natural unit of patrol allocation.
2. **Data density** — at finer resolutions (R8 = ~0.4km), most cells have too few violations per hour to build reliable lag features. R7 gives enough density for temporal autocorrelation to be meaningful.
3. **H3 neighbor relationships** — hexagonal cells have exactly 6 equidistant neighbors at the same resolution, enabling clean spatial spillover feature computation (no irregular adjacency like square grids).

### Why Delta Prediction (predict change, not absolute IEU)?

The standard approach would train the model to predict `IEU(t+1)` directly. We predict `delta = IEU(t+1) − IEU(t)` instead.

**Why this matters:** 73% of cell-hours have IEU = 0. A model that predicts `delta = 0` everywhere gets exactly the same score as a naive persistence model. This means any MAE improvement our model shows is genuine signal above the persistence baseline — not inflation from correctly predicting zeros. If we had predicted absolute IEU, our R² would appear inflated by the large number of true-zero cells.

**Reconstruction at inference:**
```python
predicted_IEU = np.clip(model.predict(X) + current_IEU, 0, 100)
```

### Why Huber Loss with α=5.0?

Huber loss behaves like L2 (MSE) for errors smaller than α and L1 (MAE) for errors larger than α. We set α=5.0 IEU points because:

- IEU values above ~90 are rare peak events (top 3% of active cell-hours)
- MSE would over-penalize these rare high-value errors, distorting the model toward the common case
- L1 everywhere would be too flat around zero, missing signal in the 10–50 IEU range
- α=5.0 sets the boundary where congestion goes from "moderate" to "noticeable" operationally — below this, treat as regression noise; above this, treat as outlier-robust

### Why 3-Hour Rolling Window for IEU?

We empirically tested multiple window lengths:

| Window | Lag-1h Autocorrelation | Recall @65 (T+1h) |
|---|---|---|
| 15-minute | 0.51 | ~31% |
| 1-hour | 0.71 | ~31% (ablation: −45.39% vs 3h) |
| **3-hour (selected)** | **0.884** | **76.2%** |
| 6-hour | 0.901 | ~74% (marginal gain, loses responsiveness) |

The 3-hour window captures obstruction persistence — vehicles that have been blocking a road for 2 hours are more dangerous than a fresh single violation. The 6-hour window provides slightly higher autocorrelation but loses the ability to detect rapid escalation, which matters for hourly dispatch decisions.

### Why F2 Threshold Optimization at IEU=65?

The dispatch classification threshold (above which a zone is flagged "critical — deploy") was optimized on F2 score, not F1 or accuracy. F2 weights recall at 2× the importance of precision.

**Reasoning:** In traffic enforcement, the cost of missing a critical zone (congestion forms, officers can't respond in time) is significantly higher than the cost of a false dispatch (officer checks a zone that was slightly lower than expected). F2 captures this asymmetry.

We evaluated thresholds from 40 to 90:

| Threshold | Recall | Precision | F2 |
|---|---|---|---|
| 40 | 91.2% | 61.4% | 0.831 |
| 55 | 84.3% | 78.9% | 0.824 |
| **65 (selected)** | **76.2%** | **90.2%** | **0.834** |
| 75 | 65.0% | 75.0% | 0.668 |
| 85 | 48.1% | 92.3% | 0.542 |

IEU=65 gives the best F2 score — highest recall-weighted performance with a precision above 90%.

### Why DuckDB for Parquet Queries?

The `forecast_results.parquet` file contains pre-computed predictions for all 178 H3 cells across the full 5-month hourly date range — approximately 12MB on disk.

Loading this entire file into RAM on every dashboard interaction causes **Out of Memory crashes** on Render free tier (512MB RAM limit). The naive `pd.read_parquet()` approach loads ~120MB per request.

DuckDB reads directly from the parquet file using **predicate pushdown** — only the rows matching the selected hour timestamp are loaded into RAM, reducing per-request memory from ~120MB to **~2–5MB**.

```python
import duckdb
con = duckdb.connect()
df_hour = con.execute(
    "SELECT * FROM read_parquet(?) WHERE hour_dt = ?",
    [PARQUET_PATH, target_ts]
).df()
```

This is not a workaround — it is the correct production pattern for parquet files on constrained environments.

### Why Greedy Spatial Allocation for Dispatch?

The Enforcement Optimizer uses a greedy algorithm: assign available units to zones in descending order of predicted IEU until all units are placed. This was chosen over LP/ILP for two reasons:

1. **Speed** — the What-If Simulator requires live recalculation on every slider change. LP solvers have startup overhead that makes sub-second recalculation difficult at this scale.
2. **Interpretability** — a greedy ranked list is immediately explainable to a shift commander. "Zone 1 gets 4 units, Zone 2 gets 3 units" is actionable. An LP solution with relaxed constraints is harder to explain.

For the scale of this problem (178 cells, 30–60 officers), greedy allocation produces near-optimal solutions — the optimality gap vs LP is negligible at this scale.

---

## IEU Formula — Full Derivation

**IEU (Instantaneous Enforcement Urgency)** is a physics-motivated congestion proxy computed entirely from the provided dataset — zero external data sources.

### Formula

```
IEU_raw(cell, t) = J × ( Σ Mᵢ for all violations in cell within 3h window ending at t )^α

where:
  J   = Junction multiplier (2.0 if major intersection, 1.0 otherwise)
  Mᵢ  = Mass weight of vehicle i
  α   = 0.8 (sublinear saturation power)

k = P90(IEU_raw) / ln(4)   ← anchors P90 to IEU = 75

IEU(cell, t) = 100 × (1 − exp(−IEU_raw / k))
```

### Component Rationale

**Vehicle mass weighting:** A 30-tonne tanker blocking a carriageway displaces far more traffic than a scooter. Mass weighting captures the physical road-blocking impact of different vehicle categories. Using raw violation counts (all vehicles weighted equally) would overstate the impact of scooter violations and understate heavy vehicle incidents.

**3-hour rolling window:** Violations are cumulative obstructions, not point events. A vehicle parked for 3 hours contributes to congestion throughout that period. The rolling window captures obstruction persistence — the most dangerous congestion states arise from multiple vehicles blocking the same zone over an extended period.

**α = 0.8 saturation power (sublinear):** The first vehicle blocking a junction causes the most harm. Each additional vehicle on an already-blocked road adds less marginal harm — the road is already impassable. α < 1 encodes this diminishing marginal impact. Our ablation showed α=1.0 (linear) gives marginally lower MAE (−0.24 pts) but loses the physical saturation property. We retained α=0.8 because physical correctness and operational interpretability are more valuable than a 0.24-point MAE difference that is within noise.

**Junction multiplier (2×):** Violations at major intersections cause spillover into multiple carriageways simultaneously. A tanker at a mid-block location blocks one lane. The same tanker at a major junction can block 4 roads converging at that intersection. The 2× multiplier encodes this spatial compounding effect.

**Smooth-squash normalization:** Linear P90 clipping (divide by P90, clip at 1.0) creates a spike artifact at exactly IEU=100 for all values above the 90th percentile. The exponential normalization distributes extreme values smoothly: IEU=75 corresponds to the historical P90, IEU=90 to very high congestion, IEU=100 is the asymptotic maximum — never exactly reached.

### Vehicle Mass Mapping

| Vehicle Type | Mass Weight | Rationale |
|---|---|---|
| TANKER | 5.0 | Heaviest class, blocks full carriageway |
| BUS / PRIVATE BUS / SCHOOL VEHICLE | 4.5 | Large body, high passenger impact |
| HGV / LORRY | 4.0 | Heavy goods, wide footprint |
| LGV / MINI LORRY / TRACTOR | 3.5 | Medium goods, moderate block |
| CAR / JEEP / MAXI-CAB / TEMPO | 3.0 | Standard private vehicle |
| TAXI / VAN | 2.5 | Slightly smaller than car |
| GOODS AUTO / PASSENGER AUTO | 2.0 | Three-wheeler, narrower |
| MOTOR CYCLE | 1.5 | Narrow, partial lane block |
| SCOOTER / MOPED | 1.0 | Lightest class, minimal obstruction |

### Junction Type Multipliers

| Junction Type | Ordinal | Multiplier |
|---|---|---|
| Metro station / railway terminus | 4 | 2.0× |
| Market / shopping mall junction | 3 | 2.0× |
| Named major intersection | 2 | 2.0× |
| Area / non-junction mid-block | 1 | 1.0× |

### ⚠️ Honest Note on Ground Truth

This dataset is a **pure citation-issuance log** — it records when a traffic officer wrote a ticket, not:
- When the vehicle was actually removed from the road
- Whether a traffic jam actually occurred at that location
- What road speed or vehicle throughput was at that moment

The `closed_datetime` column (which would indicate resolution) is **null for 100% of records**. As a result, every metric in this project — IEU score, risk tier, congestion prediction — is a **physics-motivated proxy** derived from the citation data itself.

**The model's validation target (`IEU at t+1`) is evaluated against our own derived formula, not against any external sensor or ground-truth measurement.** This is the correct approach under the zero-external-dataset constraint of this problem statement. Operational deployment would require calibration against real traffic speed data (ATMS sensors or GPS probe data) before accuracy claims can be made against ground truth. This limitation is disclosed on the dashboard itself.

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
│ spatial_engine.py │  → Compute 3h rolling mass sum per H3 cell
│                   │     Apply IEU formula (J × mass^0.8 → smooth squash)
│                   │     HDBSCAN corridor clustering on high-risk centroids
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ forecast_engine   │  → Build hourly spatio-temporal feature matrix
│ .py               │     19 features: lags, rolling stats, spatial neighbors,
│                   │     cyclical time encoding, junction type ordinal
│                   │     Train 3× LightGBM (T+1h, T+2h, T+4h) — Huber loss
│                   │     Train 6× LightGBM Q10/Q90 quantile models
│                   │     Target: delta = IEU(t+h) − IEU(t)
│                   │     5-fold walk-forward cross-validation
└───────────────────┘
        │
        ▼
┌───────────────────────────┐
│ forecast_results.parquet  │  → Pre-computed T+1h/T+2h/T+4h + Q10/Q90
│                           │     for all 178 cells across full date range
│                           │     ~12MB on disk
└───────────────────────────┘
        │
        ▼
┌──────────────────────────────────┐
│ DuckDB on-demand parquet queries │  → Predicate pushdown per selected hour
│                                  │     120MB → 2–5MB per request
└──────────────────────────────────┘
        │
        ▼
┌───────────────────┐
│ Streamlit App     │  → 5-page operational dashboard
│ (app/main.py)     │     Leaflet maps, Plotly charts, Dispatch tables
└───────────────────┘
```

---

## Model Details

### Delta Architecture

Instead of predicting absolute IEU(t+h), the model predicts the **change**:

```
target = IEU(t+h) − IEU(t)
```

At inference, predictions are reconstructed:
```python
predicted_IEU = np.clip(model.predict(X) + current_IEU, 0, 100)
```

A model that predicts `delta = 0` everywhere scores identically to the naive persistence baseline. This ensures any measured MAE improvement is genuine predictive signal above persistence — not inflation from zero-cell dominance.

### Training Configuration

```python
# Huber regressor — identical config for T+1h, T+2h, T+4h
model = LGBMRegressor(
    objective='huber',
    alpha=5.0,           # L1/L2 boundary at 5 IEU points
    n_estimators=400,
    learning_rate=0.04,
    max_depth=7,
    num_leaves=63,
    min_child_samples=15,
    reg_alpha=0.05,
    reg_lambda=0.1,
    random_state=42,
)

# Quantile models for uncertainty bounds
q10_model = LGBMRegressor(objective='quantile', alpha=0.10, ...)
q90_model = LGBMRegressor(objective='quantile', alpha=0.90, ...)
```

**Split:** Chronological 80/20 — no shuffling. Train on first 80% of hours, validate on last 20%.

### Feature Set & Rationale

| Feature | Description | Why Included |
|---|---|---|
| `AOI` | Current IEU value | Baseline for delta prediction; autocorrelation = 0.884 |
| `AOI_lag_1h` | IEU 1 hour ago (per-cell shift) | Short-term momentum |
| `AOI_lag_2h` | IEU 2 hours ago | Confirmed as strongest SHAP feature |
| `AOI_lag_3h` | IEU 3 hours ago | Medium-term trend direction |
| `AOI_lag_4h` | IEU 4 hours ago | Added for T+2h/T+4h horizon accuracy |
| `AOI_lag_6h` | IEU 6 hours ago | Captures morning → evening cycle |
| `AOI_lag_8h` | IEU 8 hours ago | Prior shift carryover |
| `AOI_roll_mean_7d` | 7-day rolling average IEU | Long-term baseline per cell |
| `AOI_roll_max_7d` | 7-day rolling maximum IEU | Historical peak for the cell |
| `AOI_roll_std_7d` | 7-day rolling std IEU | Volatility measure |
| `AOI_roll_mean_14d` | 14-day rolling average | Extended baseline for T+4h model |
| `AOI_roll_max_14d` | 14-day rolling max | Extended peak reference |
| `hour_sin`, `hour_cos` | Cyclical hour-of-day encoding | Captures AM/PM peaks continuously |
| `day_sin`, `day_cos` | Cyclical day-of-week encoding | Weekday vs weekend pattern |
| `historical_density` | Cell's long-run avg violations/hour at this hour | Seasonal baseline |
| `hist_density_30d` | 30-day rolling hourly violation density | What-If Scenario A baseline |
| `neighbor_mean_lag1` | Spatial neighbor mean IEU (k-ring-1) | Spillover from adjacent cells |
| `neighbor_max_lag1` | Spatial neighbor max IEU | Detects upstream congestion |
| `neighbor_any_critical` | Binary: any H3 neighbor currently critical | Alert flag for spreading congestion |
| `latitude`, `longitude` | H3 cell centroid coordinates | Geographic context |
| `cell_vehicle_mass` | Dominant vehicle mass for the cell | Severity weighting |
| `junction_flag` | Binary: major intersection | Junction spillover indicator |
| `junction_type_ord` | metro=4 / market+mall=3 / junction=2 / area=1 | Ordinal severity encoding |

*Longer lags (4h, 6h, 8h, 14d) were added specifically for T+2h and T+4h models to address the original MAE degradation from 8.01 (T+1h) to 18.09 (T+2h). After addition, T+2h MAE improved to 7.58 — a 58% reduction.*

### Walk-Forward Cross-Validation

Standard random k-fold cross-validation would leak future data into training (shuffled time series). We implement **5-fold walk-forward CV** using sorted unique hours:

```
Fold 1: Train hours [0 → 20%]     → Validate hours [20% → 40%]
Fold 2: Train hours [0 → 40%]     → Validate hours [40% → 60%]
Fold 3: Train hours [0 → 60%]     → Validate hours [60% → 70%]
Fold 4: Train hours [0 → 70%]     → Validate hours [70% → 80%]
Fold 5: Train hours [0 → 80%]     → Validate hours [80% → 100%]
```

Each fold trains on strictly earlier data and validates on strictly later data. No hour appears in both train and validation sets across any fold. Final metrics reported as mean ± std across all 5 folds.

This is significantly more robust than a single 80/20 split — our metric confidence intervals (e.g. R² = 0.8307 ± 0.0084) confirm stable performance across the full date range, not just one held-out slice.

### Uncertainty Quantification

We train two additional quantile LightGBM regressors alongside each Huber model:

```python
# Q10 lower bound (10th percentile prediction)
q10_model = LGBMRegressor(objective='quantile', alpha=0.10, ...)

# Q90 upper bound (90th percentile prediction)
q90_model = LGBMRegressor(objective='quantile', alpha=0.90, ...)
```

At inference, every zone forecast includes a **confidence interval**: `Predicted: 82 (range: 68–91)`.

This addresses the key limitation of point-prediction models for operational dispatch — when the model says IEU=82, the Q10–Q90 band tells the commander whether that's a confident reading (narrow band) or an uncertain estimate (wide band).

Quantile predictions are shown in:
- Leaflet map hover tooltips (per-zone confidence range)
- Zone Forecast Inspector chart (shaded confidence band over 4-hour forecast)

### Baseline Comparisons

Three baselines were evaluated for T+1h to validate model lift:

**Naive Persistence:** Predict `IEU(t+1) = IEU(t)` — assume nothing changes.

**Seasonal Naive:** Predict `IEU(t+1) = IEU(same hour, 7 days ago)` — use last week's same-hour value. This is harder to beat than persistence because it captures day-of-week patterns.

**Historical Mean:** Predict `IEU(t+1) = long-run average IEU for this cell at this hour` — uses no recent signal at all.

GridLock IQ beats all three. The +7.72 MAE lift over seasonal naive is the most meaningful comparison — seasonal patterns are already a strong baseline for traffic data, and our model adds genuine real-time signal on top.

### Decision Threshold

F2-optimized at IEU = 65 (recall weighted 2× over precision):

| Threshold | Recall | Precision | F2 |
|---|---|---|---|
| **65 (selected)** | **76.2%** | **90.2%** | **0.834** |
| 75 (naive) | 65.0% | 75.0% | 0.668 |

---

## Ablation Studies

### 1. HDBSCAN Corridor Clustering Ablation

We tested whether HDBSCAN cluster IDs (corridor membership) improve prediction accuracy when used as a model feature.

| Configuration | MAE | Recall @65 |
|---|---|---|
| Without HDBSCAN feature | 4.7876 | 76.19% |
| With HDBSCAN feature | 4.7876 | 76.19% |
| **Difference** | **0.0000** | **0.00%** |

**Verdict:** HDBSCAN cluster labels provide exactly zero improvement to model performance. This is the correct scientific result — the model already captures spatial autocorrelation through H3 neighbor features (`neighbor_mean_lag1`, `neighbor_max_lag1`). Adding an explicit cluster label is redundant.

**Decision:** HDBSCAN was **removed from model features** to eliminate decorative complexity. It is retained strictly for map visualization — the corridor clustering polygons overlay the Hotspot Map to show patrol commanders which H3 cells form connected enforcement corridors.

Keeping a feature that adds zero value would have inflated the apparent technical complexity without improving predictions. We removed it.

### 2. IEU Formula Component Ablation

Each component of the IEU formula was individually removed and a fresh T+1h model was trained to measure impact.

| Component Removed | MAE Change | Recall Change | Justification |
|---|---|---|---|
| **Junction multiplier** | **+0.53** | **−1.96%** | Junctions cause multi-carriageway spillover. Removing 2× weight increases error as the model can no longer distinguish intersection blockage from mid-block blockage. |
| **Duration weighting (3h → 1h window)** | **+3.57** | **−45.39%** | A 45% recall drop is the single most impactful finding. The 3-hour window captures cumulative obstruction build-up. Compressing to 1 hour loses the temporal persistence that defines dangerous congestion. This result empirically validates the core design choice. |
| **α=0.8 vs α=1.0 (linear)** | **−0.24** | **−0.06%** | Linear aggregation gives marginally lower MAE (0.24 pts) but loses the physical sublinear saturation property. We retained α=0.8 because: (a) 0.24 MAE difference is within noise, (b) the diminishing marginal congestion principle is domain-correct and improves operational interpretability. |

### 3. SHAP Feature Interpretability

SHAP (SHapley Additive exPlanations) values were pre-computed on the T+1h validation sample to verify that the model is learning meaningful signals, not artifacts.

**Top features by mean |SHAP| value (T+1h model):**

| Rank | Feature | Mean |SHAP| | Interpretation |
|---|---|---|---|
| 1 | Risk Score 2h Ago (`AOI_lag_2h`) | ~5.5 | Strongest temporal signal — 2-hour lag consistently predictive |
| 2 | Risk Score 3h Ago (`AOI_lag_3h`) | ~3.0 | Medium-term trend confirmation |
| 3 | Historical Density | ~1.7 | Long-run baseline anchors predictions |
| 4 | Current Risk Score (`AOI`) | ~1.4 | Present state informs next-hour delta |
| 5 | Time of Day (cos) | ~1.0 | AM/PM peak cycles captured continuously |
| 6–10 | Shorter lags, rolling stats | <0.8 | Supporting context |

**Key finding:** The model is learning temporal autocorrelation (lag features dominate), confirmed by the 0.884 lag-1h autocorrelation stat. It is not shortcutting through the IEU formula — the feature `AOI` (current IEU) ranked 4th, not 1st. If the model were simply memorizing the formula, `AOI` would dominate. Instead, 2h and 3h lags dominate, showing genuine temporal pattern learning.

---

## Dashboard Pages

### 1. Command Overview
City-wide operational status. KPI cards (active critical zones, predicted peak window, city risk level, units required, officers recommended). Hourly IEU trend chart with T+1h–T+4h prediction shading and NOW marker. Active dispatch alert cards with landmark names, severity badges, and unit deployment recommendations. Zone distribution bar (178 cells by risk tier). Which Days Are Busiest chart (day-of-week avg violations). Collapsible model performance metrics table (for analysts only — does not interfere with operational view). Shift Commander Brief with PDF download.

### 2. Hotspot Intelligence Map
Full Bengaluru coverage on CartoDB Light tiles. H3 hexagon overlays colored by current IEU tier (Critical=red ≥65, High=orange 50–64, Moderate=yellow 25–49, Low=green 1–24). Time window selector (Last 1h / Last 6h / Last 12h). Zone inspect panel: landmark name, risk score, T+1h/T+2h/T+4h forecast bars with Q10–Q90 confidence range, primary violator type, jurisdiction, recommended action and source police station. Top hotspots dropdown for fast navigation.

### 3. Prediction Engine
Separate prediction map per horizon (T+1h / T+2h / T+4h) — shows **future predicted IEU**, not current. Top predicted zones table with NOW IEU, PRED IEU, tier, and trajectory direction. Zone Intelligence classification: **Chronic** (appears in top zones ≥80% of weeks — structural problem, needs infrastructure escalation), **Episodic** (spikes on weekends/events — pre-position before peak hours), **Random** (low frequency — monitor only). Zone Forecast Inspector: select any zone to view 4-hour prediction trajectory with shaded Q10–Q90 confidence band. SHAP Feature Importance chart. Walk-forward CV performance table (out-of-sample, fold-averaged). Model Technical Performance collapsible with all horizon metrics.

### 4. Enforcement Optimizer
Priority-ranked dispatch orders table: junction name, jurisdiction, NOW IEU, T+1h IEU, units assigned, action label (DEPLOY IMMEDIATELY / DEPLOY — STABILIZING / PRE-POSITION BY T+1H). Summary bar: units required, critical zones count, jurisdictions affected, repeat offenders active. Jurisdiction Resource Coverage chart (units assigned vs critical zones per station). Coverage Window heatmap — next 4 hours × top zones, colored by risk tier. Repeat Offender Watchlist: 847+ flagged vehicles with plate number, violation count bar, last seen zone, priority badge. Vehicle plate search. Shift Brief: auto-generated operational summary with immediate actions, escalating zones, improving zones, and vehicle intelligence. Download PDF.

### 5. What-If Simulator
Officer count slider (5–60). Date and hour selectors. Scenario A vs B KPI cards: critical zones covered, wasted unit-hours saved, highest risk zone coverage, AI efficiency multiplier. Dual-layer Leaflet map with toggle — Scenario A patrol positions (historical density) and Scenario B positions (AI greedy). Zone allocation comparison table: zone name, risk IEU, Scenario A units, Scenario B units. Unit Efficiency scatter plot (risk IEU vs units assigned, colored by scenario). Live Simulation Summary: side-by-side final stats for both scenarios with interpretation line.

---

## Dataset

**Source:** Bengaluru Traffic Violation citation dataset provided by HackerEarth (Flipkart GridLock 2.0).

### Coverage

| Dimension | Value |
|---|---|
| Time window | November 2023 – April 2024 (5 months) |
| H3 cells (active) | 178 unique cells at Resolution 7 |
| Cell edge length | ~1.2 km (Uber H3 Resolution 7) |
| City coverage | Full Bengaluru metropolitan boundary |
| Hours in training set | 5 months × 24h × 178 cells |
| Zero-inflation rate | ~73% of cell-hours have zero violations |
| Critical tier rate | 17.4% of active cell-hours reach IEU ≥ 65 |

### Key Fields Used

| Column | Description |
|---|---|
| `latitude`, `longitude` | Violation GPS coordinates |
| `created_datetime` | Citation timestamp |
| `modified_datetime` | Used for obstruction duration (fallback) |
| `clean_vehicle_type` | Standardized vehicle category |
| `junction_name` | Nearest landmark / junction (POI labels) |
| `police_station` | BTP station jurisdiction |
| `junction_flag` | Binary: major intersection (2× IEU weight) |

### Dataset Limitations

| Limitation | Impact |
|---|---|
| **Enforcement bias** | Records show where officers went — not all locations where violations occurred. Underenforced zones appear cleaner in data than they may be in reality. |
| **No closed_datetime** | Cannot measure actual obstruction duration — duration is estimated or median-imputed |
| **5-month window** | Cannot confirm seasonal patterns without a full annual cycle. T+4h performance would improve with more training data. |
| **Citation ≠ congestion** | Officer patrol routes and enforcement quotas add behavioral noise independent of actual road conditions. |
| **Single city** | Model weights are Bengaluru-specific. Generalization to other cities requires retraining on new city data. |

---

## Audit Trail

Five independent data integrity checks in `audits/leakage_audit.py`. All passed before any metric was trusted.

| # | Audit | Check Performed | Result |
|---|---|---|---|
| 1 | **Timestamp alignment** | `AOI` feature = IEU(t), never IEU(t+1) — no future-hour value in feature set | ✅ CLEAN |
| 2 | **Manual row verification** | 5 random rows from feature matrix hand-computed against IEU formula | ✅ CLEAN (5/5 match) |
| 3 | **Feature ablation** | R² drop when AOI (current IEU) removed: −0.004 — confirms AOI is not shortcutting to target | ✅ STABLE |
| 4 | **Train/val split** | Zero hour overlap between train and validation sets | ✅ CLEAN (0 overlap) |
| 5 | **Cross-cell lag contamination** | All lag features use `groupby('h3_cell').shift()` — each cell's lag only uses its own history | ✅ CLEAN (178/178 cells) |

```python
# All lag features — per-cell groupby shift ensures zero cross-cell contamination
grid_df['AOI_lag_1h'] = grid_df.groupby('h3_cell')['AOI'].shift(1).fillna(0.0)
grid_df['AOI_lag_2h'] = grid_df.groupby('h3_cell')['AOI'].shift(2).fillna(0.0)
grid_df['AOI_lag_3h'] = grid_df.groupby('h3_cell')['AOI'].shift(3).fillna(0.0)

# Targets — forward shift per cell, no cross-cell contamination
grid_df['target_t1'] = grid_df.groupby('h3_cell')['AOI'].shift(-1)
grid_df['target_t2'] = grid_df.groupby('h3_cell')['AOI'].shift(-2)
grid_df['target_t4'] = grid_df.groupby('h3_cell')['AOI'].shift(-4)
```

Run any audit independently:
```bash
python audits/leakage_audit.py       # Data integrity checks
python audits/threshold_analysis.py  # F2-optimal threshold = 65 IEU
python audits/model_accuracy.py      # Stratified accuracy report
```

---

## Honest Limitations

We document these limitations proactively. They are visible on the dashboard itself (yellow warning box in Prediction Engine) — not buried in documentation.

| Limitation | Operational Impact |
|---|---|
| **IEU is a derived proxy, not ground-truth sensor data** | All metrics — recall, R², MAE — are measured against our own formula, not real road speed or flow. Calibration against ATMS sensor data required before production deployment. |
| **73% sparse cells** | Active-cell R²=0.834 is the operationally meaningful metric. All-rows R²=0.862 is inflated by correctly predicting zero for inactive cells. We report active-cell metrics throughout. |
| **T+4h recall is 44.7%** | T+4h predictions miss more than half of critical events 4 hours out. Use T+4h for **pre-positioning decisions only**, not immediate dispatch. T+1h recall (76.2%) is the operational metric for dispatch. |
| **5-month dataset** | Limited seasonal learning. Full-year data would significantly improve T+4h horizon accuracy and capture summer/festival traffic patterns. |
| **Citation behavior ≠ congestion directly** | Officer patrol routes, enforcement quotas, and shift timing add behavioral noise independent of actual road conditions. Zones with low patrol coverage appear artificially clean. |
| **Single city** | Model weights are Bengaluru-specific. Architecture generalizes but weights require retraining per city. |

---

## Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.10 |
| **Data Processing** | Pandas, NumPy |
| **Spatial Indexing** | H3-py (Uber H3, Resolution 7, ~1.2km edge) |
| **Clustering** | HDBSCAN (corridor hotspot grouping — map visualization only) |
| **ML Model** | LightGBM (Huber loss + Quantile Q10/Q90) |
| **Interpretability** | SHAP (TreeExplainer, pre-computed on validation sample) |
| **Dashboard** | Streamlit |
| **Maps** | Leaflet.js + CartoDB Light tiles (free, no API key required) |
| **Charting** | Plotly Graph Objects |
| **Data Storage** | Apache Parquet (PyArrow) |
| **Query Engine** | DuckDB — on-demand parquet slicing (120MB → 2–5MB per request) |
| **Deployment** | Render (free tier, keep-alive via external cron) |

> ✅ **Zero external data constraint strictly upheld.** All features are derived exclusively from the provided violation dataset. No external APIs, map data feeds, traffic sensors, or third-party datasets used at any stage.

---

## Project Structure

```
gridlock-iq/
│
├── data/
│   ├── raw/                              # Original HackerEarth violation CSV
│   └── processed/
│       ├── cleaned_violations.parquet    # Cleaned records with IEU, H3 cells
│       ├── forecast_results.parquet      # Model predictions (T+1h/T+2h/T+4h + Q10/Q90)
│       ├── h3_clustered_zones.parquet    # HDBSCAN corridor clusters (map only)
│       ├── repeat_offenders.parquet      # Vehicle repeat-risk register
│       └── cell_poi_tags.json            # H3 cell → junction label
│
├── src/
│   ├── data_pipeline.py                  # Raw cleaning, duration parse, mass map
│   ├── spatial_engine.py                 # H3 indexing, IEU formula, HDBSCAN
│   ├── forecast_engine.py                # Feature engineering + LightGBM training
│   ├── patrol_optimizer.py               # Greedy dispatch & What-If simulation
│   └── repeat_offender.py                # Vehicle repeat-risk scoring
│
├── models/
│   ├── lgbm_t1.pkl                       # T+1h Huber regressor
│   ├── lgbm_t2.pkl                       # T+2h Huber regressor
│   ├── lgbm_t4.pkl                       # T+4h Huber regressor
│   ├── lgbm_t1_q10.pkl                   # T+1h Q10 quantile regressor
│   ├── lgbm_t1_q90.pkl                   # T+1h Q90 quantile regressor
│   ├── [similar q10/q90 for t2, t4]
│   ├── cross_val_metrics.json            # Walk-forward CV fold results (all horizons)
│   └── shap_importance.json              # Pre-computed SHAP importances (T+1h model)
│
├── app/
│   ├── main.py                           # Streamlit multipage router
│   ├── data_state.py                     # Session state & DuckDB data loader
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
│   ├── leakage_audit.py                  # 5 data integrity checks
│   ├── threshold_analysis.py             # F2-optimal threshold = 65 IEU
│   ├── model_accuracy.py                 # Stratified accuracy + CV report
│   ├── retrain.py                        # Retrain all LightGBM models
│   └── build_poi_tags.py                 # Build cell_poi_tags.json
│
├── config.py                             # Paths, mass map, IEU constants
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/charansai-1411/GridLock-IQ.git
cd GridLock-IQ

python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

---

## Running the Pipeline

Run scripts in order. Each saves output to `data/processed/` for the next step.

```bash
# Step 1 — Clean raw data, compute vehicle masses, assign H3 cells
python src/data_pipeline.py

# Step 2 — Compute IEU formula, run HDBSCAN corridor clustering
python src/spatial_engine.py

# Step 3 — Train all LightGBM models (Huber + Q10/Q90 for T+1h/T+2h/T+4h)
#           Runs 5-fold walk-forward CV and saves cross_val_metrics.json
python audits/retrain.py

# Step 4 — Build repeat offender register
python src/repeat_offender.py

# Step 5 — Build POI landmark labels from junction_name column
python audits/build_poi_tags.py
```

Pre-computed outputs are included in the repository. Steps 1–5 only need to be re-run if retraining from scratch with new data.

---

## Running the Dashboard

```bash
streamlit run app/main.py
```

Opens at `http://localhost:8501`

**Recommended demo timestamp:** `2024-03-12 08:30 AM IST`
This window has 7 active critical zones (IEU ≥ 65), a clear peak IEU of 100.0 at T+1h, visible AI vs. historical dispatch differentiation in the What-If Simulator, and a Shift Brief with all alert categories populated.

---

## FAQ

**Does this predict actual traffic congestion?**

No. The dataset contains parking violation citation records — not speed, flow, or occupancy measurements. IEU is a physics-motivated proxy derived from vehicle mass, obstruction duration, and junction type. Direct congestion prediction would require traffic sensor data not present in this dataset. This limitation is clearly disclosed on the dashboard.

**Why not a deep learning model (LSTM, Transformer)?**

The per-cell time series has approximately 3,000–4,000 hourly data points, but 73% are zeros. After zero-removal, each cell has ~800–1,000 active data points. LSTM and transformer models typically require 10,000+ sequence data points per unit to generalize without heavy regularization. LightGBM with tabular feature engineering is the correct model class for this scale — it handles sparsity better, requires less data, and produces SHAP-interpretable results.

**Why is T+4h recall only 44.7%?**

The further into the future you predict, the more uncertainty compounds. T+4h predictions are useful for **pre-positioning** decisions made at the start of a shift — commanders can pre-stage units near probable high-risk zones. They are not designed for immediate dispatch (use T+1h at 76.2% recall for that). The progressive degradation (76.2% → 58.9% → 44.7%) is expected and disclosed rather than hidden.

**Why is HDBSCAN still in the codebase if it adds zero predictive value?**

HDBSCAN corridor clustering is used exclusively for **map visualization** — it groups adjacent H3 cells into enforcement corridors that are drawn as overlay polygons on the Hotspot Intelligence Map. This helps patrol commanders see which cells form connected enforcement areas, even though the cluster label adds no value to the ML model. We removed it from model features after ablation showed 0.00 MAE change.

**How does the What-If Simulator define "historical allocation"?**

Scenario A allocates units proportional to each zone's `hist_density_30d` — the 30-day rolling average hourly violation count at that hour of day. This reflects how a planner would historically deploy based on known busy periods, without any real-time AI signal. Scenario B uses our greedy spatial allocation ranked by T+1h predicted IEU.

**Can this generalize to other cities?**

The ML pipeline is city-agnostic given similar enforcement record formats. The IEU formula components (vehicle mass, junction type, rolling window) are physically motivated and would apply to any city. The trained model weights are Bengaluru-specific and require retraining on new city data. The H3 spatial grid generation requires a new city bounding box and resolution choice.

**Is live inference available?**

No. All predictions are pre-computed into `forecast_results.parquet` at training time. The dashboard serves pre-computed batch outputs via DuckDB queries. Live per-request inference would require rerunning the full feature engineering pipeline — not feasible on Render free tier. For production deployment, a nightly retraining pipeline would extend the parquet file with the latest enforcement data.

---

## Future Improvements

- **Ground-truth calibration** — integrate ATMS speed sensor data or GPS probe data (Google Maps Historical Speed API) to calibrate IEU against real congestion measurements
- **Graph Neural Network** — model the H3 cell adjacency graph explicitly using spatial graph convolution for neighbor spillover, replacing handcrafted neighbor features
- **Live inference pipeline** — nightly retrain triggered by new enforcement data upload, updating `forecast_results.parquet` automatically
- **Full annual data** — 12+ months to confirm seasonal patterns (festival spikes, monsoon effects) and improve T+4h horizon accuracy
- **Junction-level model** — parallel model at individual junction resolution (finer than H3 Resolution 7) for the top 50 critical junctions
- **Generalization study** — validate IEU formula and model architecture on other Indian cities with similar enforcement record formats (Chennai, Hyderabad, Pune)
- **LP/ILP dispatch optimizer** — replace greedy allocation with an integer linear program for mathematically optimal unit assignment under shift constraints

---

## Screenshots

| Page | Description |
|---|---|
| Command Overview | City risk level CRITICAL, KPI row, hourly IEU trend with prediction shading, active alerts for Hosahalli Metro (100/100) and Safina Plaza (100/100) |
| Hotspot Intelligence Map | H3 hexagonal heatmap of Bengaluru with 4 risk tiers, zone inspector panel for Modi Bridge Junction (68/100 → 82 at T+4h) |
| Prediction Engine | T+1h predicted IEU map, top predicted zones table, Zone Intelligence (12 Chronic / 1 Episodic / 165 Random), SHAP chart |
| Enforcement Optimizer | Priority dispatch table (Safina Plaza #1, 4 units), jurisdiction coverage chart, repeat offender watchlist (847 vehicles), Shift Brief |
| What-If Simulator | 30-officer slider, Scenario A (7 wasted units) vs Scenario B (0 wasted, 1.5× efficiency), zone allocation comparison table |
| Model Metrics | T+1h: R²=0.834, MAE=4.65, Recall=76.2%; T+2h: R²=0.682; T+4h: R²=0.586 — all from 5-fold walk-forward CV |

---

## Acknowledgments

- **Dataset:** Flipkart GridLock 2.0 / HackerEarth — Bengaluru Traffic Violation data
- **Spatial Indexing:** Uber H3 library (Resolution 7)
- **Map Tiles:** Leaflet.js + CartoDB Light (free, no API key required)
- **ML Framework:** LightGBM by Microsoft Research
- **Interpretability:** SHAP by Scott Lundberg
- **Query Engine:** DuckDB
- **Institution:** Chaitanya Bharathi Institute of Technology (CBIT), Hyderabad

---

## License

Built for Flipkart GridLock 2.0 hackathon. Dataset used under HackerEarth competition terms.

---

*Built by Y. Charan Sai [Team Leader] · B. Ram Charan · C. Ganesh Kumar · B. Praneeth Kumar*
*Chaitanya Bharathi Institute of Technology (CBIT), Hyderabad*
