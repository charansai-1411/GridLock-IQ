"""
Leakage audit for the AOI feature in the delta model.

Checks (in order):
1. Timestamp alignment: what does AOI resolve to relative to prediction time t?
2. Manual row verification: 5 rows, cross-check AOI/lags/target by hand from raw data
3. AOI ablation: train T+1h without AOI, measure active-cell R² collapse
4. Train/test split sanity: confirm no row-level shuffling
"""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')
from sklearn.metrics import mean_absolute_error, r2_score
from lightgbm import LGBMRegressor

# ─── LOAD ────────────────────────────────────────────────────────────────────
print("=" * 65)
print("AUDIT 1: Timestamp alignment")
print("=" * 65)

# Rebuild the feature matrix the same way forecast_engine does
from src.forecast_engine import build_features
df_raw = pd.read_parquet('data/processed/cleaned_violations.parquet')
grid = build_features(df_raw)

# Add delta targets (same as train_all_models)
grid['delta_t1'] = grid['target_t1'] - grid['AOI']
grid['delta_t2'] = grid['target_t2'] - grid['AOI']
grid['delta_t4'] = grid['target_t4'] - grid['AOI']

print("\nFor an arbitrary cell-hour row, verify column meanings:")
# Pick one cell that has a few active hours
cell = grid[grid['violation_count'] > 0]['h3_cell'].value_counts().index[0]
sample = grid[grid['h3_cell'] == cell].sort_values('hour_dt').head(6)[
    ['hour_dt', 'AOI', 'AOI_lag_1h', 'AOI_lag_2h', 'AOI_lag_3h', 'target_t1', 'delta_t1']
].reset_index(drop=True)
print(sample.to_string())

print("\nVerification rule:")
print("  row[i].AOI_lag_1h  should == row[i-1].AOI")
print("  row[i].target_t1   should == row[i+1].AOI")
print("  row[i].delta_t1    should == row[i+1].AOI - row[i].AOI")
for i in range(1, len(sample) - 1):
    lag1_ok   = abs(sample.loc[i, 'AOI_lag_1h'] - sample.loc[i-1, 'AOI']) < 0.01
    target_ok = abs(sample.loc[i, 'target_t1']  - sample.loc[i+1, 'AOI']) < 0.01
    delta_ok  = abs(sample.loc[i, 'delta_t1'] - (sample.loc[i+1, 'AOI'] - sample.loc[i, 'AOI'])) < 0.01
    print(f"  row {i}: lag1_ok={lag1_ok}  target_ok={target_ok}  delta_ok={delta_ok}")

# ─── AUDIT 2: Manual random row check ────────────────────────────────────────
print()
print("=" * 65)
print("AUDIT 2: Manual random row verification (5 active rows)")
print("=" * 65)

np.random.seed(42)
active = grid[(grid['AOI'] > 0) & grid['target_t1'].notna()].copy()
sample5 = active.sample(5, random_state=7)

for _, row in sample5.iterrows():
    cell_id  = row['h3_cell']
    t        = row['hour_dt']
    cell_seq = grid[grid['h3_cell'] == cell_id].sort_values('hour_dt').reset_index(drop=True)
    idx      = cell_seq[cell_seq['hour_dt'] == t].index[0]

    aoi_t   = cell_seq.loc[idx,   'AOI']
    aoi_tm1 = cell_seq.loc[idx-1, 'AOI'] if idx >= 1 else np.nan
    aoi_tm2 = cell_seq.loc[idx-2, 'AOI'] if idx >= 2 else np.nan
    aoi_tm3 = cell_seq.loc[idx-3, 'AOI'] if idx >= 3 else np.nan
    aoi_tp1 = cell_seq.loc[idx+1, 'AOI'] if idx+1 < len(cell_seq) else np.nan

    feat_aoi  = row['AOI']
    feat_l1   = row['AOI_lag_1h']
    feat_l2   = row['AOI_lag_2h']
    feat_l3   = row['AOI_lag_3h']
    tgt_t1    = row['target_t1']
    dlt_t1    = row['delta_t1']

    print(f"\n  cell={cell_id[:12]}...  t={t}")
    print(f"  Expected: AOI={aoi_t:.2f}  lag1={aoi_tm1:.2f}  lag2={aoi_tm2:.2f}  lag3={aoi_tm3:.2f}  tgt={aoi_tp1:.2f}")
    print(f"  Got:      AOI={feat_aoi:.2f}  lag1={feat_l1:.2f}  lag2={feat_l2:.2f}  lag3={feat_l3:.2f}  tgt={tgt_t1:.2f}")

    ok_aoi = abs(feat_aoi - aoi_t)   < 0.01
    ok_l1  = abs(feat_l1  - aoi_tm1) < 0.01 if not np.isnan(aoi_tm1) else True
    ok_l2  = abs(feat_l2  - aoi_tm2) < 0.01 if not np.isnan(aoi_tm2) else True
    ok_l3  = abs(feat_l3  - aoi_tm3) < 0.01 if not np.isnan(aoi_tm3) else True
    ok_tgt = abs(tgt_t1   - aoi_tp1) < 0.01 if not np.isnan(aoi_tp1) else True
    ok_dlt = abs(dlt_t1 - (aoi_tp1 - aoi_t)) < 0.01 if not np.isnan(aoi_tp1) else True
    all_ok = all([ok_aoi, ok_l1, ok_l2, ok_l3, ok_tgt, ok_dlt])
    print(f"  ✅ CLEAN" if all_ok else f"  ❌ MISMATCH: aoi={ok_aoi} l1={ok_l1} l2={ok_l2} l3={ok_l3} tgt={ok_tgt} dlt={ok_dlt}")

# ─── AUDIT 3: AOI ablation ────────────────────────────────────────────────────
print()
print("=" * 65)
print("AUDIT 3: Train T+1h WITHOUT 'AOI' — how much R² survives?")
print("=" * 65)

feature_cols_full = [
    'AOI', 'latitude', 'longitude', 'cell_vehicle_mass', 'junction_flag',
    'AOI_lag_1h', 'AOI_lag_2h', 'AOI_lag_3h',
    'AOI_roll_mean_7d', 'AOI_roll_max_7d', 'AOI_roll_std_7d',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'historical_density',
    'neighbor_mean_lag1', 'neighbor_max_lag1', 'neighbor_any_critical',
]
feature_cols_no_aoi = [f for f in feature_cols_full if f != 'AOI']

active_mask = (
    (grid['AOI'] > 0) | (grid['AOI_lag_1h'] > 0) | (grid['AOI_lag_2h'] > 0) |
    (grid['AOI_lag_3h'] > 0) | (grid['target_t1'] > 0) |
    (grid['target_t2'] > 0)   | (grid['target_t4'] > 0)
)
train_ready = grid[active_mask].copy()

unique_hours = sorted(grid['hour_dt'].unique())
split_time   = unique_hours[int(len(unique_hours) * 0.8)]
train_data   = train_ready[train_ready['hour_dt'] < split_time]
val_data     = train_ready[train_ready['hour_dt'] >= split_time]

train_h = train_data.dropna(subset=['delta_t1'])
val_h   = val_data.dropna(subset=['delta_t1'])

def evaluate_model(feat_cols, label):
    X_tr = train_h[feat_cols]
    y_tr = train_h['delta_t1']
    X_val = val_h[feat_cols]

    mdl = LGBMRegressor(
        objective='huber', alpha=5.0, n_estimators=400,
        learning_rate=0.04, max_depth=7, num_leaves=63,
        min_child_samples=15, reg_alpha=0.05, reg_lambda=0.1,
        random_state=42, verbose=-1, n_jobs=-1
    )
    mdl.fit(X_tr, y_tr)

    delta_preds = mdl.predict(X_val)
    ieu_preds   = np.clip(val_h['AOI'].values + delta_preds, 0.0, 100.0)
    y_true      = val_h['target_t1'].values

    mae_all  = mean_absolute_error(y_true, ieu_preds)
    r2_all   = r2_score(y_true, ieu_preds)

    act_mask = y_true > 0
    mae_act  = mean_absolute_error(y_true[act_mask], ieu_preds[act_mask])
    r2_act   = r2_score(y_true[act_mask], ieu_preds[act_mask])

    naive_mae = mean_absolute_error(y_true, np.clip(val_h['AOI'].values, 0, 100))

    print(f"\n  {label}:")
    print(f"    Active MAE={mae_act:.4f}  Active R2={r2_act:.4f}")
    print(f"    All   MAE={mae_all:.4f}  All   R2={r2_all:.4f}")
    print(f"    Naive persistence MAE={naive_mae:.4f}  lift={naive_mae - mae_all:+.4f}")
    return r2_act, mae_act

r2_with,    mae_with    = evaluate_model(feature_cols_full,    "WITH AOI   (19 features)")
r2_without, mae_without = evaluate_model(feature_cols_no_aoi, "WITHOUT AOI (18 features)")

print(f"\n  R² delta (active): {r2_with:.4f} → {r2_without:.4f}  "
      f"({'collapse' if r2_with - r2_without > 0.05 else 'stable'})")
print(f"  MAE delta (active): {mae_with:.4f} → {mae_without:.4f}")

if r2_with - r2_without > 0.05:
    print("\n  ⚠️  SIGNIFICANT DROP: most of the R² gain rides on AOI alone.")
    print("     The feature is likely legitimate (mean reversion signal),")
    print("     but the gain is not generalizable without it.")
else:
    print("\n  ✅ STABLE: AOI is not the sole source of gain.")

# ─── AUDIT 4: Split sanity ────────────────────────────────────────────────────
print()
print("=" * 65)
print("AUDIT 4: Train/test split sanity")
print("=" * 65)

print(f"\n  Split time: {split_time}")
print(f"  Train rows: {len(train_data):,}  max hour: {train_data['hour_dt'].max()}")
print(f"  Val rows:   {len(val_data):,}   min hour: {val_data['hour_dt'].min()}")

overlap = set(train_data['hour_dt'].unique()) & set(val_data['hour_dt'].unique())
print(f"  Hour overlap between train and val: {len(overlap)} hours  "
      f"({'❌ LEAKAGE' if overlap else '✅ CLEAN'})")

# Check no cell appears in train at time > split and val at time < split
train_max = train_data.groupby('h3_cell')['hour_dt'].max()
val_min   = val_data.groupby('h3_cell')['hour_dt'].min()
common_cells = train_max.index.intersection(val_min.index)
leaky_cells = (train_max[common_cells] > val_min[common_cells]).sum()
print(f"  Cells where train_max > val_min: {leaky_cells}  "
      f"({'❌ LEAKAGE' if leaky_cells else '✅ CLEAN'})")
