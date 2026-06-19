"""
Step 2: Production IEU quality — autocorr + critical% together
Step 3: Model accuracy stratified on active cells vs zero-baseline
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

print("=" * 65)
print("STEP 2: Production IEU signal quality")
print("=" * 65)

df = pd.read_parquet('data/processed/forecast_results.parquet')

# Lag-1h autocorrelation per cell (the real predictability measure)
df = df.sort_values(['h3_cell', 'hour_dt'])
df['lag1_AOI'] = df.groupby('h3_cell')['AOI'].shift(1)
valid = df[['AOI', 'lag1_AOI']].dropna()
autocorr = valid['AOI'].corr(valid['lag1_AOI'])

# Critical % (AOI >= 75) on active cell-hours only
active = df[df['violation_count'] > 0]
crit_pct  = (active['AOI'] >= 75).mean() * 100
crit_n    = (active['AOI'] >= 75).sum()
high_pct  = ((active['AOI'] >= 50) & (active['AOI'] < 75)).mean() * 100
mod_pct   = ((active['AOI'] >= 25) & (active['AOI'] < 50)).mean() * 100
low_pct   = (active['AOI'] < 25).mean() * 100

print(f"\nLag-1h autocorrelation (all cells):  {autocorr:.4f}")
print(f"  (0 = random noise | 1 = perfectly predictable)")
print(f"\nRisk tier distribution on ACTIVE cell-hours (n={len(active):,}):")
print(f"  Critical (>=75): {crit_n:,}  ({crit_pct:.1f}%)")
print(f"  High    (50-75): {high_pct:.1f}%")
print(f"  Moderate(25-50): {mod_pct:.1f}%")
print(f"  Low     (< 25): {low_pct:.1f}%")
print(f"\nAOI distribution on active cell-hours:")
print(f"  mean={active['AOI'].mean():.2f}  std={active['AOI'].std():.2f}  "
      f"p50={active['AOI'].quantile(0.50):.2f}  "
      f"p90={active['AOI'].quantile(0.90):.2f}  "
      f"max={active['AOI'].max():.2f}")

print()
print("=" * 65)
print("STEP 3: Model accuracy — ALL rows vs ACTIVE cells only")
print("=" * 65)

# Chronological split mirror (80/20 as in training)
unique_hours = sorted(df['hour_dt'].unique())
split_time   = unique_hours[int(len(unique_hours) * 0.8)]
val_df       = df[df['hour_dt'] >= split_time].copy()
print(f"\nValidation set: {split_time} onwards  ({len(val_df):,} rows)")

for horizon, pred_col, tgt_col in [
    ('T+1h', 'pred_t1', 'target_t1'),
    ('T+2h', 'pred_t2', 'target_t2'),
    ('T+4h', 'pred_t4', 'target_t4'),
]:
    sub = val_df.dropna(subset=[tgt_col])

    # --- ALL rows (including zero-zero pairs) ---
    mae_all  = mean_absolute_error(sub[tgt_col], sub[pred_col])
    rmse_all = root_mean_squared_error(sub[tgt_col], sub[pred_col])
    r2_all   = r2_score(sub[tgt_col], sub[pred_col])

    # --- ACTIVE only: rows where target > 0 (something WILL happen) ---
    act = sub[sub[tgt_col] > 0]
    if len(act) > 0:
        mae_act  = mean_absolute_error(act[tgt_col], act[pred_col])
        rmse_act = root_mean_squared_error(act[tgt_col], act[pred_col])
        r2_act   = r2_score(act[tgt_col], act[pred_col])
    else:
        mae_act = rmse_act = r2_act = float('nan')

    # --- CRITICAL only: rows where target >= 75 ---
    crit = sub[sub[tgt_col] >= 75]
    if len(crit) > 0:
        mae_crit  = mean_absolute_error(crit[tgt_col], crit[pred_col])
        rmse_crit = root_mean_squared_error(crit[tgt_col], crit[pred_col])
    else:
        mae_crit = rmse_crit = float('nan')

    print(f"\n--- {horizon} ---")
    print(f"  ALL rows     (n={len(sub):,}):        MAE={mae_all:.2f}  RMSE={rmse_all:.2f}  R2={r2_all:.4f}")
    print(f"  ACTIVE only  (n={len(act):,}, target>0): MAE={mae_act:.2f}  RMSE={rmse_act:.2f}  R2={r2_act:.4f}")
    print(f"  CRITICAL only(n={len(crit):,}, target>=75): MAE={mae_crit:.2f}  RMSE={rmse_crit:.2f}")

    # Naive baseline: predict target = lag_1h (persistence)
    lag_col = 'AOI_lag_1h'
    if lag_col in sub.columns:
        sub_naive = sub.dropna(subset=[lag_col])
        mae_naive = mean_absolute_error(sub_naive[tgt_col], sub_naive[lag_col])
        print(f"  Naive baseline (persist lag-1h): MAE={mae_naive:.2f}  "
              f"  Lift over naive: {mae_naive - mae_all:.2f} pts")

print()
print("=" * 65)
print("STEP 3b: Precision on critical zones (does the model find them?)")
print("=" * 65)
sub1 = val_df.dropna(subset=['target_t1'])
# True positive rate: when target >= 75, did model predict >= 75?
crit_mask = sub1['target_t1'] >= 75
if crit_mask.sum() > 0:
    tpr = (sub1.loc[crit_mask, 'pred_t1'] >= 75).mean()
    # False positive rate: when target < 75, did model wrongly predict >= 75?
    fpr = (sub1.loc[~crit_mask, 'pred_t1'] >= 75).mean()
    print(f"\nT+1h Critical Zone Detection (threshold >= 75):")
    print(f"  Total actual critical cell-hours: {crit_mask.sum():,}")
    print(f"  Recall    (caught):   {tpr*100:.1f}%")
    print(f"  Precision (no false alarms): ", end="")
    pred_crit = sub1['pred_t1'] >= 75
    if pred_crit.sum() > 0:
        prec = (sub1.loc[pred_crit, 'target_t1'] >= 75).mean()
        print(f"{prec*100:.1f}%  ({pred_crit.sum():,} predicted critical)")
    else:
        print("0% (model never predicted critical)")
    print(f"  False Positive Rate:  {fpr*100:.1f}%")
