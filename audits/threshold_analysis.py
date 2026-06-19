"""
Threshold optimization for critical-zone detection.
Uses existing pred_t1 predictions — no retraining needed.

Sweeps prediction cutoffs and scores with F-beta (beta=2):
  recall weighted 4x over precision — missing a congestion event
  is operationally worse than a false dispatch.
"""
import pandas as pd
import numpy as np

df = pd.read_parquet('data/processed/forecast_results.parquet')

import pytz
IST = pytz.timezone('Asia/Kolkata')

# Validation set mirror (chronological 80/20 split)
unique_hours = sorted(df['hour_dt'].unique())
split_time   = unique_hours[int(len(unique_hours) * 0.8)]
val = df[df['hour_dt'] >= split_time].dropna(subset=['target_t1']).copy()

# Ground truth: actual critical (IEU >= 75 in the next hour)
y_true = (val['target_t1'] >= 75).astype(int)
scores  = val['pred_t1'].values
n_actual = y_true.sum()

print(f"Validation rows: {len(val):,}")
print(f"Actual critical (target_t1 >= 75): {n_actual:,}  ({n_actual/len(val)*100:.2f}%)")
print()

BETA = 2.0  # recall weighted 4x (beta² = 4)

thresholds = [50, 55, 58, 60, 62, 65, 67, 70, 72, 75, 78, 80]

print(f"{'Threshold':>10} {'Predicted':>10} {'Recall':>8} {'Precision':>10} {'F{:.0f}':>8} {'F1':>8}".format(BETA))
print("-" * 62)

best_thresh, best_fbeta = 75, 0.0
results = []
for t in thresholds:
    y_pred = (scores >= t).astype(int)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fbeta     = (1 + BETA**2) * precision * recall / (BETA**2 * precision + recall) if (precision + recall) > 0 else 0.0
    n_pred    = y_pred.sum()
    fpr       = fp / (y_true == 0).sum()

    results.append({'threshold': t, 'recall': recall, 'precision': precision, 'fbeta': fbeta, 'n_pred': n_pred, 'fpr': fpr})
    marker = " ◄ current" if t == 75 else ("  ← BEST F2" if fbeta > best_fbeta else "")

    if fbeta > best_fbeta:
        best_fbeta  = fbeta
        best_thresh = t

    print(f"{t:>10}  {n_pred:>9,}  {recall:>7.1%}  {precision:>9.1%}  {fbeta:>7.4f}  {f1:>7.4f}  {marker}")

print()
best = [r for r in results if r['threshold'] == best_thresh][0]
print(f"Best F{BETA:.0f} threshold: {best_thresh}  →  recall={best['recall']:.1%}  precision={best['precision']:.1%}  F{BETA:.0f}={best['fbeta']:.4f}")
print(f"FPR at best threshold: {best['fpr']:.3%}")

# Also show: recall>=0.75 with highest precision
above_75_recall = [r for r in results if r['recall'] >= 0.75]
if above_75_recall:
    best_precision_at_75_recall = max(above_75_recall, key=lambda r: r['precision'])
    r = best_precision_at_75_recall
    print(f"\nBest precision while maintaining recall>=75%:")
    print(f"  Threshold={r['threshold']}  recall={r['recall']:.1%}  precision={r['precision']:.1%}  FPR={r['fpr']:.3%}")

print()
print("Recommendation: use threshold", best_thresh, "in production for critical alerting.")
print("The dashboard 'Predicted Tier' badge logic should be updated to flag")
print(f"  pred_t1 >= {best_thresh} as Critical (not >= 75).")
