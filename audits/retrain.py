import sys
sys.path.insert(0, '.')
from src.forecast_engine import train_all_models
metrics = train_all_models()
print()
print("=== FINAL MODEL METRICS ===")
for horizon, m in metrics.items():
    print(f"  T+{horizon}: MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  R2={m['r2']:.4f}")
