import os
import sys
import numpy as np
import pandas as pd
import joblib
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import h3

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROCESSED_DATA_PATH, MODELS_DIR, AOI_SQUASH_K, IEU_ALPHA

# Define predictions parquet path
FORECAST_RESULTS_PATH = os.path.join(os.path.dirname(PROCESSED_DATA_PATH), "forecast_results.parquet")

def build_features(df):
    """
    Ingests cleaned violations dataframe and builds an hourly spatio-temporal feature matrix.
    """
    print("Aggregating violations to hourly cell-level intervals...")
    
    # 1. Floor created_datetime to hourly intervals
    df['hour_dt'] = df['created_datetime'].dt.floor('h')
    
    # 2. Compute static cell aggregations (coordinates, junction rate, vehicle mass)
    print("Calculating static cell properties...")
    cell_stats = df.groupby('h3_cell').agg(
        latitude=('latitude', 'mean'),
        longitude=('longitude', 'mean'),
        cell_vehicle_mass=('vehicle_mass', 'mean'),
        junction_flag=('junction_flag', 'max')
    ).reset_index()
    
    # 3. Aggregate hourly vehicle mass per cell for IEU computation
    # Using mass sum (not AOI_raw sum) avoids double-counting the 3h window
    # that is already embedded in each record's AOI_raw value.
    hourly_agg = df.groupby(['h3_cell', 'hour_dt']).agg(
        hourly_mass=('vehicle_mass', 'sum'),
        violation_count=('id', 'count')
    ).reset_index()

    # 4. Construct complete grid of all cells and all hours to prevent time-series gaps
    print("Constructing dense spatio-temporal grid...")
    cells = df['h3_cell'].unique()
    min_hour = df['hour_dt'].min()
    max_hour = df['hour_dt'].max()
    hours = pd.date_range(min_hour, max_hour, freq='h')

    mux = pd.MultiIndex.from_product([cells, hours], names=['h3_cell', 'hour_dt'])
    grid_df = pd.DataFrame(index=mux).reset_index()

    # Merge hourly aggregations and static cell properties into grid
    grid_df = pd.merge(grid_df, hourly_agg, on=['h3_cell', 'hour_dt'], how='left')
    grid_df['hourly_mass']    = grid_df['hourly_mass'].fillna(0.0)
    grid_df['violation_count'] = grid_df['violation_count'].fillna(0)
    grid_df = pd.merge(grid_df, cell_stats, on='h3_cell', how='left')
    grid_df = grid_df.sort_values(['h3_cell', 'hour_dt']).reset_index(drop=True)

    # 5. Compute IEU at hourly-cell level
    #    3h rolling mass sum on the dense hourly grid (rolling(3) on hourly data = 3h window)
    print("Computing hourly IEU (3h rolling mass, alpha=0.8, smooth-squash)...")
    grid_df['mass_3h_hourly'] = grid_df.groupby('h3_cell')['hourly_mass'].transform(
        lambda x: x.rolling(3, min_periods=1).sum()
    )
    junction_mult = grid_df['junction_flag'].apply(lambda x: 2.0 if x == 1 else 1.0)
    grid_df['IEU_raw'] = junction_mult * (grid_df['mass_3h_hourly'] ** IEU_ALPHA)

    active_raw = grid_df[grid_df['IEU_raw'] > 0]['IEU_raw']
    if len(active_raw) > 0:
        p90_ieu = float(active_raw.quantile(0.90))
        k_ieu   = p90_ieu / np.log(4.0) if p90_ieu > 0 else AOI_SQUASH_K
    else:
        k_ieu = AOI_SQUASH_K
    print(f"Hourly IEU calibration: p90_raw={p90_ieu:.3f}  k={k_ieu:.4f}")
    grid_df['AOI'] = (100.0 * (1.0 - np.exp(-grid_df['IEU_raw'] / k_ieu))).clip(0, 100)
    grid_df = grid_df.drop(columns=['IEU_raw', 'mass_3h_hourly', 'hourly_mass'])
    print("Engineering lag features...")
    grid_df['AOI_lag_1h'] = grid_df.groupby('h3_cell')['AOI'].shift(1).fillna(0.0)
    grid_df['AOI_lag_2h'] = grid_df.groupby('h3_cell')['AOI'].shift(2).fillna(0.0)
    grid_df['AOI_lag_3h'] = grid_df.groupby('h3_cell')['AOI'].shift(3).fillna(0.0)
    grid_df['AOI_lag_4h'] = grid_df.groupby('h3_cell')['AOI'].shift(4).fillna(0.0)
    grid_df['AOI_lag_6h'] = grid_df.groupby('h3_cell')['AOI'].shift(6).fillna(0.0)
    grid_df['AOI_lag_8h'] = grid_df.groupby('h3_cell')['AOI'].shift(8).fillna(0.0)

    # 5b. Spatial neighbor features: k-ring-1 (adjacent H3 cells)
    # Congestion is spatially contagious: a cell about to spike is usually
    # surrounded by cells already elevated. The model previously had zero
    # awareness of what its neighbours were doing.
    print("Engineering spatial neighbor features (k-ring-1)...")
    cells_set = set(grid_df['h3_cell'].unique())
    neighbor_pairs = [
        (cell, nb)
        for cell in cells_set
        for nb in h3.k_ring(cell, 1)
        if nb != cell and nb in cells_set
    ]
    if neighbor_pairs:
        nb_df = pd.DataFrame(neighbor_pairs, columns=['h3_cell', 'neighbor_cell'])
        # Lag-1h lookup: for each neighbor, what was its IEU last hour?
        lag_lookup = (
            grid_df[['h3_cell', 'hour_dt', 'AOI_lag_1h']]
            .rename(columns={'h3_cell': 'neighbor_cell', 'AOI_lag_1h': 'nb_lag1'})
        )
        nb_lags = nb_df.merge(lag_lookup, on='neighbor_cell', how='left')
        nb_agg = (
            nb_lags.groupby(['h3_cell', 'hour_dt'])
            .agg(
                neighbor_mean_lag1=('nb_lag1', 'mean'),
                neighbor_max_lag1 =('nb_lag1', 'max'),
                neighbor_any_critical=('nb_lag1', lambda x: float((x >= 75.0).any()))
            )
            .reset_index()
        )
        grid_df = grid_df.merge(nb_agg, on=['h3_cell', 'hour_dt'], how='left')
    else:
        grid_df['neighbor_mean_lag1']    = 0.0
        grid_df['neighbor_max_lag1']     = 0.0
        grid_df['neighbor_any_critical'] = 0.0
    grid_df['neighbor_mean_lag1']    = grid_df['neighbor_mean_lag1'].fillna(0.0)
    grid_df['neighbor_max_lag1']     = grid_df['neighbor_max_lag1'].fillna(0.0)
    grid_df['neighbor_any_critical'] = grid_df['neighbor_any_critical'].fillna(0.0)
    
    # 6. Feature Engineering: Rolling Statistics
    print("Engineering rolling features (7-day & 14-day windows)...")
    # 7 days = 168 hours
    grid_df['AOI_roll_mean_7d'] = grid_df.groupby('h3_cell')['AOI'].transform(
        lambda x: x.rolling(168, min_periods=1).mean()
    ).fillna(0.0)
    grid_df['AOI_roll_max_7d'] = grid_df.groupby('h3_cell')['AOI'].transform(
        lambda x: x.rolling(168, min_periods=1).max()
    ).fillna(0.0)
    grid_df['AOI_roll_std_7d'] = grid_df.groupby('h3_cell')['AOI'].transform(
        lambda x: x.rolling(168, min_periods=1).std()
    ).fillna(0.0)
    
    # 14 days = 336 hours
    grid_df['AOI_roll_mean_14d'] = grid_df.groupby('h3_cell')['AOI'].transform(
        lambda x: x.rolling(336, min_periods=1).mean()
    ).fillna(0.0)
    grid_df['AOI_roll_max_14d'] = grid_df.groupby('h3_cell')['AOI'].transform(
        lambda x: x.rolling(336, min_periods=1).max()
    ).fillna(0.0)
    
    # 7. Feature Engineering: Cyclic Time Encoded features
    print("Engineering cyclic temporal features...")
    # Extract hour and day of week from hour_dt (handling timezone if present)
    # The timestamps are timezone-aware. Let's make sure we access local hour
    local_hours = grid_df['hour_dt'].dt.hour
    local_days = grid_df['hour_dt'].dt.dayofweek
    
    grid_df['hour_sin'] = np.sin(2 * np.pi * local_hours / 24.0)
    grid_df['hour_cos'] = np.cos(2 * np.pi * local_hours / 24.0)
    grid_df['day_sin'] = np.sin(2 * np.pi * local_days / 7.0)
    grid_df['day_cos'] = np.cos(2 * np.pi * local_days / 7.0)
    grid_df['hour'] = local_hours  # keep raw hour for historical density
    
    # 8. Feature Engineering: Historical Hourly Density
    print("Engineering historical density features...")
    # Calculate average violations in this cell at this specific hour of the day
    hist_density = grid_df.groupby(['h3_cell', 'hour'])['violation_count'].mean().reset_index()
    hist_density.rename(columns={'violation_count': 'historical_density'}, inplace=True)
    grid_df = pd.merge(grid_df, hist_density, on=['h3_cell', 'hour'], how='left')
    grid_df['historical_density'] = grid_df['historical_density'].fillna(0.0)
    
    # 30-day rolling historical density for Scenario A baseline
    # Shifting by 1 day (shift(1)) ensures we only use PAST 30 days of data at that hour-of-day
    grid_df['hist_density_30d'] = grid_df.groupby(['h3_cell', 'hour'])['violation_count'].transform(
        lambda x: x.shift(1).rolling(30, min_periods=1).mean()
    ).fillna(0.0)
    
    # 9. Create Forecasting Targets (Shifted AOI values)
    print("Creating forecasting target horizons (T+1h, T+2h, T+4h)...")
    grid_df['target_t1'] = grid_df.groupby('h3_cell')['AOI'].shift(-1)
    grid_df['target_t2'] = grid_df.groupby('h3_cell')['AOI'].shift(-2)
    grid_df['target_t4'] = grid_df.groupby('h3_cell')['AOI'].shift(-4)
    
    return grid_df

def train_all_models():
    """
    Loads features, splits data chronologically, trains LightGBM models for T+1, T+2, T+4 horizons,
    saves serialized models, and precomputes validation predictions.
    """
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError(f"Cleaned violations parquet not found at {PROCESSED_DATA_PATH}. Run spatial engine first.")
        
    df = pd.read_parquet(PROCESSED_DATA_PATH)
    features_df = build_features(df)

    # Delta targets: IEU change from the current hour to each forecast horizon.
    # Training on delta instead of absolute IEU has one key structural property:
    # a model that predicts delta=0 everywhere EXACTLY matches the naive
    # persistence baseline (pred = current AOI). Any MAE improvement is therefore
    # genuine signal above persistence, not inflation from the 73% zero rows.
    # Tweedie cannot be used (delta is negative during recovery events);
    # Huber loss is used instead — L2 for small changes, L1 for large spikes,
    # which is exactly the right behaviour for a mostly-quiet signal with rare jumps.
    features_df['delta_t1'] = features_df['target_t1'] - features_df['AOI']
    features_df['delta_t2'] = features_df['target_t2'] - features_df['AOI']
    features_df['delta_t4'] = features_df['target_t4'] - features_df['AOI']

    # ── Static cell feature: junction type ordinal ───────────────────────────────
    # Loaded from cell_poi_tags.json (built from provided dataset junction_name column).
    # metro_station=4, market/mall=3, named_junction=2, area/default=1
    # Gives the model structural context: metro cells peak at commute hours,
    # market cells peak daytime/weekends — different from generic junctions.
    poi_path = os.path.join(os.path.dirname(PROCESSED_DATA_PATH), 'cell_poi_tags.json')
    if os.path.exists(poi_path):
        import json as _json
        with open(poi_path, encoding='utf-8') as f:
            _poi = _json.load(f)
        _ord_map = {k: v.get('junction_type_ord', 1) for k, v in _poi.items()}
        features_df['junction_type_ord'] = features_df['h3_cell'].map(_ord_map).fillna(1).astype(float)
    else:
        features_df['junction_type_ord'] = 1.0
    print(f"  junction_type_ord distribution: {features_df['junction_type_ord'].value_counts().to_dict()}")

    # Define features to use for training
    # Standard features designed for T+1h model:
    feature_cols_t1 = [
        'AOI',              # current IEU — the baseline for delta prediction
        'latitude', 'longitude', 'cell_vehicle_mass', 'junction_flag',
        'junction_type_ord',   # metro=4 / market+mall=3 / junction=2 / area=1
        'AOI_lag_1h', 'AOI_lag_2h', 'AOI_lag_3h',
        'AOI_roll_mean_7d', 'AOI_roll_max_7d', 'AOI_roll_std_7d',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
        'historical_density',
        # Spatial: what are the immediate neighbours doing?
        'neighbor_mean_lag1', 'neighbor_max_lag1', 'neighbor_any_critical',
    ]

    # For T+2h model: add lag_4h, lag_6h, roll_mean_14d
    feature_cols_t2 = feature_cols_t1 + ['AOI_lag_4h', 'AOI_lag_6h', 'AOI_roll_mean_14d']

    # For T+4h model: add lag_6h, lag_8h, roll_mean_14d, roll_max_14d
    feature_cols_t4 = feature_cols_t1 + ['AOI_lag_6h', 'AOI_lag_8h', 'AOI_roll_mean_14d', 'AOI_roll_max_14d']

    # Dict of feature columns per horizon
    horizon_features = {
        1: feature_cols_t1,
        2: feature_cols_t2,
        4: feature_cols_t4,
    }
    
    print(f"Total rows in full dense grid: {len(features_df)}")
    
    # Precompute seasonal naive columns on features_df before filtering
    features_df['seasonal_naive_t1'] = features_df.groupby('h3_cell')['AOI'].shift(167).fillna(0.0)
    features_df['seasonal_naive_t2'] = features_df.groupby('h3_cell')['AOI'].shift(166).fillna(0.0)
    features_df['seasonal_naive_t4'] = features_df.groupby('h3_cell')['AOI'].shift(164).fillna(0.0)

    # Train on the full dense grid to avoid target leakage selection bias and model prediction inflation
    train_ready_df = features_df.copy()
    print(f"Using full dense spatio-temporal grid for training: {len(train_ready_df)} rows")
    
    # Chronological split: 80% train, 20% validation based on unique hour timestamps
    unique_hours = np.array(sorted(features_df['hour_dt'].unique()))
    split_idx = int(len(unique_hours) * 0.8)
    split_time = unique_hours[split_idx]
    
    print(f"Chronological Split Time: {split_time}")
    
    # Split the dataset for final model evaluation
    train_mask = train_ready_df['hour_dt'] < split_time
    val_mask = train_ready_df['hour_dt'] >= split_time
    
    train_data = train_ready_df[train_mask]
    val_data = train_ready_df[val_mask]
    
    print(f"Train set rows: {len(train_data)}, Validation set rows: {len(val_data)}")
    
    # Create models directory if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Initialize prediction columns
    features_df['pred_t1'] = 0.0
    features_df['pred_t1_q10'] = 0.0
    features_df['pred_t1_q90'] = 0.0
    features_df['pred_t2'] = 0.0
    features_df['pred_t2_q10'] = 0.0
    features_df['pred_t2_q90'] = 0.0
    features_df['pred_t4'] = 0.0
    features_df['pred_t4_q10'] = 0.0
    features_df['pred_t4_q90'] = 0.0
    
    horizons = {
        1: ('target_t1', 'delta_t1'),
        2: ('target_t2', 'delta_t2'),
        4: ('target_t4', 'delta_t4'),
    }
    
    # ─── Walk-Forward Cross-Validation ───
    print("\n--- Running 5-Fold Walk-Forward Cross-Validation ---")
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    
    cv_metrics_log = {h: [] for h in [1, 2, 4]}
    
    # Walk-forward CV on unique training hours to evaluate Huber models
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(unique_hours)):
        train_hours_set = set(unique_hours[train_idx])
        val_hours_set = set(unique_hours[val_idx])
        
        fold_train = train_ready_df[train_ready_df['hour_dt'].isin(train_hours_set)]
        fold_val = train_ready_df[train_ready_df['hour_dt'].isin(val_hours_set)]
        
        for h, (target_col, delta_col) in horizons.items():
            f_train = fold_train.dropna(subset=[delta_col])
            f_val = fold_val.dropna(subset=[delta_col])
            if len(f_train) == 0 or len(f_val) == 0:
                continue
            
            feat_cols = horizon_features[h]
            X_tr, y_tr = f_train[feat_cols], f_train[delta_col]
            X_vl, y_vl = f_val[feat_cols], f_val[delta_col]
            
            cv_model = LGBMRegressor(
                objective='huber',
                alpha=5.0,
                n_estimators=400,
                learning_rate=0.04,
                max_depth=7,
                num_leaves=63,
                min_child_samples=15,
                reg_alpha=0.05,
                reg_lambda=0.1,
                random_state=42,
                verbose=-1,
                n_jobs=-1
            )
            cv_model.fit(X_tr, y_tr)
            
            preds_delta = cv_model.predict(X_vl)
            preds_ieu = np.clip(f_val['AOI'].values + preds_delta, 0.0, 100.0)
            
            mae = mean_absolute_error(f_val[target_col], preds_ieu)
            r2 = r2_score(f_val[target_col], preds_ieu)
            
            # Recall @ 65 threshold
            is_crit = (f_val[target_col] >= 65.0)
            if is_crit.sum() > 0:
                rec = (is_crit & (preds_ieu >= 65.0)).sum() / is_crit.sum()
            else:
                rec = 0.0
                
            cv_metrics_log[h].append({"mae": float(mae), "r2": float(r2), "recall": float(rec)})
            
    # Calculate and output CV statistics
    cv_results = {}
    for h in [1, 2, 4]:
        maes = [m['mae'] for m in cv_metrics_log[h]]
        r2s = [m['r2'] for m in cv_metrics_log[h]]
        recs = [m['recall'] for m in cv_metrics_log[h]]
        
        cv_results[f"t{h}"] = {
            "mae_mean": float(np.mean(maes)) if maes else 0.0,
            "mae_std": float(np.std(maes)) if maes else 0.0,
            "r2_mean": float(np.mean(r2s)) if r2s else 0.0,
            "r2_std": float(np.std(r2s)) if r2s else 0.0,
            "recall_mean": float(np.mean(recs)) if recs else 0.0,
            "recall_std": float(np.std(recs)) if recs else 0.0
        }
        print(f"T+{h}h Walk-Forward CV across 5 folds:")
        print(f"  MAE:    {cv_results[f't{h}']['mae_mean']:.4f} ± {cv_results[f't{h}']['mae_std']:.4f}")
        print(f"  R2:     {cv_results[f't{h}']['r2_mean']:.4f} ± {cv_results[f't{h}']['r2_std']:.4f}")
        print(f"  Recall: {cv_results[f't{h}']['recall_mean']:.2%} ± {cv_results[f't{h}']['recall_std']:.2%}")
        
    # ─── Final Model Training & Baseline Evaluations ───
    final_metrics = {}
    
    for h, (target_col, delta_col) in horizons.items():
        print(f"\n--- Training Final LightGBM Model for Horizon T+{h}h ({delta_col}) ---")

        # Drop rows where target is NaN (boundary of training period)
        train_h = train_data.dropna(subset=[delta_col])
        val_h   = val_data.dropna(subset=[delta_col])

        feat_cols = horizon_features[h]
        X_train, y_train_delta = train_h[feat_cols], train_h[delta_col]
        X_val = val_h[feat_cols]
        y_val_ieu  = val_h[target_col]   # evaluate on absolute IEU for comparability
        y_val_aoi  = val_h['AOI']        # current AOI for reconstruction

        # Train main Huber Model
        model = LGBMRegressor(
            objective='huber',
            alpha=5.0,
            n_estimators=400,
            learning_rate=0.04,
            max_depth=7,
            num_leaves=63,
            min_child_samples=15,
            reg_alpha=0.05,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1,
            n_jobs=-1
        )
        model.fit(X_train, y_train_delta, eval_set=[(X_val, val_h[delta_col])])

        # Train Quantile 10 Model
        q10_model = LGBMRegressor(
            objective='quantile',
            alpha=0.1,
            n_estimators=400,
            learning_rate=0.04,
            max_depth=7,
            num_leaves=63,
            min_child_samples=15,
            reg_alpha=0.05,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1,
            n_jobs=-1
        )
        q10_model.fit(X_train, y_train_delta)

        # Train Quantile 90 Model
        q90_model = LGBMRegressor(
            objective='quantile',
            alpha=0.9,
            n_estimators=400,
            learning_rate=0.04,
            max_depth=7,
            num_leaves=63,
            min_child_samples=15,
            reg_alpha=0.05,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1,
            n_jobs=-1
        )
        q90_model.fit(X_train, y_train_delta)

        # Reconstruct IEU from delta predictions
        delta_preds = model.predict(X_val)
        val_preds_ieu = np.clip(y_val_aoi.values + delta_preds, 0.0, 100.0)

        delta_q10_preds = q10_model.predict(X_val)
        val_preds_q10 = np.clip(y_val_aoi.values + delta_q10_preds, 0.0, 100.0)

        delta_q90_preds = q90_model.predict(X_val)
        val_preds_q90 = np.clip(y_val_aoi.values + delta_q90_preds, 0.0, 100.0)

        # Force order: Q10 <= Huber <= Q90
        val_preds_q10 = np.minimum(val_preds_q10, val_preds_ieu)
        val_preds_q90 = np.maximum(val_preds_q90, val_preds_ieu)

        # Evaluate Baselines
        # 1. Persistence
        persistence_preds = np.clip(y_val_aoi.values, 0.0, 100.0)
        mae_persist = mean_absolute_error(y_val_ieu, persistence_preds)

        # 2. Seasonal Naive
        val_sn_preds = val_h[f'seasonal_naive_t{h}'].values
        mae_sn = mean_absolute_error(y_val_ieu, val_sn_preds)

        # 3. Historical Mean
        hist_mean_df = train_h.groupby(['h3_cell', 'hour'])['AOI'].mean().reset_index()
        hist_mean_df.rename(columns={'AOI': 'hist_mean_pred'}, inplace=True)
        val_with_hm = val_h.merge(hist_mean_df, on=['h3_cell', 'hour'], how='left')
        val_hm_preds = val_with_hm['hist_mean_pred'].fillna(0.0).values
        mae_hm = mean_absolute_error(y_val_ieu, val_hm_preds)

        # Final Huber Metrics
        mae  = mean_absolute_error(y_val_ieu, val_preds_ieu)
        rmse = root_mean_squared_error(y_val_ieu, val_preds_ieu)
        r2   = r2_score(y_val_ieu, val_preds_ieu)

        # Recall @ 65
        is_critical = (y_val_ieu >= 65.0)
        if is_critical.sum() > 0:
            recall = (is_critical & (val_preds_ieu >= 65.0)).sum() / is_critical.sum()
            precision = (is_critical & (val_preds_ieu >= 65.0)).sum() / max(1, (val_preds_ieu >= 65.0).sum())
        else:
            recall, precision = 0.0, 0.0

        print(f"Validation Metrics for T+{h}h (IEU scale):")
        print(f"  MAE:          {mae:.4f}")
        print(f"  Persistence:  {mae_persist:.4f} (lift: {mae_persist - mae:+.4f})")
        print(f"  Seas Naive:   {mae_sn:.4f} (lift: {mae_sn - mae:+.4f})")
        print(f"  Hist Mean:    {mae_hm:.4f} (lift: {mae_hm - mae:+.4f})")
        print(f"  RMSE:         {rmse:.4f}")
        print(f"  R2:           {r2:.4f}")
        print(f"  Recall @65:   {recall:.2%}")
        print(f"  Precision:    {precision:.2%}")

        final_metrics[f"t{h}"] = {
            "mae": float(mae), "rmse": float(rmse), "r2": float(r2), 
            "recall": float(recall), "precision": float(precision),
            "persist_mae": float(mae_persist), 
            "seasonal_naive_mae": float(mae_sn), 
            "historical_mean_mae": float(mae_hm)
        }
        
        # Save Huber, Q10, and Q90 models
        joblib.dump(model, os.path.join(MODELS_DIR, f"lgbm_t{h}.pkl"))
        joblib.dump(q10_model, os.path.join(MODELS_DIR, f"lgbm_t{h}_q10.pkl"))
        joblib.dump(q90_model, os.path.join(MODELS_DIR, f"lgbm_t{h}_q90.pkl"))
        print(f"Saved T+{h}h models to {MODELS_DIR}")

        # Compute SHAP Importance for T+1h model
        if h == 1:
            print("Computing SHAP feature importance for T+1h model...")
            import shap
            import json as _json
            X_val_sample = X_val.sample(n=min(1000, len(X_val)), random_state=42)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_val_sample)
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            shap_importance = dict(zip(X_val.columns, [float(v) for v in mean_abs_shap]))
            shap_path = os.path.join(MODELS_DIR, 'shap_importance.json')
            with open(shap_path, 'w') as sf:
                _json.dump(shap_importance, sf, indent=4)
            print(f"Saved SHAP values to {shap_path}")

            importance = pd.DataFrame({
                'feature': feat_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            print("\nTop 5 Important Features:")
            print(importance.head(5).to_string(index=False))

        # Inference: predict delta for all rows, reconstruct IEU + Quantiles
        delta_preds_all = model.predict(features_df[feat_cols])
        delta_q10_preds_all = q10_model.predict(features_df[feat_cols])
        delta_q90_preds_all = q90_model.predict(features_df[feat_cols])
        
        pred_abs = np.clip(features_df['AOI'] + delta_preds_all, 0.0, 100.0)
        pred_q10_abs = np.clip(features_df['AOI'] + delta_q10_preds_all, 0.0, 100.0)
        pred_q90_abs = np.clip(features_df['AOI'] + delta_q90_preds_all, 0.0, 100.0)
        
        # Enforce ordering bounds
        pred_q10_abs = np.minimum(pred_q10_abs, pred_abs)
        pred_q90_abs = np.maximum(pred_q90_abs, pred_abs)

        features_df[f'pred_t{h}'] = pred_abs
        features_df[f'pred_t{h}_q10'] = pred_q10_abs
        features_df[f'pred_t{h}_q90'] = pred_q90_abs
    
    # Save the combined cross-validation and final metrics to json
    combined_metrics = {
        "cv": cv_results,
        "cv_folds": cv_metrics_log,
        "final": final_metrics
    }
    import json as _json
    with open(os.path.join(MODELS_DIR, 'cross_val_metrics.json'), 'w') as mf:
        _json.dump(combined_metrics, mf, indent=4)
    print(f"Saved validation and cross-validation metrics to cross_val_metrics.json")
    
    # Save the complete cell-hour grid with predictions for uvicorn/streamlit.
    print(f"\nSaving precomputed predictions to: {FORECAST_RESULTS_PATH}...")
    output_df = features_df.copy()
    output_df.to_parquet(FORECAST_RESULTS_PATH, index=False, engine='pyarrow')
    print(f"Saved {len(output_df)} rows of prediction results.")
    
    print("\nTraining completed successfully!")
    return final_metrics

if __name__ == '__main__':
    train_all_models()
