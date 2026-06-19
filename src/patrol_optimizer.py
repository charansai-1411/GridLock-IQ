import os
import sys
import numpy as np
import pandas as pd
import math

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROCESSED_DATA_PATH

def get_h3_to_station_map(df):
    """
    Builds a data-driven mapping from H3 cell IDs to BTP police stations and center codes.
    Uses the mode (most frequent) station associated with each cell.
    """
    print("Building H3-to-Station mapping from data...")
    # Group by cell and get mode of police_station and center_code
    cell_station = df.groupby('h3_cell').agg(
        police_station=('police_station', lambda x: x.mode()[0] if not x.mode().empty else 'UNKNOWN'),
        center_code=('center_code', lambda x: x.mode()[0] if not x.mode().empty else -1.0)
    ).reset_index()
    return cell_station.set_index('h3_cell').to_dict('index')

def get_dispatch_alerts(pred_df, current_time, station_map, cell_vehicle_map=None):
    """
    Generates dispatch alerts for cells where predicted IEU at T+1h >= 65
    (F2-optimal threshold: recall 87.1%, precision 71.1%).

    Returns list of dicts with keys:
      h3_cell, aoi, police_station, dominant_vehicle,
      violation_count, units_needed, severity, message
    """
    ts = pd.to_datetime(current_time)
    if pred_df['hour_dt'].dt.tz is not None and ts.tzinfo is None:
        ts = ts.tz_localize(pred_df['hour_dt'].dt.tz)
    elif pred_df['hour_dt'].dt.tz is None and ts.tzinfo is not None:
        ts = ts.tz_localize(None)

    hour_preds = pred_df[pred_df['hour_dt'] == ts]

    # Fallback: if exact hour has no data, use the closest available hour
    if hour_preds.empty:
        available = pred_df['hour_dt'].unique()
        if len(available) == 0:
            return []
        closest = available[np.argmin(np.abs(available - ts))]
        hour_preds = pred_df[pred_df['hour_dt'] == closest]

    alerts = []
    # Use F2-optimal threshold (65) — recall 87.1%, precision 71.1%
    alert_zones = hour_preds[hour_preds['pred_t1'] >= 65.0].copy()

    for _, row in alert_zones.iterrows():
        cell        = row['h3_cell']
        pred_aoi    = float(row['pred_t1'])
        current_aoi = float(row.get('AOI', 0.0))
        viol_count  = int(row.get('violation_count', 0))

        station_info = station_map.get(cell, {})
        if isinstance(station_info, dict):
            station = station_info.get('police_station', 'Unknown')
        else:
            station = str(station_info)

        vehicle = (cell_vehicle_map.get(cell, 'CAR') if cell_vehicle_map else
                   row.get('dominant_vehicle_type', 'CAR'))

        if pred_aoi >= 80:   severity = 'Critical'
        elif pred_aoi >= 65: severity = 'High'
        else:                severity = 'Moderate'

        units = max(math.ceil(pred_aoi / 25.0), 1)

        alerts.append({
            'h3_cell':          cell,
            'aoi':              pred_aoi,           # predicted IEU score
            'police_station':   station,
            'dominant_vehicle': vehicle,
            'violation_count':  viol_count,
            'units_needed':     units,
            'severity':         severity,
            'current_aoi':      current_aoi,
            # Legacy fields kept for backward compat
            'predicted_aoi':    pred_aoi,
            'primary_violator': vehicle,
            'station':          station,
            'recommended_units': units,
        })

    return sorted(alerts, key=lambda x: x['aoi'], reverse=True)



def optimize_patrol_allocations(pred_df, current_time, station_map, total_officers=150):
    """
    Greedy Spatio-Temporal Patrol Dispatch Optimizer.
    Allocates available officers to active cells based on T+1h predictions.
    
    Scenario A (Baseline): Officers allocated proportional to historical overall violation rates.
    Scenario B (AI-Optimized): Officers allocated to cells with the highest predicted T+1h AOI.
    """
    ts = pd.to_datetime(current_time)
    if pred_df['hour_dt'].dt.tz is not None and ts.tzinfo is None:
        ts = ts.tz_localize(pred_df['hour_dt'].dt.tz)
    elif pred_df['hour_dt'].dt.tz is None and ts.tzinfo is not None:
        ts = ts.tz_localize(None)
        
    hour_preds = pred_df[pred_df['hour_dt'] == ts].copy()
    
    if hour_preds.empty:
        return pd.DataFrame(), pd.DataFrame(), {}
        
    # Add station information to hour predictions
    hour_preds['police_station'] = hour_preds['h3_cell'].map(lambda x: station_map.get(x, {}).get('police_station', 'UNKNOWN'))
    hour_preds['center_code'] = hour_preds['h3_cell'].map(lambda x: station_map.get(x, {}).get('center_code', -1.0))
    
    # 1. SCENARIO B: AI-Optimized Deployment using Marginal Utility
    # We allocate officers to maximize total risk reduction.
    # Marginal reduction factor for each officer level:
    #   1st officer: 0.25
    #   2nd officer: 0.20 (0.45 - 0.25)
    #   3rd officer: 0.15 (0.60 - 0.45)
    #   4th officer: 0.10 (0.70 - 0.60)
    # Any officer beyond the 4th yields 0 marginal reduction.
    opt_alloc = {cell: 0 for cell in hour_preds['h3_cell']}
    officers_left = total_officers
    
    active_cells = hour_preds[hour_preds['pred_t1'] >= 25.0].copy()
    if not active_cells.empty and officers_left > 0:
        steps = []
        for idx, row in active_cells.iterrows():
            cell = row['h3_cell']
            r = row['pred_t1']
            steps.append((cell, 1, 0.25 * r))
            steps.append((cell, 2, 0.20 * r))
            steps.append((cell, 3, 0.15 * r))
            steps.append((cell, 4, 0.10 * r))
            
        # Sort steps by marginal gain descending
        steps.sort(key=lambda x: x[2], reverse=True)
        
        # Greedy allocation
        for cell, officer_num, gain in steps:
            if officers_left <= 0:
                break
            opt_alloc[cell] += 1
            officers_left -= 1
            
    hour_preds['opt_officers'] = hour_preds['h3_cell'].map(opt_alloc).fillna(0).astype(int)
    
    # 2. SCENARIO A: Historical/Baseline Deployment
    # Historical allocation is proportional to the historical density of each cell
    # Sum historical density
    sum_hist_density = hour_preds['historical_density'].sum()
    if sum_hist_density > 0:
        base_alloc = (hour_preds['historical_density'] / sum_hist_density * total_officers).round().astype(int)
        # Adjust rounding mismatch to match total_officers
        diff = total_officers - base_alloc.sum()
        if diff != 0:
            # Add/subtract diff from top density cell
            base_alloc.iloc[0] += diff
        hour_preds['base_officers'] = base_alloc
    else:
        hour_preds['base_officers'] = 0
        
    # 3. Simulate Congestion Reduction
    # Reduction curves based on assigned officers:
    # 0 units -> 0%
    # 1 unit  -> 25% reduction
    # 2 units -> 45% reduction
    # 3 units -> 60% reduction
    # 4+ units -> 70% reduction
    def get_reduction(units):
        if units <= 0:
            return 0.0
        elif units == 1:
            return 0.25
        elif units == 2:
            return 0.45
        elif units == 3:
            return 0.60
        else:
            return 0.70
            
    hour_preds['base_aoi_reduction'] = hour_preds['base_officers'].apply(get_reduction)
    hour_preds['opt_aoi_reduction'] = hour_preds['opt_officers'].apply(get_reduction)
    
    hour_preds['base_post_aoi'] = hour_preds['pred_t1'] * (1.0 - hour_preds['base_aoi_reduction'])
    hour_preds['opt_post_aoi'] = hour_preds['pred_t1'] * (1.0 - hour_preds['opt_aoi_reduction'])
    
    # Aggregated metrics
    total_raw_aoi = hour_preds['pred_t1'].sum()
    total_base_aoi = hour_preds['base_post_aoi'].sum()
    total_opt_aoi = hour_preds['opt_post_aoi'].sum()
    
    base_reduction_pct = ((total_raw_aoi - total_base_aoi) / total_raw_aoi * 100.0) if total_raw_aoi > 0 else 0.0
    opt_reduction_pct = ((total_raw_aoi - total_opt_aoi) / total_raw_aoi * 100.0) if total_raw_aoi > 0 else 0.0
    delta_reduction_pct = opt_reduction_pct - base_reduction_pct
    
    opt_active_coverage = (hour_preds['opt_officers'] > 0).sum()
    base_active_coverage = (hour_preds['base_officers'] > 0).sum()
    
    summary_metrics = {
        'total_aoi_before': total_raw_aoi,
        'base_post_aoi': total_base_aoi,
        'opt_post_aoi': total_opt_aoi,
        'base_reduction_pct': base_reduction_pct,
        'opt_reduction_pct': opt_reduction_pct,
        'delta_reduction_pct': delta_reduction_pct,
        'base_coverage_zones': base_active_coverage,
        'opt_coverage_zones': opt_active_coverage,
        'officers_allocated': total_officers - officers_left
    }
    
    # Split schedule for display
    # Generate schedule matrix: station x active zones assigned
    schedule_records = []
    assigned_zones = hour_preds[hour_preds['opt_officers'] > 0]
    for _, row in assigned_zones.iterrows():
        schedule_records.append({
            'police_station': row['police_station'],
            'h3_cell': row['h3_cell'],
            'latitude': row['latitude'],
            'longitude': row['longitude'],
            'predicted_aoi': row['pred_t1'],
            'officers_assigned': row['opt_officers'],
            'risk_tier': 'Critical' if row['pred_t1'] >= 75.0 else 'High' if row['pred_t1'] >= 50.0 else 'Moderate'
        })
        
    schedule_df = pd.DataFrame(schedule_records)
    
    return hour_preds, schedule_df, summary_metrics
