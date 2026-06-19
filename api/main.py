import os
import sys
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from datetime import datetime

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROCESSED_DATA_PATH
from src.patrol_optimizer import get_h3_to_station_map, get_dispatch_alerts, optimize_patrol_allocations

app = FastAPI(
    title="GridLock IQ API",
    description="Spatio-Temporal Parking Intelligence & Predictive Resource Optimization Engine Backend",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths to processed datasets
PROCESSED_DIR = os.path.dirname(PROCESSED_DATA_PATH)
CLUSTERED_ZONES_PATH = os.path.join(PROCESSED_DIR, "h3_clustered_zones.parquet")
FORECAST_RESULTS_PATH = os.path.join(PROCESSED_DIR, "forecast_results.parquet")
REPEAT_OFFENDERS_PATH = os.path.join(PROCESSED_DIR, "repeat_offenders.parquet")

# Global variables to cache dataframes for sub-millisecond response times
df_violations = None
df_clustered_zones = None
df_forecast = None
df_repeat_offenders = None
station_map = None
max_timestamp = None

def load_data():
    global df_violations, df_clustered_zones, df_forecast, df_repeat_offenders, station_map, max_timestamp
    print("Loading datasets into API memory cache...")
    
    if os.path.exists(PROCESSED_DATA_PATH):
        df_violations = pd.read_parquet(PROCESSED_DATA_PATH)
        station_map = get_h3_to_station_map(df_violations)
    else:
        print(f"Warning: Violations file not found at {PROCESSED_DATA_PATH}")
        
    if os.path.exists(CLUSTERED_ZONES_PATH):
        df_clustered_zones = pd.read_parquet(CLUSTERED_ZONES_PATH)
    else:
        print(f"Warning: Clustered zones file not found at {CLUSTERED_ZONES_PATH}")
        
    if os.path.exists(FORECAST_RESULTS_PATH):
        df_forecast = pd.read_parquet(FORECAST_RESULTS_PATH)
        # Identify the maximum available timestamp for prediction defaulting
        max_timestamp = df_forecast['hour_dt'].max()
        print(f"Dataset max prediction timestamp: {max_timestamp}")
    else:
        print(f"Warning: Forecast results file not found at {FORECAST_RESULTS_PATH}")
        
    if os.path.exists(REPEAT_OFFENDERS_PATH):
        df_repeat_offenders = pd.read_parquet(REPEAT_OFFENDERS_PATH)
    else:
        print(f"Warning: Repeat offenders file not found at {REPEAT_OFFENDERS_PATH}")

@app.on_event("startup")
async def startup_event():
    load_data()

def get_target_time(timestamp_str: str = None) -> datetime:
    """Helper to parse query timestamp or fallback to the latest available timestamp."""
    global max_timestamp
    if timestamp_str:
        try:
            return pd.to_datetime(timestamp_str)
        except Exception:
            raise HTTPException(status_code=400, detail=f"Invalid timestamp format: {timestamp_str}")
    if max_timestamp is not None:
        return max_timestamp
    return pd.to_datetime(datetime.now())

@app.get("/")
async def root():
    return {
        "app": "GridLock IQ API",
        "status": "healthy",
        "data_cached": {
            "violations": df_violations is not None,
            "clustered_zones": df_clustered_zones is not None,
            "forecast": df_forecast is not None,
            "repeat_offenders": df_repeat_offenders is not None
        }
    }

@app.get("/hotspots")
async def get_hotspots(
    limit: int = Query(10, description="Number of hotspots to return"),
    timestamp: str = Query(None, description="ISO timestamp for query")
):
    """Retrieve top cells by current AOI score."""
    if df_forecast is None:
        raise HTTPException(status_code=500, detail="Forecast dataset not loaded.")
        
    target_time = get_target_time(timestamp)
    
    # Ensure timezone awareness matches the dataset
    if df_forecast['hour_dt'].dt.tz is not None and target_time.tzinfo is None:
        target_time = target_time.tz_localize(df_forecast['hour_dt'].dt.tz)
        
    sub_df = df_forecast[df_forecast['hour_dt'] == target_time]
    
    if sub_df.empty:
        # Fallback: find closest hourly timestamp
        closest_time = df_forecast.iloc[(df_forecast['hour_dt'] - target_time).abs().argsort()[:1]]['hour_dt'].values[0]
        sub_df = df_forecast[df_forecast['hour_dt'] == closest_time]
        target_time = closest_time
        
    top_zones = sub_df.sort_values('AOI', ascending=False).head(limit)
    
    results = []
    for _, row in top_zones.iterrows():
        cell = row['h3_cell']
        info = station_map.get(cell, {'police_station': 'UNKNOWN', 'center_code': -1.0})
        results.append({
            "h3_cell": cell,
            "latitude": row['latitude'],
            "longitude": row['longitude'],
            "aoi": float(row['AOI']),
            "violation_count": int(row['violation_count']),
            "police_station": info['police_station'],
            "pred_t1": float(row['pred_t1']),
            "pred_t2": float(row['pred_t2']),
            "pred_t4": float(row['pred_t4']),
            "timestamp": str(target_time)
        })
        
    return {"timestamp": str(target_time), "hotspots": results}

@app.get("/predict")
async def get_prediction(
    h3_cell: str = Query(..., description="Uber H3 Cell ID"),
    timestamp: str = Query(None, description="ISO timestamp for reference")
):
    """Get T+1h, T+2h, T+4h predictions per cell and a 24h trend forecast."""
    if df_forecast is None:
        raise HTTPException(status_code=500, detail="Forecast dataset not loaded.")
        
    target_time = get_target_time(timestamp)
    if df_forecast['hour_dt'].dt.tz is not None and target_time.tzinfo is None:
        target_time = target_time.tz_localize(df_forecast['hour_dt'].dt.tz)
        
    # Find prediction row for cell and hour
    cell_preds = df_forecast[df_forecast['h3_cell'] == h3_cell]
    if cell_preds.empty:
        raise HTTPException(status_code=404, detail=f"Cell {h3_cell} not found in predictions dataset.")
        
    specific_row = cell_preds[cell_preds['hour_dt'] == target_time]
    
    if specific_row.empty:
        # Fallback to closest
        specific_row = cell_preds.iloc[(cell_preds['hour_dt'] - target_time).abs().argsort()[:1]]
        target_time = specific_row['hour_dt'].values[0]
        
    row = specific_row.iloc[0]
    info = station_map.get(h3_cell, {'police_station': 'UNKNOWN', 'center_code': -1.0})
    
    # Extract 24-hour historical + forecast trend (12h before to 12h after)
    trend_start = target_time - pd.Timedelta(hours=12)
    trend_end = target_time + pd.Timedelta(hours=12)
    
    trend_df = cell_preds[(cell_preds['hour_dt'] >= trend_start) & (cell_preds['hour_dt'] <= trend_end)].sort_values('hour_dt')
    
    trend_data = []
    for _, t_row in trend_df.iterrows():
        trend_data.append({
            "timestamp": str(t_row['hour_dt']),
            "actual_aoi": float(t_row['AOI']),
            "pred_t1": float(t_row['pred_t1']),
            "pred_t2": float(t_row['pred_t2']),
            "pred_t4": float(t_row['pred_t4'])
        })
        
    return {
        "h3_cell": h3_cell,
        "police_station": info['police_station'],
        "center_code": float(info['center_code']),
        "latitude": float(row['latitude']),
        "longitude": float(row['longitude']),
        "reference_time": str(target_time),
        "current_aoi": float(row['AOI']),
        "forecast": {
            "t1": float(row['pred_t1']),
            "t2": float(row['pred_t2']),
            "t4": float(row['pred_t4'])
        },
        "trend_24h": trend_data
    }

@app.get("/patrol-schedule")
async def get_patrol_schedule(
    timestamp: str = Query(None, description="ISO timestamp for scheduling"),
    total_officers: int = Query(150, description="Total officers/towing units to allocate")
):
    """Retrieve optimized patrol schedules and What-If comparison for the given time."""
    if df_forecast is None or station_map is None:
        raise HTTPException(status_code=500, detail="Forecast or Station Map not loaded.")
        
    target_time = get_target_time(timestamp)
    
    _, schedule_df, summary = optimize_patrol_allocations(
        df_forecast, target_time, station_map, total_officers=total_officers
    )
    
    if schedule_df.empty:
        return {"timestamp": str(target_time), "schedule": [], "summary": summary}
        
    schedule_list = schedule_df.to_dict('records')
    return {
        "timestamp": str(target_time),
        "summary": summary,
        "schedule": schedule_list
    }

@app.get("/zone/{id}")
async def get_zone_details(id: str):
    """Fetch details card metrics for a specific H3 cell."""
    if df_clustered_zones is None or df_violations is None:
        raise HTTPException(status_code=500, detail="Clustered zones or Violations dataset not loaded.")
        
    cell_info = df_clustered_zones[df_clustered_zones['h3_cell'] == id]
    if cell_info.empty:
        # Fallback: check violations dataset directly
        cell_violations = df_violations[df_violations['h3_cell'] == id]
        if cell_violations.empty:
            raise HTTPException(status_code=404, detail=f"H3 Zone {id} not found.")
            
        # Re-aggregate dynamically
        lat, lng = h3.h3_to_geo(id)
        max_aoi = float(cell_violations['AOI'].max())
        mean_aoi = float(cell_violations['AOI'].mean())
        count = len(cell_violations)
        dominant_vehicle = cell_violations['clean_vehicle_type'].mode()[0] if not cell_violations['clean_vehicle_type'].mode().empty else 'CAR'
        cluster_lbl = -1
        junction_rate = float(cell_violations['junction_flag'].mean())
    else:
        row = cell_info.iloc[0]
        max_aoi = float(row['max_aoi'])
        mean_aoi = float(row['mean_aoi'])
        count = int(row['violation_count'])
        dominant_vehicle = row['dominant_vehicle_type']
        cluster_lbl = int(row['cluster_label'])
        lat, lng = float(row['latitude']), float(row['longitude'])
        junction_rate = float(row['junction_rate'])
        
    # Get BTP Station
    info = station_map.get(id, {'police_station': 'UNKNOWN', 'center_code': -1.0})
    
    # Calculate Enforcement ROI for this specific cell
    # Formula: approved_violations / (approved + rejected) * 100
    cell_viols = df_violations[df_violations['h3_cell'] == id]
    approved = len(cell_viols[cell_viols['validation_status'] == 'approved'])
    rejected = len(cell_viols[cell_viols['validation_status'] == 'rejected'])
    total_validated = approved + rejected
    roi = (approved / total_validated * 100.0) if total_validated > 0 else 75.0  # default/median fallback
    
    # Risk Tier classification
    risk_tier = 'Low'
    if max_aoi >= 75.0:
        risk_tier = 'Critical'
    elif max_aoi >= 50.0:
        risk_tier = 'High'
    elif max_aoi >= 25.0:
        risk_tier = 'Moderate'
        
    # Fetch recent violation logs for details table
    recent_logs = cell_viols.sort_values('created_datetime', ascending=False).head(5)
    logs_list = []
    for _, l in recent_logs.iterrows():
        logs_list.append({
            "id": l['id'],
            "created_at": str(l['created_datetime']),
            "vehicle_number": l['vehicle_number'],
            "vehicle_type": l['clean_vehicle_type'],
            "violation_type": l['violation_type'],
            "validation_status": str(l['validation_status'])
        })
        
    return {
        "h3_cell": id,
        "police_station": info['police_station'],
        "center_code": float(info['center_code']),
        "latitude": lat,
        "longitude": lng,
        "max_aoi": max_aoi,
        "mean_aoi": mean_aoi,
        "violation_count": count,
        "dominant_vehicle_type": dominant_vehicle,
        "cluster_label": cluster_lbl,
        "junction_rate": junction_rate,
        "risk_tier": risk_tier,
        "enforcement_roi": roi,
        "recent_violations": logs_list
    }

@app.get("/alerts")
async def get_alerts(timestamp: str = Query(None, description="ISO timestamp for query")):
    """Get active critical dispatch alerts (AOI >= 75)."""
    if df_forecast is None or station_map is None:
        raise HTTPException(status_code=500, detail="Forecast or Station Map not loaded.")
        
    target_time = get_target_time(timestamp)
    alerts = get_dispatch_alerts(df_forecast, target_time, station_map)
    return {"timestamp": str(target_time), "count": len(alerts), "alerts": alerts}

@app.get("/repeat-offenders")
async def get_repeat_offenders(limit: int = Query(50, description="Top N repeat offenders")):
    """Retrieve ranked list of repeat offenders."""
    if df_repeat_offenders is None:
        raise HTTPException(status_code=500, detail="Repeat offenders dataset not loaded.")
        
    top_offenders = df_repeat_offenders.head(limit)
    return {
        "count": len(top_offenders),
        "offenders": top_offenders.to_dict('records')
    }
