import streamlit as st
import pandas as pd
import os
import sys
from datetime import datetime

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROCESSED_DATA_PATH
from src.patrol_optimizer import get_h3_to_station_map

# 1. Page Config
st.set_page_config(
    page_title="GridLock IQ",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Cache Data Loading for uvicorn/streamlit speed
PROCESSED_DIR = os.path.dirname(PROCESSED_DATA_PATH)
CLUSTERED_ZONES_PATH = os.path.join(PROCESSED_DIR, "h3_clustered_zones.parquet")
FORECAST_RESULTS_PATH = os.path.join(PROCESSED_DIR, "forecast_results.parquet")
REPEAT_OFFENDERS_PATH = os.path.join(PROCESSED_DIR, "repeat_offenders.parquet")

def get_data_file_versions():
    """Build a cache key from data file modification times."""
    paths = (
        PROCESSED_DATA_PATH,
        CLUSTERED_ZONES_PATH,
        FORECAST_RESULTS_PATH,
        REPEAT_OFFENDERS_PATH,
    )
    return tuple((path, os.path.getmtime(path) if os.path.exists(path) else None) for path in paths)

@st.cache_data
def load_all_data(data_file_versions):
    _ = data_file_versions
    df_violations = None
    df_clustered_zones = None
    df_forecast = None
    df_repeat_offenders = None
    station_map = {}
    
    if os.path.exists(PROCESSED_DATA_PATH):
        df_violations = pd.read_parquet(PROCESSED_DATA_PATH)
        # Parse timestamp column
        df_violations['created_datetime'] = pd.to_datetime(df_violations['created_datetime'])
        station_map = get_h3_to_station_map(df_violations)
        
    if os.path.exists(CLUSTERED_ZONES_PATH):
        df_clustered_zones = pd.read_parquet(CLUSTERED_ZONES_PATH)
        
    if os.path.exists(FORECAST_RESULTS_PATH):
        df_forecast = pd.read_parquet(FORECAST_RESULTS_PATH)
        df_forecast['hour_dt'] = pd.to_datetime(df_forecast['hour_dt'])
        
    if os.path.exists(REPEAT_OFFENDERS_PATH):
        df_repeat_offenders = pd.read_parquet(REPEAT_OFFENDERS_PATH)
        
    return df_violations, df_clustered_zones, df_forecast, df_repeat_offenders, station_map

if st.sidebar.button("Reload Data", key="reload_data_cache", use_container_width=True):
    load_all_data.clear()
    st.rerun()

# Load datasets into memory
with st.spinner("Initializing GridLock IQ Intelligence Engine..."):
    df_violations, df_clustered_zones, df_forecast, df_repeat_offenders, station_map = load_all_data(get_data_file_versions())

# Store in session state for page components
st.session_state['df_violations'] = df_violations
st.session_state['df_clustered_zones'] = df_clustered_zones
st.session_state['df_forecast'] = df_forecast
st.session_state['df_repeat_offenders'] = df_repeat_offenders
st.session_state['station_map'] = station_map

# 3. Time Selection Controller in Sidebar
st.sidebar.image("https://img.icons8.com/nolan/128/traffic-light.png", width=64)
st.sidebar.markdown("<h2 style='font-family: Outfit; font-weight: 700; color: #ffffff;'>GridLock IQ</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color: #9ca3af; font-size: 13px;'>Predictive Parking Intelligence & Resource Optimization</p>", unsafe_allow_html=True)
st.sidebar.markdown("---")

if df_forecast is not None:
    forecast_updated_at = datetime.fromtimestamp(os.path.getmtime(FORECAST_RESULTS_PATH)).strftime("%Y-%m-%d %H:%M:%S")
    latest_hour = df_forecast['hour_dt'].max()
    latest_rows = df_forecast[df_forecast['hour_dt'] == latest_hour]
    latest_max_pred_t1 = float(latest_rows['pred_t1'].max()) if not latest_rows.empty else 0.0
    st.sidebar.caption(
        f"Data loaded: {forecast_updated_at} | latest T+1 max AOI: {latest_max_pred_t1:.2f}"
    )

    # Identify unique date-hours in the forecast parquet
    unique_datetimes = sorted(df_forecast['hour_dt'].unique())
    max_dt = unique_datetimes[-1]
    
    # Enforce default busy hour (2024-03-12 03:00:00 UTC) if available in the dataset
    default_dt = pd.Timestamp("2024-03-12 03:00:00", tz="UTC")
    if default_dt not in unique_datetimes:
        default_dt = max_dt
        
    # Let user select a date
    st.sidebar.subheader("📅 Time Horizon Controller")
    selected_dt = st.sidebar.select_slider(
        "Simulation Target Hour",
        options=unique_datetimes,
        value=default_dt,
        format_func=lambda x: pd.to_datetime(x).strftime("%Y-%m-%d %H:%M")
    )
    st.session_state['selected_time'] = selected_dt
    
    # Retrieve time window from session state or initialize it
    if 'time_window' not in st.session_state:
        st.session_state['time_window'] = 'Single Hour'
    window_opt = st.session_state['time_window']
    
    # Compute timezone-aware ref_ts
    ref_ts = pd.to_datetime(selected_dt)
    if df_forecast['hour_dt'].dt.tz is not None and ref_ts.tzinfo is None:
        ref_ts = ref_ts.tz_localize(df_forecast['hour_dt'].dt.tz)
    elif df_forecast['hour_dt'].dt.tz is None and ref_ts.tzinfo is not None:
        ref_ts = ref_ts.tz_localize(None)
        
    # Calculate start time based on selection
    if window_opt == "Single Hour":
        start_ts = ref_ts
        end_ts = ref_ts + pd.Timedelta(hours=1) - pd.Timedelta(seconds=1)
    elif "24 Hours" in window_opt:
        start_ts = ref_ts - pd.Timedelta(hours=24)
        end_ts = ref_ts
    elif "7 Days" in window_opt:
        start_ts = ref_ts - pd.Timedelta(days=7)
        end_ts = ref_ts
    elif "30 Days" in window_opt:
        start_ts = ref_ts - pd.Timedelta(days=30)
        end_ts = ref_ts
    else: # All Time
        start_ts = df_forecast['hour_dt'].min()
        end_ts = ref_ts
        
    # Filter forecast data
    sub_forecast = df_forecast[(df_forecast['hour_dt'] >= start_ts) & (df_forecast['hour_dt'] <= ref_ts)]
    
    # Handle timezone for raw violations
    tz = df_violations['created_datetime'].dt.tz if df_violations is not None else None
    if tz is not None and start_ts.tzinfo is None:
        start_ts_viols = start_ts.tz_localize(tz)
        end_ts_viols = end_ts.tz_localize(tz)
    elif tz is None and start_ts.tzinfo is not None:
        start_ts_viols = start_ts.tz_localize(None)
        end_ts_viols = end_ts.tz_localize(None)
    else:
        start_ts_viols = start_ts
        end_ts_viols = end_ts
        
    if df_violations is not None:
        sub_violations = df_violations[(df_violations['created_datetime'] >= start_ts_viols) & (df_violations['created_datetime'] <= end_ts_viols)]
    else:
        sub_violations = None
        
    # Perform aggregation if window is larger than single hour
    if window_opt != "Single Hour" and not sub_forecast.empty:
        agg_forecast = sub_forecast.groupby('h3_cell').agg(
            AOI=('AOI', 'mean'),
            pred_t1=('pred_t1', 'mean'),
            pred_t2=('pred_t2', 'mean'),
            pred_t4=('pred_t4', 'mean'),
            latitude=('latitude', 'mean'),
            longitude=('longitude', 'mean'),
            cell_vehicle_mass=('cell_vehicle_mass', 'mean'),
            junction_flag=('junction_flag', 'max'),
            violation_count=('violation_count', 'sum'),
            historical_density=('historical_density', 'mean')
        ).reset_index()
        # Add timestamp to match schema
        agg_forecast['hour_dt'] = ref_ts
    else:
        agg_forecast = sub_forecast.copy()
        
    st.session_state['filtered_forecast'] = agg_forecast
    st.session_state['filtered_violations'] = sub_violations
else:
    st.sidebar.warning("Forecast database not generated yet. Use the training script first.")
    st.session_state['selected_time'] = datetime.now()
    st.session_state['filtered_forecast'] = None
    st.session_state['filtered_violations'] = None

# 4. Multipage Navigation Setup using modern Streamlit Pages API
# Ensure pages folder exists and has corresponding files
pages_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pages")
os.makedirs(pages_dir, exist_ok=True)

# Define pages path mappings
overview_file = os.path.join(pages_dir, "command_Overview.py")
map_file = os.path.join(pages_dir, "Hotspot_Map.py")
predict_file = os.path.join(pages_dir, "Prediction_Engine.py")
optimizer_file = os.path.join(pages_dir, "Enforcement_Optimizer.py")
simulator_file = os.path.join(pages_dir, "Whatif_Simulator.py")

# Create blank files if they don't exist yet (to prevent import crashes)
for f in [overview_file, map_file, predict_file, optimizer_file, simulator_file]:
    if not os.path.exists(f):
        with open(f, "w") as fp:
            fp.write("")

overview_page = st.Page("pages/command_Overview.py", title="Command Overview", icon="📊", default=True)
map_page = st.Page("pages/Hotspot_Map.py", title="Hotspot Intelligence Map", icon="🗺️")
predict_page = st.Page("pages/Prediction_Engine.py", title="Prediction Engine", icon="🔮")
optimizer_page = st.Page("pages/Enforcement_Optimizer.py", title="Enforcement Optimizer", icon="⚡")
simulator_page = st.Page("pages/Whatif_Simulator.py", title="What-If Simulator", icon="🎲")

pg = st.navigation([
    overview_page,
    map_page,
    predict_page,
    optimizer_page,
    simulator_page
])

# Add custom footer in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='font-size: 11px; color: #6b7280; font-family: sans-serif;'>"
    " Flipkart Grid 5.0 | Bengaluru Traffic Police<br/>"
    " <b>Version:</b> 1.0.0 (Production Build)<br/>"
    " <i>Strictly Zero External Data Constraint</i>"
    "</div>",
    unsafe_allow_html=True
)

pg.run()
