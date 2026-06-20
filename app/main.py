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

# 2. Import and run data state loader
from app.data_state import ensure_data_loaded

# Load datasets into memory using shared data state module
with st.spinner("Initializing GridLock IQ Intelligence Engine..."):
    ensure_data_loaded()

# Multipage Navigation Setup using modern Streamlit Pages API
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

# 3. Sidebar Redesign (Dynamic Custom CSS & HTML)
st.markdown("""
<style>
/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #0d0f18 !important;
    border-right: 1px solid rgba(255,255,255,0.07);
}

/* Remove default padding */
[data-testid="stSidebar"] > div:first-child {
    padding-top: 0px;
    padding-left: 0px;
    padding-right: 0px;
    padding-bottom: 80px !important;
}

/* Brand block */
.sb-brand {
    padding: 20px 16px 14px 16px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
}
.sb-brand-row {
    display: flex;
    align-items: center;
    gap: 10px;
}
.sb-logo {
    font-size: 22px;
    line-height: 1;
}
.sb-title {
    font-size: 15px;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: 0.3px;
}
.sb-subtitle {
    font-size: 10px;
    color: #6b7280;
    margin-top: 2px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* Status block */
.sb-status {
    padding: 12px 16px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
}
.sb-status-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(0,255,136,0.1);
    border: 1px solid rgba(0,255,136,0.25);
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 10px;
    color: #00ff88;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
}
.sb-status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #00ff88;
    animation: pulse-dot 2s infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}
.sb-data-date {
    font-size: 11px;
    color: #9ca3af;
}
.sb-data-date span {
    color: #ffffff;
    font-weight: 600;
}
.sb-peak-ieu {
    font-size: 11px;
    color: #9ca3af;
    margin-top: 4px;
}
.sb-peak-ieu span {
    color: #ff4444;
    font-weight: 700;
}

/* Navigation section */
.sb-nav-label {
    padding: 14px 16px 6px 16px;
    font-size: 9px;
    font-weight: 700;
    color: #4b5563;
    letter-spacing: 1.2px;
    text-transform: uppercase;
}

/* Hide native navigation */
[data-testid="stSidebarNav"] {
    display: none !important;
}

/* Style page links in sidebar */
[data-testid="stSidebar"] [data-testid="stPageLink"] {
    padding: 0 !important;
    margin: 0 !important;
    z-index: 1;
}

[data-testid="stSidebar"] [data-testid="stPageLink"] a {
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    padding: 10px 16px !important;
    margin: 2px 8px !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    font-size: 13px !important;
    color: #9ca3af !important;
    background-color: transparent !important;
    border: 1px solid transparent !important;
    transition: all 0.15s ease !important;
    text-decoration: none !important;
}

[data-testid="stSidebar"] [data-testid="stPageLink"] a:hover {
    background-color: rgba(255,255,255,0.05) !important;
    color: #ffffff !important;
}

[data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"] {
    background-color: rgba(79,142,247,0.15) !important;
    color: #4f8ef7 !important;
    font-weight: 600 !important;
    border: 1px solid rgba(79,142,247,0.25) !important;
}

[data-testid="stSidebar"] [data-testid="stPageLink"] a span {
    color: inherit !important;
    font-size: 13px !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stSidebar"] [data-testid="stPageLink"] a [data-testid="stIcon"] {
    font-size: 15px !important;
    width: 20px !important;
    text-align: center !important;
    display: inline-block !important;
    margin-right: 0px !important;
}

/* Divider */
.sb-divider {
    height: 1px;
    background: rgba(255,255,255,0.07);
    margin: 10px 16px;
}

/* Reload button override */
[data-testid="stSidebar"] .stButton > button {
    width: calc(100% - 32px);
    margin: 4px 16px;
    background: transparent;
    border: 1px solid rgba(255,255,255,0.12);
    color: #9ca3af;
    font-size: 12px;
    padding: 8px;
    border-radius: 8px;
    transition: all 0.15s ease;
}
[data-testid="stSidebar"] .stButton > button:hover {
    border-color: rgba(255,255,255,0.25);
    color: #ffffff;
    background: rgba(255,255,255,0.05);
}

/* Footer */
.sb-footer {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    padding: 12px 16px;
    border-top: 1px solid rgba(255,255,255,0.07);
    background: #0d0f18;
}
.sb-footer-line1 {
    font-size: 10px;
    color: #4b5563;
    font-weight: 600;
}
.sb-footer-line2 {
    font-size: 10px;
    color: #374151;
    margin-top: 2px;
}
.sb-footer-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 9px;
    color: #00ff88;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown(f"""
<div class="sb-brand" style="display: flex; flex-direction: column; align-items: center; text-align: center; padding: 24px 16px 16px 16px;">
    <div class="sb-logo" style="font-size: 52px; line-height: 1; margin-bottom: 8px;">🚦</div>
    <div class="sb-title" style="font-size: 18px; font-weight: 800; letter-spacing: 0.5px;">GridLock IQ</div>
    <div class="sb-subtitle" style="font-size: 10px; color: #6b7280; margin-top: 4px; letter-spacing: 1px;">Bengaluru Traffic Police</div>
</div>
""", unsafe_allow_html=True)

if df_forecast is not None:
    # Identify unique date-hours in the forecast parquet
    unique_datetimes = sorted(df_forecast['hour_dt'].unique())
    max_dt = unique_datetimes[-1]
    
    # Enforce default busy hour (2024-03-12 03:00:00 UTC) if available in the dataset
    default_dt = pd.Timestamp("2024-03-12 03:00:00", tz="UTC")
    if default_dt not in unique_datetimes:
        default_dt = max_dt
        
    if 'selected_time' not in st.session_state:
        st.session_state['selected_time'] = default_dt
        
    selected_dt = st.session_state['selected_time']
    
    # Compute timezone-aware ref_ts
    ref_ts = pd.to_datetime(selected_dt)
    if df_forecast['hour_dt'].dt.tz is not None and ref_ts.tzinfo is None:
        ref_ts = ref_ts.tz_localize(df_forecast['hour_dt'].dt.tz)
    elif df_forecast['hour_dt'].dt.tz is None and ref_ts.tzinfo is not None:
        ref_ts = ref_ts.tz_localize(None)

    # Calculate peak predicted T+1h IEU dynamically
    df_hour = df_forecast[df_forecast['hour_dt'] == ref_ts]
    peak_ieu = float(df_hour['pred_t1'].max()) if not df_hour.empty else 0.0
    
    # Convert ref_ts to local IST for display
    target_ts_ist = ref_ts.tz_convert("Asia/Kolkata") if ref_ts.tzinfo else ref_ts
    data_date_str = target_ts_ist.strftime("%Y-%m-%d %H:%M")
    
    # Determine IEU color & label
    if peak_ieu >= 75:
        ieu_color = "#ff4444"
        ieu_label = "CRITICAL"
    elif peak_ieu >= 50:
        ieu_color = "#ff8c00"
        ieu_label = "HIGH"
    else:
        ieu_color = "#2ecc71"
        ieu_label = "MODERATE"
        
    # ── Status Block ───────────────────────────────────────────────
    st.sidebar.markdown(f"""
    <div class="sb-status">
        <div class="sb-status-pill">
            <div class="sb-status-dot"></div>
            LIVE
        </div>
        <div class="sb-data-date">
            Snapshot: <span>{data_date_str}</span>
        </div>
        <div class="sb-peak-ieu">
            Peak IEU (T+1h): 
            <span style="color:{ieu_color}">
                {peak_ieu:.1f} · {ieu_label}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ── Navigation ─────────────────────────────────────────────────
    st.sidebar.markdown('<div class="sb-nav-label">Navigation</div>', unsafe_allow_html=True)
    st.sidebar.page_link(overview_page, label="Command Overview", icon="📊")
    st.sidebar.page_link(map_page, label="Hotspot Map", icon="🗺️")
    st.sidebar.page_link(predict_page, label="Prediction Engine", icon="🔮")
    st.sidebar.page_link(optimizer_page, label="Enforcement Optimizer", icon="⚡")
    st.sidebar.page_link(simulator_page, label="What-If Simulator", icon="🎲")
    
    # ── Divider + Reload ───────────────────────────────────────────
    st.sidebar.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    
    if st.sidebar.button("🔄  Reload Data"):
        load_all_data.clear()
        st.rerun()
        
    # ── Footer ─────────────────────────────────────────────────────
    st.sidebar.markdown(f"""
    <div class="sb-footer">
        <div class="sb-footer-line1">
            Flipkart GridLock 2.0 · v1.0.0
        </div>
        <div class="sb-footer-line2">
            Production Build · Mar 2024
        </div>
        <div class="sb-footer-badge">
            ✓ Zero External Data
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Retrieve time window from session state or initialize it
    if 'time_window' not in st.session_state:
        st.session_state['time_window'] = 'Single Hour'
    window_opt = st.session_state['time_window']
    
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

# Navigation already initialized at top

# Running Streamlit Page Router

pg.run()
