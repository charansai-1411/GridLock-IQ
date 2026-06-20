import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import json
import plotly.graph_objects as go
import h3

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.data_state import ensure_data_loaded
from src.patrol_optimizer import optimize_patrol_allocations

# ─── Constants & Mappings ──────────────────────────────────────────────────
CELL_JURISDICTION_LOCATION_MAP = {
    "8760145a2ffffff": ("Shivajinagar", "Safina Plaza Jn"),
    "8761892e9ffffff": ("Vijayanagara", "Hosahalli Metro Stn"),
    "8760145b5ffffff": ("Upparpet", "Upparpet Junction"),
    "876189209ffffff": ("HAL Old Air.", "HAL Airport Road"),
    "876189254ffffff": ("Bellandur", "Bellandur Junction"),
    "8760145b3ffffff": ("Malleshwaram", "Malleshwaram Circle"),
    "87618925bffffff": ("Halasuru", "Halasuru Gate"),
    "8760145a6ffffff": ("Basavanagudi", "Basavanagudi Market"),
    "8760145b0ffffff": ("Malleshwaram", "SP Road Malleshwaram"),
    "876014584ffffff": ("Vijayanagara", "Infantry Road Jn"),
}

@st.cache_data
def _load_poi_tags():
    path = os.path.join(PROJECT_ROOT, "data", "processed", "cell_poi_tags.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}

def _get_cell_coords(cell):
    try:
        lat, lng = h3.h3_to_geo(cell)
        return float(lat), float(lng)
    except Exception:
        return 12.9716, 77.5946

def _get_cell_geojson(cell):
    try:
        verts = h3.h3_to_geo_boundary(cell)
        coords = [[float(lng), float(lat)] for lat, lng in verts]
        coords.append(coords[0]) # close loop
        return coords
    except Exception:
        return []

# ─── Render Page ─────────────────────────────────────────────────────────────
def render_simulator_page():
    # Page setup
    st.set_page_config(layout="wide", page_title="What-If Simulator — GridLock IQ")
    ensure_data_loaded()
    
    df_forecast = st.session_state.get('df_forecast')
    station_map = st.session_state.get('station_map')
    selected_time = st.session_state.get('selected_time')
    
    if df_forecast is None:
        st.error("Error: Datasets failed to load. Please verify your data files.")
        return
        
    # Default snapshot to 2024-03-12 03:00:00 UTC / 08:30 AM IST (peak morning window)
    if selected_time is None:
        selected_time = pd.Timestamp("2024-03-12 03:00:00", tz="UTC")
        st.session_state['selected_time'] = selected_time
        
    target_ts = pd.to_datetime(selected_time)
    if target_ts.tzinfo is None:
        target_ts = target_ts.tz_localize("UTC")
    target_ts_ist = target_ts.tz_convert("Asia/Kolkata")
    
    # Custom CSS style for KPI cards
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@700;800&family=Roboto+Mono:wght@700&display=swap');
        
        .kpi-row {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 25px;
            font-family: 'Inter', sans-serif;
        }
        .kpi-card {
            background-color: #131722;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 8px;
            padding: 16px;
            min-height: 130px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .kpi-title {
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            color: #94a3b8;
            letter-spacing: 0.05em;
            margin-bottom: 8px;
        }
        .kpi-value-container {
            flex-grow: 1;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .kpi-val-line {
            font-size: 1.1rem;
            font-weight: 700;
            color: #e2e8f0;
            margin-bottom: 4px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .kpi-val-big {
            font-family: 'Outfit', sans-serif;
            font-size: 2.2rem;
            font-weight: 800;
            line-height: 1;
            margin-bottom: 8px;
        }
        .kpi-subtitle {
            font-size: 0.8rem;
            color: #64748b;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # ─── Component 1: Header Strip + Controls ────────────────────────────────
    st.markdown(
        """
        <div style="background-color: #0f1117; padding: 12px 16px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.05); margin-bottom: 20px;">
            <h1 style="font-family: 'Outfit', sans-serif; font-weight: 800; color: #ffffff; margin: 0 0 4px 0; font-size: 1.25rem;">
                🎲 What-If Simulator — Deployment Scenario Comparison
            </h1>
            <p style="color: #9ca3af; font-family: 'Inter', sans-serif; font-size: 0.85rem; margin: 0;">
                Compare AI-optimized dispatch vs historical patrol patterns
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Inline controls row using selectboxes (safest, most robust date-time pattern)
    col_date, col_hour, col_slider = st.columns([1.2, 1.0, 2.2])
    
    # Unique times in local IST from metadata
    df_forecast_hours = st.session_state.get("df_forecast_hours")
    all_times_ist = pd.DatetimeIndex(df_forecast_hours['hour_dt'].unique()).tz_convert("Asia/Kolkata")
    unique_dates = sorted(list(set(all_times_ist.date)))
    date_labels = [str(d) for d in unique_dates]
    
    current_date_str = str(target_ts_ist.date())
    default_di = date_labels.index(current_date_str) if current_date_str in date_labels else 0
    
    with col_date:
        sel_date_str = st.selectbox("Date", options=date_labels, index=default_di, key="sim_date_box")
    sel_date = pd.Timestamp(sel_date_str).date()
    
    # Hours available for selected date
    available_hours = sorted(list(set([t.hour for t in all_times_ist if t.date() == sel_date])))
    hour_labels = [f"{h:02d}:00" for h in available_hours]
    
    rounded_hour = (target_ts_ist + pd.Timedelta(minutes=30)).hour
    default_hour_str = f"{rounded_hour:02d}:00"
    default_hi = hour_labels.index(default_hour_str) if default_hour_str in hour_labels else 0
    
    with col_hour:
        sel_hour_str = st.selectbox("Hour", options=hour_labels, index=default_hi, key="sim_hour_box")
    sel_hour = int(sel_hour_str.split(":")[0])
    
    with col_slider:
        total_officers = st.slider("Total officers available this shift", min_value=5, max_value=60, value=30, step=5, key="sim_officers_slider")
        
    # Resolve selected local time to UTC
    sel_ts_local = pd.Timestamp(f"{sel_date_str} {sel_hour:02d}:00:00", tz="Asia/Kolkata")
    sel_ts_utc = sel_ts_local.tz_convert("UTC")
    
    # Align to the nearest hour in df_forecast_hours to prevent empty dataset returns
    all_utc_idx = pd.DatetimeIndex(df_forecast_hours['hour_dt'].unique())
    deltas = np.abs((all_utc_idx - sel_ts_utc).total_seconds())
    nearest_ts = all_utc_idx[int(np.argmin(deltas))]
    
    # Sync back to session state silently and update data if it changed
    if nearest_ts != st.session_state.get('selected_time'):
        st.session_state['selected_time'] = nearest_ts
        from app.data_state import update_filtered_data
        update_filtered_data()
        df_forecast = st.session_state.get('df_forecast')
        
    # Load optimizer data using aligned nearest timestamp
    df_hour, schedule_df, summary = optimize_patrol_allocations(
        df_forecast, nearest_ts, station_map, total_officers=total_officers
    )
    
    if df_hour.empty:
        st.info("No active congestion data for the selected hour to simulate.")
        return
        
    # Get top 8 active zones for analysis
    top_8 = df_hour.sort_values('pred_t1', ascending=False).head(8).copy()
    poi_tags = _load_poi_tags()
    
    # Map locations
    top_8['location'] = top_8['h3_cell'].map(lambda x: CELL_JURISDICTION_LOCATION_MAP.get(x, (None, poi_tags.get(x, {}).get("poi_label", "Zone " + x[:8])))[1])
    top_8['station'] = top_8['h3_cell'].map(lambda x: CELL_JURISDICTION_LOCATION_MAP.get(x, (station_map.get(x, {}).get("police_station", "Unknown"), None))[0])
    
    # ─── Calculations for KPI Cards ──────────────────────────────────────────
    total_critical = (df_hour['pred_t1'] >= 75).sum()
    
    # Card 1: Critical Zones Covered
    crit_cov_A = ((df_hour['pred_t1'] >= 75) & (df_hour['base_officers'] > 0)).sum()
    crit_cov_B = ((df_hour['pred_t1'] >= 75) & (df_hour['opt_officers'] > 0)).sum()
    badge_A = "✅" if crit_cov_A >= total_critical and total_critical > 0 else "❌"
    badge_B = "✅" if crit_cov_B >= total_critical and total_critical > 0 else "❌"
    
    # Card 2: Wasted Unit Hours
    wasted_A = int(df_hour.loc[df_hour['pred_t1'] < 25, 'base_officers'].sum())
    wasted_B = int(df_hour.loc[df_hour['pred_t1'] < 25, 'opt_officers'].sum())
    saved_wasted = wasted_A - wasted_B
    saved_wasted_str = f"+{saved_wasted}" if saved_wasted >= 0 else f"{saved_wasted}"
    
    # Card 3: Highest Risk Zone Status
    highest_row = df_hour.loc[df_hour['pred_t1'].idxmax()]
    highest_cell = highest_row['h3_cell']
    highest_name = CELL_JURISDICTION_LOCATION_MAP.get(highest_cell, (None, poi_tags.get(highest_cell, {}).get("poi_label", "Zone " + highest_cell[:8])))[1]
    
    cov_status_A = "✅ COVERED" if highest_row['base_officers'] > 0 else "❌ MISSED"
    cov_status_B = "✅ COVERED" if highest_row['opt_officers'] > 0 else "❌ MISSED"
    cov_color_A = "#2ecc71" if highest_row['base_officers'] > 0 else "#ff4444"
    cov_color_B = "#2ecc71" if highest_row['opt_officers'] > 0 else "#ff4444"
    
    # Card 4: AI Coverage Efficiency
    ieu_covered_A = (df_hour.loc[df_hour['base_officers'] > 0, 'pred_t1']).sum()
    ieu_covered_B = (df_hour.loc[df_hour['opt_officers'] > 0, 'pred_t1']).sum()
    
    eff_A = ieu_covered_A / total_officers if total_officers > 0 else 0
    eff_B = ieu_covered_B / total_officers if total_officers > 0 else 0
    eff_ratio = eff_B / eff_A if eff_A > 0 else 1.0
    
    # ─── Component 2: KPI Cards Row ──────────────────────────────────────────
    kpis_html = f"""
    <div class="kpi-row">
        <!-- Card 1 -->
        <div class="kpi-card">
            <div class="kpi-title">Critical Zones Covered</div>
            <div class="kpi-value-container">
                <div class="kpi-val-line"><span>A (Baseline):</span> <strong>{crit_cov_A} / {total_critical} &nbsp;{badge_A}</strong></div>
                <div class="kpi-val-line"><span>B (AI-Opt):</span> <strong>{crit_cov_B} / {total_critical} &nbsp;{badge_B}</strong></div>
            </div>
            <div class="kpi-subtitle">towing units dispatched to critical risk</div>
        </div>
        <!-- Card 2 -->
        <div class="kpi-card">
            <div class="kpi-title">Wasted Unit-Hours Saved</div>
            <div class="kpi-value-container">
                <div class="kpi-val-big" style="color: #ff8c00;">{saved_wasted_str}</div>
            </div>
            <div class="kpi-subtitle">unit-hours better vs baseline (IEU &lt; 25)</div>
        </div>
        <!-- Card 3 -->
        <div class="kpi-card">
            <div class="kpi-title">Highest Risk Zone Status</div>
            <div class="kpi-value-container">
                <div class="kpi-val-line" style="font-size:0.95rem;"><span>A:</span> <strong style="color: {cov_color_A};">{cov_status_A}</strong></div>
                <div class="kpi-val-line" style="font-size:0.95rem;"><span>B:</span> <strong style="color: {cov_color_B};">{cov_status_B}</strong></div>
            </div>
            <div class="kpi-subtitle" style="text-overflow:ellipsis; overflow:hidden; white-space:nowrap;">{highest_name}</div>
        </div>
        <!-- Card 4 -->
        <div class="kpi-card">
            <div class="kpi-title">AI Coverage Efficiency</div>
            <div class="kpi-value-container">
                <div class="kpi-val-big" style="color: #4f8ef7;">{eff_ratio:.1f}&times;</div>
            </div>
            <div class="kpi-subtitle">more IEU points covered per unit</div>
        </div>
    </div>
    """
    st.html(kpis_html)

    # ─── Component 3: Scenario Comparison Map ───────────────────────────────
    map_cells = []
    for _, row in df_hour.iterrows():
        cell = row['h3_cell']
        pred_t1 = float(row['pred_t1'])
        curr_ieu = float(row['AOI'])
        base_off = int(row['base_officers'])
        opt_off = int(row['opt_officers'])
        
        if pred_t1 < 1.0 and curr_ieu < 1.0 and base_off == 0 and opt_off == 0:
            continue
            
        lat, lng = _get_cell_coords(cell)
        poly = _get_cell_geojson(cell)
        name = CELL_JURISDICTION_LOCATION_MAP.get(cell, (None, poi_tags.get(cell, {}).get("poi_label", "Zone " + cell[:8])))[1]
        
        map_cells.append({
            "lat": lat,
            "lng": lng,
            "ieu": pred_t1,
            "units_a": base_off,
            "units_b": opt_off,
            "name": name,
            "polygon": poly
        })
        
    map_cells_json = json.dumps(map_cells)
    
    map_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
        <style>
            html, body {{ margin:0; padding:0; height:100%; width:100%; background:#0f1117; font-family: 'Inter', sans-serif; }}
            #map {{ height: 100%; width: 100%; background: #0f1117; }}
            .leaflet-container {{ background: #0f1117 !important; }}
            
            #toggle-bar {{
                position: absolute;
                top: 15px;
                left: 50%;
                transform: translateX(-50%);
                z-index: 1000;
                display: flex;
                background: rgba(15, 17, 23, 0.85);
                backdrop-filter: blur(8px);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 20px;
                padding: 3px;
                box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
            }}
            .toggle-btn {{
                padding: 6px 16px;
                border-radius: 17px;
                font-size: 0.75rem;
                font-weight: 700;
                color: #8899aa;
                cursor: pointer;
                transition: all 0.2s ease;
                white-space: nowrap;
                user-select: none;
            }}
            .toggle-btn:hover {{
                color: #ffffff;
            }}
            .toggle-btn.active {{
                background: #4f8ef7;
                color: #ffffff;
                box-shadow: 0 2px 8px rgba(79, 142, 247, 0.4);
            }}
            
            #legend {{
                position: absolute;
                bottom: 20px;
                left: 20px;
                z-index: 1000;
                background: rgba(13, 15, 22, 0.9);
                backdrop-filter: blur(6px);
                border-radius: 6px;
                padding: 10px 12px;
                width: 140px;
                border: 1px solid rgba(255,255,255,0.06);
            }}
            #legend h4 {{
                margin: 0 0 8px 0;
                font-size: 0.65rem;
                color: #8899aa;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-weight: 700;
            }}
            .lr {{
                display: flex;
                align-items: center;
                gap: 8px;
                color: #e2e8f0;
                font-size: 0.75rem;
                margin-bottom: 4px;
            }}
            .ld {{
                width: 8px;
                height: 8px;
                border-radius: 50%;
                flex-shrink: 0;
            }}
        </style>
    </head>
    <body>
        <div id="toggle-bar">
            <div id="btn-a" class="toggle-btn" onclick="switchScenario('A')">Scenario A: Historical Baseline</div>
            <div id="btn-b" class="toggle-btn active" onclick="switchScenario('B')">Scenario B: AI-Optimized</div>
        </div>
        
        <div id="legend">
            <h4>Predicted Risk</h4>
            <div class="lr"><div class="ld" style="background:#ff4444"></div><span>Critical</span></div>
            <div class="lr"><div class="ld" style="background:#ff8c00"></div><span>High</span></div>
            <div class="lr"><div class="ld" style="background:#f0c040"></div><span>Moderate</span></div>
            <div class="lr"><div class="ld" style="background:#2ecc71"></div><span>Low</span></div>
            <h4>Patrols</h4>
            <div class="lr"><div class="ld" style="background:#8899aa; border: 1px solid #ffffff;"></div><span>Scenario A (Hist)</span></div>
            <div class="lr"><div class="ld" style="background:#4f8ef7; border: 1px solid #ffffff;"></div><span>Scenario B (AI)</span></div>
        </div>
        
        <div id="map"></div>
        
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <script>
            var cellsData = {map_cells_json};
            
            var map = L.map('map', {{
                center: [12.9716, 77.5946],
                zoom: 12,
                zoomControl: false
            }});
            L.control.zoom({{ position: 'topright' }}).addTo(map);
            
            L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
                attribution: '&copy; OSM &copy; CARTO',
                subdomains: 'abcd', maxZoom: 19
            }}).addTo(map);
            
            var layerHexagons = L.layerGroup().addTo(map);
            var layerA = L.layerGroup();
            var layerB = L.layerGroup().addTo(map); // B active by default
            
            cellsData.forEach(function(c) {{
                if (c.polygon && c.polygon.length > 0) {{
                    var hexColor = '#2ecc71';
                    if (c.ieu >= 75) hexColor = '#ff4444';
                    else if (c.ieu >= 50) hexColor = '#ff8c00';
                    else if (c.ieu >= 25) hexColor = '#f0c040';
                    
                    var hexOpacity = 0.05 + 0.25 * (c.ieu / 100);
                    
                    var poly = L.polygon(c.polygon.map(function(pt){{ return [pt[1], pt[0]]; }}), {{
                        fillColor: hexColor,
                        fillOpacity: hexOpacity,
                        color: hexColor,
                        weight: 1,
                        opacity: 0.3
                    }}).addTo(layerHexagons);
                    
                    poly.bindTooltip("<strong>" + c.name + "</strong><br>Risk IEU: " + c.ieu.toFixed(1) + "<br>Units A: " + c.units_a + " | B: " + c.units_b);
                }}
                
                if (c.units_a > 0) {{
                    var markerA = L.circleMarker([c.lat, c.lng], {{
                        radius: 6 + c.units_a * 4,
                        fillColor: '#8899aa',
                        fillOpacity: 0.8,
                        color: '#ffffff',
                        weight: 1.5,
                        opacity: 0.9
                    }});
                    markerA.bindTooltip("<strong>" + c.name + "</strong><br>Scenario A: <strong>" + c.units_a + " units</strong> (Historical)");
                    markerA.addTo(layerA);
                }}
                
                if (c.units_b > 0) {{
                    var markerB = L.circleMarker([c.lat, c.lng], {{
                        radius: 6 + c.units_b * 4,
                        fillColor: '#4f8ef7',
                        fillOpacity: 0.8,
                        color: '#ffffff',
                        weight: 1.5,
                        opacity: 0.9
                    }});
                    markerB.bindTooltip("<strong>" + c.name + "</strong><br>Scenario B: <strong>" + c.units_b + " units</strong> (AI-Optimized)");
                    markerB.addTo(layerB);
                }}
            }});
            
            function switchScenario(scen) {{
                if (scen === 'A') {{
                    map.removeLayer(layerB);
                    map.addLayer(layerA);
                    document.getElementById('btn-a').classList.add('active');
                    document.getElementById('btn-b').classList.remove('active');
                }} else {{
                    map.removeLayer(layerA);
                    map.addLayer(layerB);
                    document.getElementById('btn-b').classList.add('active');
                    document.getElementById('btn-a').classList.remove('active');
                }}
            }}
        </script>
    </body>
    </html>
    """
    
    st.components.v1.html(map_html, height=420)
    st.markdown("<br/>", unsafe_allow_html=True)
    
    # ─── Component 4 & 5: Table and Efficiency Chart Row ────────────────────
    col_left, col_right = st.columns([1.1, 0.9])
    
    # Component 4: Zone Coverage Comparison Table (Left, 52%)
    with col_left:
        table_rows = []
        for _, row in top_8.reset_index().iterrows():
            location = row['location']
            station = row['station']
            pred_t1 = row['pred_t1']
            units_a = int(row['base_officers'])
            units_b = int(row['opt_officers'])
            
            # Row style color overlays
            row_style = ""
            if units_a > 0 and units_b == 0:
                row_style = "background-color: rgba(46, 204, 113, 0.05);"
            elif units_b > 0 and units_a == 0:
                row_style = "background-color: rgba(255, 68, 68, 0.07);"
                
            # Formatting cells A & B
            cell_A = f'<span style="color: #8899aa; font-weight: 700;">✅ {units_a} units</span>' if units_a > 0 else '<span style="color: #64748b; font-weight: 500;">—— No units</span>'
            cell_B = f'<span style="color: #4f8ef7; font-weight: 700;">✅ {units_b} units</span>' if units_b > 0 else '<span style="color: #64748b; font-weight: 500;">—— No units</span>'
            
            # IEU colored number + mini bar
            color = "#ff4444" if pred_t1 >= 75 else ("#ff8c00" if pred_t1 >= 50 else ("#f0c040" if pred_t1 >= 25 else "#2ecc71"))
            ieu_bar_width = (pred_t1 / 100) * 100
            ieu_cell_html = f"""
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-weight: 700; color: {color}; width: 35px;">{pred_t1:.1f}</span>
                <div style="width: 50px; background-color: rgba(255,255,255,0.05); height: 4px; border-radius: 2px; overflow: hidden;">
                    <div style="background-color: {color}; width: {ieu_bar_width}%; height: 100%;"></div>
                </div>
            </div>
            """
            
            table_rows.append(f"""
            <tr style="height: 52px; border-bottom: 1px solid rgba(255,255,255,0.03); {row_style}">
                <td style="padding: 10px 14px; font-weight: 700; color: #ffffff;">{location}</td>
                <td style="padding: 10px 14px;">{ieu_cell_html}</td>
                <td style="padding: 10px 14px;">{cell_A}</td>
                <td style="padding: 10px 14px;">{cell_B}</td>
            </tr>
            """)
            
        table_html = f"""
        <div style="background-color: #11141c; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.08); overflow: hidden;">
            <div style="background-color: rgba(255, 255, 255, 0.02); padding: 12px 18px; border-bottom: 1px solid rgba(255,255,255,0.05);">
                <span style="font-weight: 700; color: #ffffff; font-family: Outfit; font-size: 0.95rem;">
                    Where Do the Units Actually Go?
                </span>
            </div>
            <table style="width: 100%; border-collapse: collapse; font-family: Inter, sans-serif; text-align: left; font-size: 0.85rem;">
                <thead>
                    <tr style="border-bottom: 2px solid rgba(255,255,255,0.08); font-size: 0.75rem; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.05em; background-color: rgba(0,0,0,0.15); height: 36px;">
                        <th style="padding: 10px 14px;">Zone</th>
                        <th style="padding: 10px 14px;">Risk IEU</th>
                        <th style="padding: 10px 14px;">Scenario A (Hist)</th>
                        <th style="padding: 10px 14px;">Scenario B (AI)</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(table_rows)}
                </tbody>
            </table>
            <div style="background-color: rgba(255,255,255,0.02); padding: 12px 18px; border-top: 1px solid rgba(255,255,255,0.05); font-family: Inter, sans-serif; font-size: 0.8rem; color: #94a3b8; line-height: 1.5;">
                • <strong>Scenario A:</strong> {wasted_A} units on low-risk zones (IEU &lt; 25) — wasted capacity<br/>
                • <strong>Scenario B:</strong> {wasted_B} units on low-risk zones — <strong>{saved_wasted} units reallocated</strong> to critical zones
            </div>
        </div>
        """
        st.html(table_html)
        
    # Component 5: Efficiency Chart (Right, 48%)
    with col_right:
        st.markdown(
            """
            <div style="margin-bottom: 6px;">
                <h4 style="margin: 0; font-family: Outfit, sans-serif; font-weight: 700; color: #ffffff; font-size: 0.95rem;">
                    Unit Efficiency — IEU Covered Per Officer Deployed
                </h4>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        fig = go.Figure()
        
        # Scenario A dots (Gray)
        df_a_active = df_hour[df_hour['base_officers'] > 0]
        fig.add_trace(go.Scatter(
            x=df_a_active['pred_t1'],
            y=df_a_active['base_officers'],
            mode='markers',
            name='Scenario A (Hist)',
            marker=dict(
                color='#8899aa',
                size=9,
                opacity=0.7,
                line=dict(width=1, color='rgba(255,255,255,0.2)')
            ),
            hovertemplate="Risk IEU: %{x:.1f}<br>Units: %{y}<extra></extra>"
        ))
        
        # Scenario B dots (Blue)
        df_b_active = df_hour[df_hour['opt_officers'] > 0]
        fig.add_trace(go.Scatter(
            x=df_b_active['pred_t1'],
            y=df_b_active['opt_officers'],
            mode='markers',
            name='Scenario B (AI)',
            marker=dict(
                color='#4f8ef7',
                size=11,
                opacity=0.9,
                line=dict(width=1.5, color='#ffffff')
            ),
            hovertemplate="Risk IEU: %{x:.1f}<br>Units: %{y}<extra></extra>"
        ))
        
        # Vertical Line at IEU = 75
        max_y = max(df_hour['opt_officers'].max(), df_hour['base_officers'].max(), 1)
        fig.add_shape(
            type="line",
            x0=75, y0=0, x1=75, y1=max_y + 1,
            line=dict(color="rgba(255, 255, 255, 0.2)", width=1.5, dash="dash")
        )
        
        fig.update_layout(
            height=300,
            margin=dict(l=40, r=20, t=20, b=40),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(color="#aabbcc", size=9)
            ),
            xaxis=dict(
                title="Zone Risk IEU",
                titlefont=dict(color="#8899aa", size=10),
                tickfont=dict(color="#e2e8f0", size=9),
                gridcolor='rgba(255,255,255,0.05)',
                zeroline=False,
                range=[0, 105]
            ),
            yaxis=dict(
                title="Units Assigned",
                titlefont=dict(color="#8899aa", size=10),
                tickfont=dict(color="#e2e8f0", size=9),
                gridcolor='rgba(255,255,255,0.05)',
                zeroline=False
            ),
            annotations=[
                dict(
                    x=37.5, y=-0.12,
                    xref="x", yref="paper",
                    text="Low/Medium Risk",
                    showarrow=False,
                    font=dict(color="rgba(255,255,255,0.4)", size=8)
                ),
                dict(
                    x=87.5, y=-0.12,
                    xref="x", yref="paper",
                    text="Critical Risk",
                    showarrow=False,
                    font=dict(color="#ff4444", size=8)
                ),
                dict(
                    x=0.01, y=1.05,
                    xref="paper", yref="paper",
                    text="A clusters left (low risk) | B clusters right (high risk)",
                    showarrow=False,
                    font=dict(color="#8899aa", size=9)
                )
            ]
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # ─── Component 6: Unit Slider + Live Outcome ─────────────────────────────
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="background-color: #11141c; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.08); padding: 20px; margin-top: 15px;">
            <h4 style="margin: 0 0 15px 0; font-family: Outfit, sans-serif; font-weight: 700; color: #ffffff; font-size: 0.95rem; display: flex; align-items: center; gap: 8px;">
                🎛️ Live Simulation Summary
            </h4>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Display side-by-side outcome boxes
    col_out_a, col_out_b = st.columns([1, 1])
    
    avg_ieu_A = df_hour.loc[df_hour['base_officers'] > 0, 'pred_t1'].mean() if (df_hour['base_officers'] > 0).any() else 0.0
    avg_ieu_B = df_hour.loc[df_hour['opt_officers'] > 0, 'pred_t1'].mean() if (df_hour['opt_officers'] > 0).any() else 0.0
    
    with col_out_a:
        st.markdown(
            f"""
            <div style="border: 1px solid #8899aa; border-radius: 8px; padding: 16px; background-color: rgba(136,153,170,0.03); min-height: 120px;">
                <h4 style="margin: 0 0 10px 0; color: #8899aa; font-family: Outfit; font-size: 0.9rem; font-weight:700;">Scenario A: Historical Baseline</h4>
                <div style="font-size: 0.85rem; line-height: 1.5; color: #e2e8f0; font-family: Inter;">
                    • Critical zones covered: <strong>{crit_cov_A} / {total_critical}</strong><br>
                    • Wasted units: <strong>{wasted_A}</strong><br>
                    • Avg IEU covered: <strong>{avg_ieu_A:.1f}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col_out_b:
        st.markdown(
            f"""
            <div style="border: 1px solid #4f8ef7; border-radius: 8px; padding: 16px; background-color: rgba(79,142,247,0.05); min-height: 120px;">
                <h4 style="margin: 0 0 10px 0; color: #4f8ef7; font-family: Outfit; font-size: 0.9rem; font-weight:700;">Scenario B: AI-Optimized</h4>
                <div style="font-size: 0.85rem; line-height: 1.5; color: #e2e8f0; font-family: Inter;">
                    • Critical zones covered: <strong>{crit_cov_B} / {total_critical}</strong><br>
                    • Wasted units: <strong>{wasted_B}</strong><br>
                    • Avg IEU covered: <strong>{avg_ieu_B:.1f}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    # Generate summary text below outcomes
    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
    if total_officers <= 10:
        unpatrolled_A = total_critical - crit_cov_A
        summary_text = f"At this staffing level, neither scenario can cover all critical zones — {unpatrolled_A} zones remain unpatrolled in both cases."
    else:
        diff_crit = crit_cov_B - crit_cov_A
        if diff_crit > 0:
            summary_text = f"With {total_officers} officers, AI-optimized deployment covers <strong>{diff_crit} more critical zones</strong> than the historical pattern."
        else:
            summary_text = f"With {total_officers} officers, both scenarios cover the same number of critical zones, but Scenario B reallocates <strong>{wasted_A - wasted_B} units</strong> from low-risk zones to optimize density."
            
    st.markdown(
        f"""
        <div style="text-align: center; color: #ffffff; font-family: Outfit, sans-serif; font-size: 1rem; border-top: 1px dashed rgba(255,255,255,0.08); padding-top: 15px;">
            ── {summary_text} ──
        </div>
        """,
        unsafe_allow_html=True
    )

render_simulator_page()
