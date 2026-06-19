import streamlit as st
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# HYBRID ALERTING ENGINE
#
# Design decision (documented in absence of live BTP operator feedback):
#   A flat global AOI threshold is not viable across 26 stations with 40x
#   volume spread and 3x severity spread — it floods Upparpet (52/day at
#   Option B) while silencing Thalagattapura (0/day).
#
#   A pure station-relative threshold (top X% of each station's distribution)
#   manufactures urgency at quiet stations: a station with mild violations would
#   still get "critical" flags every day.
#
#   Hybrid fix: station-relative rank (top X% of the trailing 30-day window)
#   GATED by an absolute AOI floor. A station whose best violations that day
#   don't clear the floor correctly shows zero critical alerts.
#
#   This requires validation against real dispatch capacity if deployed live.
# ─────────────────────────────────────────────────────────────────────────────

ALERT_FLOOR_AOI = 25.0          # Absolute minimum to qualify (Moderate boundary)
ALERT_STATION_PERCENTILE = 0.90  # Top 10% within each station's trailing window
ALERT_TRAILING_DAYS = 30         # Window for computing per-station thresholds


def compute_station_thresholds(df: pd.DataFrame,
                                trailing_days: int = ALERT_TRAILING_DAYS,
                                percentile: float = ALERT_STATION_PERCENTILE,
                                floor_aoi: float = ALERT_FLOOR_AOI) -> dict:
    """
    Computes per-station AOI thresholds from the trailing N-day window.

    Returns a dict {police_station: threshold_aoi}.
    A violation must exceed BOTH the per-station threshold AND the absolute
    floor to be classified as Critical.

    Uses a 30-day trailing window rather than single-day so small stations
    (e.g. Thalagattapura at ~6 violations/day) have enough volume to produce
    a stable percentile estimate.
    """
    if 'AOI' not in df.columns or 'police_station' not in df.columns:
        return {}

    # Use trailing window if created_datetime is available, else use full dataset
    if 'created_datetime' in df.columns and pd.api.types.is_datetime64_any_dtype(df['created_datetime']):
        cutoff = df['created_datetime'].max() - pd.Timedelta(days=trailing_days)
        window_df = df[df['created_datetime'] >= cutoff].copy()
    else:
        window_df = df.copy()

    if window_df.empty:
        return {}

    # For each station, compute the station-relative percentile threshold
    # Only consider violations that clear the absolute floor
    above_floor = window_df[window_df['AOI'] >= floor_aoi]
    if above_floor.empty:
        return {}

    thresholds = (
        above_floor.groupby('police_station')['AOI']
        .quantile(percentile)
        .to_dict()
    )
    # Stations with no violations above the floor get no threshold (= no critical alerts)
    return thresholds


def classify_violation_alert(violation_aoi: float,
                              station: str,
                              station_thresholds: dict,
                              floor_aoi: float = ALERT_FLOOR_AOI) -> str:
    """
    Returns alert tier for a single violation given the hybrid criteria.

    Tiers:
      'Critical'  — above station-relative p90 threshold AND above floor
      'High'      — above floor but below station threshold (in top 10–25%)
      'Moderate'  — above floor but in lower station percentiles
      'Low'       — below floor; nothing actionable
    """
    if violation_aoi < floor_aoi:
        return 'Low'

    threshold = station_thresholds.get(station)
    if threshold is None:
        # Station has no history above the floor — treat as Low
        return 'Low'

    if violation_aoi >= threshold:
        return 'Critical'
    elif violation_aoi >= threshold * 0.75:
        return 'High'
    else:
        return 'Moderate'


def render_alert_card(alert):

    """
    Renders a single alert card using custom HTML for a premium look.
    """
    aoi = alert.get('predicted_aoi', 80)
    station = alert.get('station', 'UNKNOWN')
    violator = alert.get('primary_violator', 'CAR')
    location = alert.get('location', 'UNKNOWN')
    units = alert.get('recommended_units', 2)
    message = alert.get('message', '')
    
    # Select color scheme based on severity
    if aoi >= 75:
        border_color = "#ef4444" # Red
        bg_color = "rgba(239, 68, 68, 0.08)"
        text_color = "#fca5a5"
        label = "🚨 CRITICAL GRIDLOCK PREDICTED"
    elif aoi >= 50:
        border_color = "#f97316" # Orange
        bg_color = "rgba(249, 115, 22, 0.08)"
        text_color = "#fdbb2d"
        label = "⚠️ HIGH CONGESTION RISK"
    else:
        border_color = "#eab308" # Yellow
        bg_color = "rgba(234, 179, 8, 0.08)"
        text_color = "#fef08a"
        label = "⚡ MODERATE CONGESTION RISK"
        
    html_content = f"""
    <div style="
        background-color: {bg_color}; 
        border-left: 4px solid {border_color}; 
        padding: 14px; 
        border-radius: 6px; 
        margin-bottom: 12px; 
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    ">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <span style="color: {border_color}; font-weight: 700; font-size: 13px; letter-spacing: 0.05em; font-family: 'Outfit', sans-serif;">
                {label}
            </span>
            <span style="background-color: {border_color}; color: #ffffff; font-weight: 800; font-size: 11px; padding: 2px 6px; border-radius: 3px; font-family: sans-serif;">
                AOI: {aoi:.0f}
            </span>
        </div>
        <div style="color: #ffffff; font-size: 14px; font-weight: 500; line-height: 1.4; margin-bottom: 8px; font-family: 'Inter', sans-serif;">
            {message}
        </div>
        <div style="display: flex; gap: 15px; font-size: 12px; color: #9ca3af; font-family: 'Inter', sans-serif;">
            <span><b>Jurisdiction:</b> {station}</span>
            <span><b>Target:</b> {violator}s</span>
            <span><b>Action:</b> Deploy {units} Unit(s)</span>
        </div>
    </div>
    """
    st.markdown(html_content, unsafe_allow_html=True)

def render_alert_feed(alerts, max_display=10):
    """
    Renders an active feed of dispatch alerts in a sidebar or main feed.
    """
    if not alerts:
        st.info("🟢 No critical alerts active. Traffic flows within normal parameters.")
        return
        
    st.markdown(f"<h3 style='font-family: Outfit; font-weight: 600; color: #ffffff; margin-bottom: 15px;'>Active Alerts ({len(alerts)})</h3>", unsafe_allow_html=True)
    
    for alert in alerts[:max_display]:
        render_alert_card(alert)
        
    if len(alerts) > max_display:
        st.caption(f"Showing top {max_display} of {len(alerts)} active alerts.")
