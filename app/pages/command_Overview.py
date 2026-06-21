"""
GridLock IQ — Command Overview
Redesigned for operational use: police-readable, action-oriented, no analyst jargon.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json, os, sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.data_state import ensure_data_loaded
from src.patrol_optimizer import get_dispatch_alerts

def _h(html: str) -> str:
    """Strip per-line leading whitespace so Streamlit's markdown parser
    never treats indented HTML lines as code blocks (4-space rule)."""
    return "\n".join(line.lstrip() for line in html.split("\n") if line.strip())

# ─── Color system ────────────────────────────────────────────────────────────
CRITICAL   = "#ff4444"
HIGH       = "#ff8c00"
MODERATE   = "#f0c040"
LOW        = "#2ecc71"
BACKGROUND = "#0f1117"
CARD_BG    = "#1a1d26"
TEXT_PRI   = "#ffffff"
TEXT_SEC   = "#8899aa"
ACCENT     = "#4f8ef7"
BORDER     = "rgba(255,255,255,0.07)"

# ─── Load POI tag lookup (junction names from provided dataset) ───────────────
@st.cache_data
def load_poi_tags():
    path = os.path.join(PROJECT_ROOT, "data", "processed", "cell_poi_tags.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


# ═══════════════════════════════════════════════════════════════════════════
#  HERO TREND CHART — with peak bands, NOW line, prediction zone
# ═══════════════════════════════════════════════════════════════════════════
def build_hero_trend_chart(df_today, df_hist_avg, current_hour):
    """Hourly violation area chart with operational overlays."""
    fig = go.Figure()

    # ── Peak risk background bands ──
    for band_start, band_end, label in [(7, 9, "Peak"), (17, 19, "Peak")]:
        fig.add_vrect(
            x0=band_start, x1=band_end,
            fillcolor="rgba(240,165,0,0.07)", layer="below", line_width=0,
            annotation_text=label, annotation_position="top left",
            annotation_font=dict(color="#f0c040", size=9)
        )

    # ── 7-Day historic avg — dashed gray ──
    fig.add_trace(go.Scatter(
        x=df_hist_avg["hour"], y=df_hist_avg["violation_count"],
        mode="lines", name="7-Day Avg",
        line=dict(color="rgba(136,153,170,0.6)", width=2, dash="dash"),
        hovertemplate="7-Day Avg: %{y:.0f}<extra></extra>"
    ))

    # ── Today — solid blue area ──
    fig.add_trace(go.Scatter(
        x=df_today["hour"], y=df_today["violation_count"],
        mode="lines", name="Today",
        line=dict(color=ACCENT, width=3, shape="spline"),
        fill="tozeroy", fillcolor="rgba(79,142,247,0.15)",
        hovertemplate="<b>%{x}:00</b><br>Today: %{y} violations<extra></extra>"
    ))

    # ── Prediction zone: NOW → NOW+4h ──
    pred_end = min(current_hour + 4, 23)
    if current_hour < 23:
        fig.add_vrect(
            x0=current_hour, x1=pred_end,
            fillcolor="rgba(255,100,100,0.07)", layer="below", line_width=0,
        )
        fig.add_vline(
            x=pred_end, line_width=1,
            line_dash="dot", line_color="rgba(255,100,100,0.4)"
        )
        fig.add_annotation(
            x=current_hour + (pred_end - current_hour) / 2, y=1, yref="paper",
            text="<i>Predicted</i>", showarrow=False,
            font=dict(color="rgba(255,100,100,0.7)", size=9)
        )

    # ── NOW vertical marker ──
    fig.add_vline(
        x=current_hour, line_width=2,
        line_dash="dash", line_color="#ffffff",
        annotation_text="NOW", annotation_position="top",
        annotation_font=dict(color="#ffffff", size=10, family="Outfit")
    )

    # ── Inline end-of-line labels ──
    if not df_today.empty and not df_hist_avg.empty:
        last_today = df_today.iloc[-1]
        last_hist = df_hist_avg.iloc[-1]
        fig.add_annotation(x=23, y=last_today["violation_count"],
            text="Today", showarrow=False, xanchor="left",
            font=dict(color=ACCENT, size=10))
        fig.add_annotation(x=23, y=last_hist["violation_count"],
            text="Avg", showarrow=False, xanchor="left",
            font=dict(color="#8899aa", size=10))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit, Inter, sans-serif", color=TEXT_PRI),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=CARD_BG, font_size=13, font_family="Inter"),
        margin=dict(l=55, r=60, t=30, b=45),
        height=380,
        showlegend=False,
        xaxis=dict(
            title=dict(text="Hour of Day", font=dict(color=TEXT_SEC, size=11)),
            tickmode="linear", tick0=0, dtick=2,
            tickfont=dict(color=TEXT_SEC),
            gridcolor="rgba(255,255,255,0.05)", zeroline=False,
            range=[-0.5, 24],
        ),
        yaxis=dict(
            title=dict(text="Violations / hour", font=dict(color=TEXT_SEC, size=11)),
            tickfont=dict(color=TEXT_SEC),
            gridcolor="rgba(255,255,255,0.05)", zeroline=False,
            nticks=4,
        ),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  RISK DISTRIBUTION BAR — tall, readable, counts inside
# ═══════════════════════════════════════════════════════════════════════════
def build_risk_distribution_bar(hour_forecasts):
    """Tall horizontal stacked bar with counts and percentages inside segments."""
    def classify(aoi):
        if aoi >= 65: return "Critical"
        if aoi >= 50: return "High"
        if aoi >= 25: return "Moderate"
        return "Low"

    tiers = hour_forecasts["AOI"].apply(classify).value_counts()
    total = max(tiers.sum(), 1)
    tier_colors = {"Critical": CRITICAL, "High": HIGH, "Moderate": MODERATE, "Low": LOW}

    fig = go.Figure()
    for tier in ["Critical", "High", "Moderate", "Low"]:
        count = tiers.get(tier, 0)
        pct = count / total * 100
        label = f"{count} zones  ({pct:.0f}%)" if pct >= 6 else (f"{count}" if pct >= 3 else "")
        fig.add_trace(go.Bar(
            y=["zones"], x=[pct],
            name=tier, orientation="h",
            marker=dict(color=tier_colors[tier]),
            text=f"{'🔴' if tier=='Critical' else '🟠' if tier=='High' else '🟡' if tier=='Moderate' else '🟢'} {tier}  {label}",
            textposition="inside" if pct >= 8 else "outside",
            textfont=dict(color="#ffffff", size=12, family="Inter"),
            hovertemplate=f"<b>{tier}</b>: {count} zones ({pct:.1f}%)<extra></extra>"
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        barmode="stack", height=72,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        xaxis=dict(visible=False, range=[0, 100]),
        yaxis=dict(visible=False),
    )
    return fig, tiers, total


# ═══════════════════════════════════════════════════════════════════════════
#  DAY-OF-WEEK BAR CHART — vertical, all historical data (bug fix)
# ═══════════════════════════════════════════════════════════════════════════
def build_weekday_bar_chart(current_weekday):
    """Vertical bar chart of avg violations/hour per day using precomputed stats."""
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    
    # Load precomputed weekday averages
    stats_path = os.path.join(PROJECT_ROOT, "data", "processed", "overview_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            stats = json.load(f)
        avg_values = stats.get("weekday_avg", [0]*7)
    else:
        avg_values = [0]*7
        
    avg_per_day = pd.Series(avg_values, index=range(7))

    colors = [HIGH if i == current_weekday else ACCENT for i in range(7)]

    fig = go.Figure(go.Bar(
        x=day_names, y=avg_per_day.values,
        marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.08)", width=1)),
        hovertemplate="<b>%{x}</b><br>Avg %{y:.0f} violations/hr<extra></extra>",
        text=[f"{v:.0f}" for v in avg_per_day.values],
        textposition="outside",
        textfont=dict(color=TEXT_SEC, size=10),
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit, Inter, sans-serif", color=TEXT_PRI),
        margin=dict(l=10, r=10, t=10, b=30),
        height=220,
        showlegend=False,
        xaxis=dict(tickfont=dict(color=TEXT_SEC), gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(tickfont=dict(color=TEXT_SEC), gridcolor="rgba(255,255,255,0.05)",
                   nticks=3, zeroline=False),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN PAGE RENDER
# ═══════════════════════════════════════════════════════════════════════════
def render_overview():
    ensure_data_loaded()
    poi_tags = load_poi_tags()

    df_forecast   = st.session_state.get("df_forecast")
    station_map   = st.session_state.get("station_map", {})
    selected_time = st.session_state.get("selected_time")

    if df_forecast is None:
        st.error("Datasets failed to load.")
        return

    # ── Resolve timestamp ──
    target_ts = pd.to_datetime(selected_time)
    if df_forecast["hour_dt"].dt.tz is not None and target_ts.tzinfo is None:
        target_ts = target_ts.tz_localize(df_forecast["hour_dt"].dt.tz)

    # Clamp to data range (prevent What-If slider pollution)
    data_min = df_forecast["hour_dt"].min()
    data_max = df_forecast["hour_dt"].max()
    target_ts = max(data_min, min(data_max, target_ts))

    selected_date   = target_ts.date()
    current_hour    = target_ts.hour
    current_weekday = target_ts.weekday()

    # ── Load filtered data (or full dataset) ──
    hour_forecasts = st.session_state.get("filtered_forecast", df_forecast)
    if hour_forecasts is None or hour_forecasts.empty:
        hour_forecasts = df_forecast



    # ── Compute KPI metrics ──
    aoi_col  = "AOI_max" if "AOI_max" in hour_forecasts.columns else "AOI"
    crit_df  = hour_forecasts[hour_forecasts[aoi_col] >= 65]           # F2-optimal threshold
    high_df  = hour_forecasts[(hour_forecasts[aoi_col] >= 50) & (hour_forecasts[aoi_col] < 65)]
    critical_count = len(crit_df)
    active_aoi     = hour_forecasts[hour_forecasts["AOI"] > 0]["AOI"]
    avg_aoi        = float(active_aoi.mean()) if not active_aoi.empty else 0.0
    total_active   = len(hour_forecasts[hour_forecasts["AOI"] > 0])

    # City risk level (used by Card 1)
    if critical_count >= 5 or avg_aoi >= 65:
        city_risk = "CRITICAL"; risk_bg = "linear-gradient(135deg,#7f1d1d,#991b1b)"
        risk_text_color = "#fecaca"
    elif critical_count >= 2 or avg_aoi >= 45:
        city_risk = "HIGH"; risk_bg = "linear-gradient(135deg,#7c2d12,#9a3412)"
        risk_text_color = "#fed7aa"
    elif avg_aoi >= 25:
        city_risk = "MODERATE"; risk_bg = "linear-gradient(135deg,#713f12,#854d0e)"
        risk_text_color = "#fef08a"
    else:
        city_risk = "LOW"; risk_bg = "linear-gradient(135deg,#14532d,#166534)"
        risk_text_color = "#bbf7d0"

    # Critical delta vs 1h ago
    prev_ts = target_ts - pd.Timedelta(hours=1)
    prev_fc  = df_forecast[df_forecast["hour_dt"] == prev_ts]
    prev_crit = len(prev_fc[prev_fc["AOI"] >= 65]) if not prev_fc.empty else 0
    crit_delta = critical_count - prev_crit
    delta_symbol = f"▲{crit_delta}" if crit_delta > 0 else (f"▼{abs(crit_delta)}" if crit_delta < 0 else "→ flat")
    delta_color  = CRITICAL if crit_delta > 0 else (LOW if crit_delta < 0 else TEXT_SEC)

    # Predicted peak window — display in IST
    target_ts_ist = target_ts.tz_convert("Asia/Kolkata") if target_ts.tzinfo else target_ts
    peak_sums  = {1: hour_forecasts["pred_t1"].sum(), 2: hour_forecasts["pred_t2"].sum(), 4: hour_forecasts["pred_t4"].sum()}
    peak_h     = max(peak_sums, key=peak_sums.get)
    peak_time  = (target_ts_ist + pd.Timedelta(hours=peak_h)).strftime("%I:%M %p")
    peak_label = f"{peak_time}  (+{peak_h}h)"

    # Active violations this hour — compare in parquet's timezone
    try:
        parquet_tz = df_forecast["hour_dt"].dt.tz
        if parquet_tz is not None:
            match_ts = target_ts.tz_convert(parquet_tz) if target_ts.tzinfo else target_ts.tz_localize(parquet_tz)
        else:
            match_ts = target_ts.tz_localize(None) if target_ts.tzinfo else target_ts
        active_viols_count = int(df_forecast[df_forecast["hour_dt"] == match_ts]["violation_count"].sum())
    except Exception:
        active_viols_count = int(hour_forecasts["violation_count"].sum()) if "violation_count" in hour_forecasts.columns else 0

    # Officers recommended (3 per critical zone + 1 per high zone)
    officers_rec = max(critical_count * 3 + len(high_df), 1)

    # Ticket validation rate (using precomputed approvals/rejections from overview_stats.json)
    stats_path = os.path.join(PROJECT_ROOT, "data", "processed", "overview_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            stats = json.load(f)
        app_n = stats.get("approved_count", 0)
        rej_n = stats.get("rejected_count", 0)
    else:
        app_n, rej_n = 0, 0
    val_rt = app_n / max(app_n + rej_n, 1) * 100

    # ── Load data timestamp ──
    from config import PROCESSED_DATA_PATH
    data_mtime = os.path.getmtime(PROCESSED_DATA_PATH) if os.path.exists(PROCESSED_DATA_PATH) else None
    if data_mtime:
        data_dt   = datetime.fromtimestamp(data_mtime)
        age_min   = (datetime.now() - data_dt).total_seconds() / 60
        is_fresh  = age_min < 15
        live_color = "#00ff88" if is_fresh else "#ff4444"
        data_label = data_dt.strftime("%H:%M IST, %b %d %Y")
    else:
        live_color = "#888"; data_label = "Unknown"; is_fresh = False

    # ════════════════════════════════════════════════════════════════
    #  COMPONENT 1 — HEADER STRIP (slim, 48px)
    # ════════════════════════════════════════════════════════════════
    col_header, col_picker = st.columns([3.2, 0.8])
    with col_header:
        st.markdown(f"""
        <style>
        @keyframes pulse {{
            0% {{ opacity: 1; }}
            50% {{ opacity: 0.3; }}
            100% {{ opacity: 1; }}
        }}
        .live-dot {{ animation: pulse 1.8s ease-in-out infinite; }}
        .header-strip {{
            display: flex; align-items: center; justify-content: space-between;
            background: {BACKGROUND}; border-bottom: 1px solid {BORDER};
            padding: 0 8px; height: 48px; margin: -1rem -1rem 1.2rem -1rem;
            font-family: 'Outfit', sans-serif;
        }}
        .header-left {{ font-size: 15px; font-weight: 800; color: {TEXT_PRI}; white-space: nowrap; }}
        .header-center {{ font-size: 12px; color: {TEXT_SEC}; display: flex; align-items: center; gap: 6px; }}
        .header-right {{ font-size: 11px; color: {TEXT_SEC}; text-align: right; white-space: nowrap; }}
        </style>
        <div class="header-strip">
            <div class="header-left">🚨 GridLock IQ &nbsp;—&nbsp; Command Center</div>
            <div class="header-center">
                <span class="live-dot" style="color:{live_color}; font-size:16px;">●</span>
                <span>{'LIVE' if is_fresh else 'STALE'} &nbsp;|&nbsp; Data as of {data_label}</span>
            </div>
            <div class="header-right">Window: {target_ts_ist.strftime('%Y-%m-%d %I:%M %p')} IST</div>
        </div>
        """, unsafe_allow_html=True)
    with col_picker:
        with st.popover("📅 Change Hour", use_container_width=True):
            # Use df_forecast_hours (all available hours) NOT df_forecast (single-day slice)
            df_forecast_hours = st.session_state.get("df_forecast_hours")
            all_times_ist = pd.DatetimeIndex(df_forecast_hours['hour_dt'].unique()).tz_convert("Asia/Kolkata")
            unique_dates = sorted(list(set(all_times_ist.date)))

            target_ts_ist = target_ts.tz_convert("Asia/Kolkata") if target_ts.tzinfo else target_ts
            sel_date = st.date_input("Select Date", value=target_ts_ist.date(), min_value=min(unique_dates), max_value=max(unique_dates), key="ov_date_picker")

            available_hours = sorted(list(set([t.hour for t in all_times_ist if t.date() == sel_date])))
            if not available_hours:
                available_hours = list(range(24))
                
            rounded_hour = (target_ts_ist + pd.Timedelta(minutes=30)).hour
            sel_hour = st.selectbox("Select Hour", options=available_hours, format_func=lambda h: f"{h:02d}:00", index=available_hours.index(rounded_hour) if rounded_hour in available_hours else 0, key="ov_hour_picker")
            
            if st.button("Apply Changes", use_container_width=True, key="ov_apply_btn"):
                # Match user selected date and hour to the exact database timestamp in all_times_ist
                matching_ts = [t for t in all_times_ist if t.date() == sel_date and t.hour == sel_hour]
                if matching_ts:
                    new_ts_local = matching_ts[0]
                else:
                    new_ts_local = pd.Timestamp(f"{sel_date} {sel_hour:02d}:30:00", tz="Asia/Kolkata")
                st.session_state['selected_time'] = new_ts_local.tz_convert("UTC")
                st.rerun()

    # ════════════════════════════════════════════════════════════════
    #  COMPONENT 2 — KPI CARDS (5 cards)
    # ════════════════════════════════════════════════════════════════
    c1, c2, c3, c4, c5 = st.columns([1.3, 1.0, 1.0, 1.0, 1.0])

    # Card 1 — City Risk Level (color-filled)
    c1.markdown(f"""
    <div style="background:{risk_bg}; min-height:140px; padding:20px 22px;
                border-radius:10px; box-shadow:0 8px 24px rgba(0,0,0,0.4);">
        <div style="color:{risk_text_color}; font-size:11px; font-weight:700;
                    letter-spacing:.1em; text-transform:uppercase; margin-bottom:8px;">
            City Risk Level
        </div>
        <div style="color:{risk_text_color}; font-size:36px; font-weight:900;
                    font-family:Outfit; letter-spacing:-.01em;">
            {city_risk}
        </div>
        <div style="color:rgba(255,255,255,0.6); font-size:11px; margin-top:10px;">
            Current city-wide assessment
        </div>
    </div>""", unsafe_allow_html=True)

    # Card 2 — Critical Hotspots
    c2.markdown(f"""
    <div style="background:{CARD_BG}; border-left:4px solid {CRITICAL}; min-height:140px;
                padding:20px 22px; border-radius:10px; box-shadow:0 8px 24px rgba(0,0,0,0.3);">
        <div style="color:{TEXT_SEC}; font-size:11px; font-weight:700;
                    letter-spacing:.08em; text-transform:uppercase; margin-bottom:8px;">
            Critical Hotspots
        </div>
        <div style="color:{CRITICAL}; font-size:40px; font-weight:900; font-family:Outfit;">
            {critical_count}
        </div>
        <div style="color:{TEXT_SEC}; font-size:11px; margin-top:4px;">
            Zones above Risk Score 65 — immediate dispatch
        </div>
        <div style="color:{delta_color}; font-size:12px; font-weight:700; margin-top:8px;">
            {delta_symbol} from last hour
        </div>
    </div>""", unsafe_allow_html=True)

    # Card 3 — Predicted Peak Window
    c3.markdown(f"""
    <div style="background:{CARD_BG}; border-left:4px solid {MODERATE}; min-height:140px;
                padding:20px 22px; border-radius:10px; box-shadow:0 8px 24px rgba(0,0,0,0.3);">
        <div style="color:{TEXT_SEC}; font-size:11px; font-weight:700;
                    letter-spacing:.08em; text-transform:uppercase; margin-bottom:8px;">
            ⏱ Predicted Peak Window
        </div>
        <div style="color:{MODERATE}; font-size:28px; font-weight:900; font-family:Outfit; margin-top:4px;">
            {peak_label}
        </div>
        <div style="color:{TEXT_SEC}; font-size:11px; margin-top:10px;">
            Forecasted highest congestion window
        </div>
    </div>""", unsafe_allow_html=True)

    # Card 4 — Active Violations This Hour
    c4.markdown(f"""
    <div style="background:{CARD_BG}; border-left:4px solid {ACCENT}; min-height:140px;
                padding:20px 22px; border-radius:10px; box-shadow:0 8px 24px rgba(0,0,0,0.3);">
        <div style="color:{TEXT_SEC}; font-size:11px; font-weight:700;
                    letter-spacing:.08em; text-transform:uppercase; margin-bottom:8px;">
            Active Violations
        </div>
        <div style="color:{TEXT_PRI}; font-size:40px; font-weight:900; font-family:Outfit;">
            {active_viols_count:,}
        </div>
        <div style="color:{TEXT_SEC}; font-size:11px; margin-top:4px;">
            Violations logged this hour city-wide
        </div>
    </div>""", unsafe_allow_html=True)

    # Card 5 — Officers Recommended
    c5.markdown(f"""
    <div style="background:{CARD_BG}; border-left:4px solid #4fc3f7; min-height:140px;
                padding:20px 22px; border-radius:10px; box-shadow:0 8px 24px rgba(0,0,0,0.3);">
        <div style="color:{TEXT_SEC}; font-size:11px; font-weight:700;
                    letter-spacing:.08em; text-transform:uppercase; margin-bottom:8px;">
            Officers Recommended
        </div>
        <div style="color:#4fc3f7; font-size:36px; font-weight:900; font-family:Outfit;">
            {officers_rec}
        </div>
        <div style="color:{TEXT_SEC}; font-size:11px; margin-top:4px;">
            Units advised for dispatch
        </div>
        <div style="color:#4fc3f7; font-size:11px; font-weight:600; margin-top:8px;">
            Based on {critical_count} critical + {len(high_df)} high zones
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════
    #  COMPONENTS 3 & 4 — HERO TREND (60%) + ACTIVE ALERTS (40%)
    # ════════════════════════════════════════════════════════════════
    col_trend, col_alerts = st.columns([1.5, 1.0])

    with col_trend:
        # Build hourly data for the selected date from df_forecast (no violations needed)
        df_day = df_forecast.copy()
        df_day["local_dt"] = df_day["hour_dt"].dt.tz_convert("Asia/Kolkata")
        df_day_selected = df_day[df_day["local_dt"].dt.date == pd.to_datetime(selected_date).date()]

        all_hrs = pd.DataFrame({"hour": range(24)})
        if not df_day_selected.empty:
            day_hourly = df_day_selected.groupby(df_day_selected["local_dt"].dt.hour)["violation_count"].sum().reset_index()
            day_hourly.columns = ["hour", "violation_count"]
            day_hourly = pd.merge(all_hrs, day_hourly, on="hour", how="left").fillna(0)
            day_hourly["violation_count"] = day_hourly["violation_count"].astype(int)
        else:
            day_hourly = pd.DataFrame({"hour": range(24), "violation_count": [0]*24})

        # Historic baseline — load precomputed average for same weekday from overview_stats.json
        stats_path = os.path.join(PROJECT_ROOT, "data", "processed", "overview_stats.json")
        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                stats = json.load(f)
            hist_list = stats.get("hourly_hist_avg", {}).get(str(current_weekday), [0]*24)
        else:
            hist_list = [0]*24
        hist_hrly = pd.DataFrame({"hour": range(24), "violation_count": hist_list})

        st.plotly_chart(
            build_hero_trend_chart(day_hourly, hist_hrly, current_hour),
            use_container_width=True
        )

    with col_alerts:
        # Build alerts using poi_tags for human-readable zone names
        # Load dominant vehicle map from precomputed JSON
        stats_path = os.path.join(PROJECT_ROOT, "data", "processed", "overview_stats.json")
        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                stats = json.load(f)
            cell_vehicle_map = stats.get("cell_vehicle_map", {})
        else:
            cell_vehicle_map = {}
        alerts = get_dispatch_alerts(df_forecast, target_ts, station_map, cell_vehicle_map)
        n_alerts = len(alerts)

        st.markdown(_h(f"""
        <div style="display:flex; justify-content:space-between; align-items:center;
                    margin-bottom:10px;">
            <span style="color:{TEXT_PRI}; font-family:Outfit; font-weight:800;
                         font-size:15px;">🚨 Active Alerts ({n_alerts})</span>
            <span style="color:{ACCENT}; font-size:12px; cursor:pointer;">View All →</span>
        </div>
        """), unsafe_allow_html=True)

        alert_container_html = ""
        for al in alerts[:7]:
            cell_id  = al.get("h3_cell", "")
            poi      = poi_tags.get(cell_id, {})
            zone_name = poi.get("poi_label") or al.get("police_station", cell_id[:10])
            juris    = al.get("police_station", "Unknown")
            ieu      = min(float(al.get("aoi", 0)), 100)
            veh      = al.get("dominant_vehicle", "CAR")
            viols    = int(al.get("violation_count", 0))
            units    = max(int(al.get("units_needed", 2)), 1)
            sev      = al.get("severity", "Moderate")

            if sev == "Critical":
                border_c = CRITICAL; sev_bg = "rgba(255,68,68,0.08)"
            elif sev == "High":
                border_c = HIGH; sev_bg = "rgba(255,140,0,0.08)"
            else:
                border_c = MODERATE; sev_bg = "rgba(240,192,64,0.06)"

            bar_w = min(int(ieu), 100)
            bar_c = CRITICAL if ieu >= 65 else HIGH if ieu >= 50 else MODERATE

            alert_container_html += f"""
            <div style="background:{sev_bg}; border-left:6px solid {border_c};
                        border-radius:6px; padding:12px 14px; margin-bottom:8px;
                        font-family:Inter,sans-serif;">
                <div style="color:{border_c}; font-size:11px; font-weight:700;
                             letter-spacing:.06em; text-transform:uppercase;">
                    {sev.upper()} — {zone_name}
                </div>
                <div style="background:rgba(255,255,255,0.08); border-radius:3px;
                             height:4px; margin:6px 0; width:100%;">
                    <div style="background:{bar_c}; border-radius:3px;
                                 height:4px; width:{bar_w}%;"></div>
                </div>
                <div style="color:{TEXT_SEC}; font-size:11px; margin-top:2px;">
                    Risk Score: <b style="color:{TEXT_PRI}">{ieu:.0f}/100</b> &nbsp;·&nbsp;
                    Primary: {veh} &nbsp;·&nbsp; {viols} violations now
                </div>
                <div style="color:{TEXT_SEC}; font-size:11px; margin-top:3px;">
                    Jurisdiction: {juris}
                </div>
                <div style="color:{TEXT_PRI}; font-size:12px; font-weight:700;
                             margin-top:8px; letter-spacing:.01em;">
                    → Deploy {units} unit{'s' if units != 1 else ''} immediately
                </div>
            </div>"""

        if not alert_container_html:
            alert_container_html = f"""
            <div style="color:{TEXT_SEC}; font-size:13px; padding:20px; text-align:center;">
                ✅ No active critical alerts in this window
            </div>"""

        st.markdown(_h(f"""
        <div style="max-height:380px; overflow-y:auto; padding-right:4px;">
            {alert_container_html}
        </div>"""), unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════
    #  COMPONENT 5 — RISK DISTRIBUTION BAR (full width, tall)
    # ════════════════════════════════════════════════════════════════
    risk_fig, tiers, total = build_risk_distribution_bar(hour_forecasts)
    st.markdown(_h(f"""
    <div style="font-family:Outfit; font-size:13px; color:{TEXT_SEC}; margin-bottom:6px;">
        City-wide Zone Distribution — <b style="color:{TEXT_PRI}">{total} active cells</b>
    </div>"""), unsafe_allow_html=True)
    st.plotly_chart(risk_fig, use_container_width=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════
    #  COMPONENTS 6 & 7 — TOP HOTSPOTS TABLE (55%) + WEEKDAY (45%)
    # ════════════════════════════════════════════════════════════════
    col_table, col_dow = st.columns([1.2, 1.0])

    with col_table:
        st.markdown(_h(f"""
        <div style="font-family:Outfit; font-size:14px; font-weight:700;
                    color:{TEXT_PRI}; margin-bottom:10px;">
            🔥 Top Hotspot Zones
        </div>"""), unsafe_allow_html=True)

        # Build top-8 ranked cells with IEU, trend, jurisdiction, poi name
        top_cells = hour_forecasts.sort_values("AOI", ascending=False).head(8).copy()
        prev_aoi_map = (
            df_forecast[df_forecast["hour_dt"] == (target_ts - pd.Timedelta(hours=1))]
            .set_index("h3_cell")["AOI"].to_dict()
        )

        header_html = f"""
        <div style="display:grid; grid-template-columns:36px 1fr 100px 80px 100px 90px;
                    gap:0; padding:6px 10px; border-bottom:1px solid {BORDER};
                    font-size:11px; font-weight:700; color:{TEXT_SEC};
                    font-family:Inter; letter-spacing:.04em; text-transform:uppercase;">
            <div>#</div><div>Zone</div><div>Risk Score</div>
            <div>Trend</div><div>Jurisdiction</div><div>Status</div>
        </div>"""

        rows_html = ""
        for i, (_, row) in enumerate(top_cells.iterrows()):
            cell_id  = row["h3_cell"]
            poi      = poi_tags.get(cell_id, {})
            zone_nm  = poi.get("poi_label") or station_map.get(cell_id, {}).get("police_station", cell_id[:12])
            juris    = station_map.get(cell_id, {}).get("police_station", "—")
            ieu      = float(row["AOI"])
            prev     = prev_aoi_map.get(cell_id, ieu)
            delta    = ieu - prev
            trend_s  = f"▲ +{delta:.0f}" if delta > 1 else (f"▼ {delta:.0f}" if delta < -1 else "→ flat")
            trend_c  = CRITICAL if delta > 1 else (LOW if delta < -1 else TEXT_SEC)

            if ieu >= 65:   status="🔴 CRITICAL"; stat_c=CRITICAL
            elif ieu >= 50: status="🟠 HIGH";     stat_c=HIGH
            elif ieu >= 25: status="🟡 MODERATE"; stat_c=MODERATE
            else:           status="🟢 LOW";      stat_c=LOW

            bar_w = min(int(ieu), 100)
            zebra = "rgba(255,255,255,0.02)" if i % 2 == 0 else "transparent"

            rows_html += f"""
            <div style="display:grid; grid-template-columns:36px 1fr 100px 80px 100px 90px;
                        gap:0; padding:10px 10px; background:{zebra};
                        border-bottom:1px solid rgba(255,255,255,0.03);
                        font-family:Inter; align-items:center; min-height:52px;">
                <div style="color:{TEXT_SEC}; font-size:12px; font-weight:700;">{i+1}</div>
                <div style="color:{TEXT_PRI}; font-size:12px; font-weight:600;
                             overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
                             padding-right:8px;">{zone_nm}</div>
                <div>
                    <span style="color:{stat_c}; font-size:13px; font-weight:800;
                                 font-family:Outfit;">{ieu:.0f}</span>
                    <span style="color:{TEXT_SEC}; font-size:10px;">/100</span>
                    <div style="background:rgba(255,255,255,0.08); border-radius:2px;
                                height:3px; margin-top:3px; width:80px;">
                        <div style="background:{stat_c}; border-radius:2px;
                                    height:3px; width:{bar_w * 0.8:.0f}px;"></div>
                    </div>
                </div>
                <div style="color:{trend_c}; font-size:12px; font-weight:600;">{trend_s}</div>
                <div style="color:{TEXT_SEC}; font-size:11px;">{juris}</div>
                <div style="color:{stat_c}; font-size:11px; font-weight:700;">{status}</div>
            </div>"""

        st.markdown(_h(f"""
        <div style="background:{CARD_BG}; border-radius:10px; overflow:hidden;
                    border:1px solid {BORDER}; box-shadow:0 4px 20px rgba(0,0,0,0.3);">
            {header_html}
            {rows_html}
            <div style="padding:10px 10px; font-size:11px; color:{ACCENT};
                        border-top:1px solid {BORDER}; text-align:right;">
                View all active zones →
            </div>
        </div>"""), unsafe_allow_html=True)

    with col_dow:
        st.markdown(_h(f"""
        <div style="font-family:Outfit; font-size:14px; font-weight:700;
                    color:{TEXT_PRI}; margin-bottom:10px;">
            📅 Which Days Are Busiest
        </div>
        <div style="font-family:Inter; font-size:11px; color:{TEXT_SEC}; margin-bottom:8px;">
            City average violations/hr &nbsp;·&nbsp;
            <span style="color:{HIGH};">■</span> Today ({'Mon Tue Wed Thu Fri Sat Sun'.split()[current_weekday]})
        </div>"""), unsafe_allow_html=True)

        st.plotly_chart(
            build_weekday_bar_chart(current_weekday),
            use_container_width=True
        )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════
    #  COMPONENT 8 — TECHNICAL METRICS (collapsed expander)
    # ════════════════════════════════════════════════════════════════
    with st.expander("📊 Model Performance Metrics — Technical Detail", expanded=False):
        st.caption("For supervisors and analysts only — not required for operational dispatch decisions.")
        m1, m2 = st.columns(2)

        with m1:
            st.markdown("**Model Accuracy**")
            st.markdown(f"""
| Horizon | Active R² | Critical MAE | Recall |
|---------|-----------|-------------|--------|
| T+1h | 0.724 | 9.08 | 87.1% |
| T+2h | — | 18.09 | — |
| T+4h | — | 18.04 | — |
""")

        with m2:
            st.markdown("**Signal Quality**")
            st.markdown(f"""
| Metric | Value |
|--------|-------|
| Lag-1h Autocorrelation | 0.884 |
| Critical tier (active) | 17.4% |
| Precision @65 threshold | 71.1% |
| F2 Score | 0.834 |
| False Positive Rate | 0.47% |
| Ticket Validation Rate | {val_rt:.1f}% |
""")


render_overview()
