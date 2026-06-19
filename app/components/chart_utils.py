import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# ─────────────────────── Premium Dark Color Palette ───────────────────────
DARK_BG = "rgba(14, 17, 23, 0)"
CARD_BG = "#1e293b"
COLOR_PRIMARY = "#6366f1"   # Indigo-500
COLOR_PRIMARY_LIGHT = "rgba(99,102,241,0.15)"
COLOR_CYAN = "#22d3ee"      # Cyan-400
COLOR_GREEN = "#10b981"     # Emerald-500
COLOR_GREEN_LIGHT = "rgba(16,185,129,0.15)"
COLOR_YELLOW = "#f59e0b"    # Amber-500
COLOR_ORANGE = "#f97316"    # Orange-500
COLOR_RED = "#ef4444"       # Rose-500
COLOR_RED_LIGHT = "rgba(239,68,68,0.12)"
COLOR_PURPLE = "#a855f7"    # Purple-500
COLOR_PINK = "#ec4899"      # Pink-500
COLOR_TEAL = "#14b8a6"      # Teal-500

GRADIENT_BLUE = ["#6366f1", "#818cf8", "#a5b4fc"]
GRADIENT_WARM = ["#ef4444", "#f97316", "#f59e0b", "#84cc16", "#10b981"]
RISK_COLORS = {"Critical": COLOR_RED, "High": COLOR_ORANGE, "Moderate": COLOR_YELLOW, "Low": COLOR_GREEN}

GRID_COLOR = "rgba(255, 255, 255, 0.06)"
FONT_STYLE = dict(family="Outfit, Inter, Roboto, sans-serif", size=12, color="#e2e8f0")

# Reusable dark layout template
_BASE_LAYOUT = dict(
    paper_bgcolor=DARK_BG,
    plot_bgcolor=DARK_BG,
    font=FONT_STYLE,
    hovermode="x unified",
    hoverlabel=dict(bgcolor="#1e293b", font_size=12, font_family="Inter"),
)

# Default legend style for charts that use it
_DEFAULT_LEGEND = dict(
    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
    font=dict(color="#e2e8f0", size=11),
    bgcolor="rgba(0,0,0,0)"
)


# ═══════════════════════════════════════════════════════════════════════════
#  1. HOURLY TREND (Area chart with gradient fill)
# ═══════════════════════════════════════════════════════════════════════════
def create_hourly_trend_chart(df_today, df_historical_avg):
    """Hourly violation area chart – today vs 7-day average."""
    fig = go.Figure()

    # Historical band (translucent fill to zero)
    fig.add_trace(go.Scatter(
        x=df_historical_avg['hour'], y=df_historical_avg['violation_count'],
        mode='lines', name='7-Day Historic Avg',
        line=dict(color='rgba(148,163,184,0.5)', width=2, dash='dot'),
        fill='tozeroy', fillcolor='rgba(148,163,184,0.06)',
        hovertemplate='Avg: %{y:.0f}<extra></extra>'
    ))

    # Today – gradient area fill
    fig.add_trace(go.Scatter(
        x=df_today['hour'], y=df_today['violation_count'],
        mode='lines+markers', name='Selected Day',
        line=dict(color=COLOR_PRIMARY, width=3, shape='spline'),
        marker=dict(size=7, color=COLOR_PRIMARY, line=dict(width=2, color='#ffffff')),
        fill='tozeroy', fillcolor=COLOR_PRIMARY_LIGHT,
        hovertemplate='Today: %{y}<extra></extra>'
    ))

    fig.update_layout(
        **_BASE_LAYOUT,
        margin=dict(l=50, r=20, t=50, b=40),
        legend=_DEFAULT_LEGEND,
        title=dict(text="📈 Hourly Violation Trend", font=dict(size=17, color="#ffffff", family="Outfit")),
        xaxis=dict(
            title=dict(text="Hour of Day", font=dict(color="#94a3b8")),
            tickfont=dict(color="#94a3b8"), gridcolor=GRID_COLOR,
            tickmode='linear', tick0=0, dtick=2,
            zeroline=False
        ),
        yaxis=dict(
            title=dict(text="Violations", font=dict(color="#94a3b8")),
            tickfont=dict(color="#94a3b8"), gridcolor=GRID_COLOR,
            zeroline=False
        ),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  2. DAILY TIME-SERIES TREND (area + bar combo)
# ═══════════════════════════════════════════════════════════════════════════
def create_time_series_trend_chart(df_trend, window_label):
    """Daily violation time-series with gradient fill and 7-day rolling avg overlay."""
    fig = go.Figure()

    # Bar background
    fig.add_trace(go.Bar(
        x=df_trend['date'], y=df_trend['violation_count'],
        name='Daily Violations',
        marker=dict(color=COLOR_PRIMARY, opacity=0.35, line=dict(width=0)),
        hovertemplate='Date: %{x}<br>Count: %{y}<extra></extra>'
    ))

    # Rolling 7-day average
    if len(df_trend) >= 7:
        df_trend = df_trend.copy()
        df_trend['rolling_avg'] = df_trend['violation_count'].rolling(7, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=df_trend['date'], y=df_trend['rolling_avg'],
            mode='lines', name='7-Day Rolling Avg',
            line=dict(color=COLOR_CYAN, width=3, shape='spline'),
            hovertemplate='Avg: %{y:.0f}<extra></extra>'
        ))

    fig.update_layout(
        **_BASE_LAYOUT,
        margin=dict(l=50, r=20, t=50, b=40),
        legend=_DEFAULT_LEGEND,
        title=dict(text=f"📊 Daily Violation Trend — {window_label}", font=dict(size=17, color="#ffffff", family="Outfit")),
        barmode='overlay',
        xaxis=dict(title=dict(text="Date", font=dict(color="#94a3b8")), tickfont=dict(color="#94a3b8"), gridcolor=GRID_COLOR),
        yaxis=dict(title=dict(text="Violations", font=dict(color="#94a3b8")), tickfont=dict(color="#94a3b8"), gridcolor=GRID_COLOR),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  3. FEATURE IMPORTANCE (horizontal gradient bar)
# ═══════════════════════════════════════════════════════════════════════════
def create_feature_importance_chart(importances):
    """LightGBM feature importance with gradient colouring."""
    importances = importances.sort_values('importance', ascending=True)
    n = len(importances)
    colors = [f'rgba(99,102,241,{0.35 + 0.65*(i/max(n-1,1))})' for i in range(n)]

    fig = go.Figure(go.Bar(
        x=importances['importance'], y=importances['feature'],
        orientation='h',
        marker=dict(color=colors, line=dict(color='rgba(255,255,255,0.08)', width=1)),
        hovertemplate='%{y}: %{x:.0f}<extra></extra>'
    ))

    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(text="🧠 LightGBM Feature Importance", font=dict(size=17, color="#ffffff", family="Outfit")),
        margin=dict(l=160, r=20, t=50, b=30),
        xaxis=dict(title=dict(text="Importance", font=dict(color="#94a3b8")), tickfont=dict(color="#94a3b8"), gridcolor=GRID_COLOR),
        yaxis=dict(tickfont=dict(color="#d1d5db", size=11), gridcolor='rgba(0,0,0,0)'),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  4. STATION ROI (Lollipop chart)
# ═══════════════════════════════════════════════════════════════════════════
def create_station_roi_chart(station_roi):
    """Station ROI lollipop chart with conditional colouring."""
    station_roi = station_roi.sort_values('roi', ascending=True).tail(15)

    colors = []
    for val in station_roi['roi']:
        if val >= 80.0:
            colors.append(COLOR_GREEN)
        elif val >= 50.0:
            colors.append(COLOR_YELLOW)
        else:
            colors.append(COLOR_RED)

    fig = go.Figure()

    # Stems
    for i, (_, row) in enumerate(station_roi.iterrows()):
        fig.add_trace(go.Scatter(
            x=[0, row['roi']], y=[row['police_station'], row['police_station']],
            mode='lines', line=dict(color=colors[i], width=2),
            showlegend=False, hoverinfo='skip'
        ))

    # Dots
    fig.add_trace(go.Scatter(
        x=station_roi['roi'], y=station_roi['police_station'],
        mode='markers', name='ROI',
        marker=dict(size=12, color=colors, line=dict(width=2, color='#0f172a')),
        hovertemplate='%{y}<br>ROI: %{x:.1f}%<extra></extra>'
    ))

    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(text="🏛️ Police Station Enforcement ROI (%)", font=dict(size=17, color="#ffffff", family="Outfit")),
        margin=dict(l=180, r=30, t=50, b=30),
        xaxis=dict(
            title=dict(text="ROI (%)", font=dict(color="#94a3b8")),
            tickfont=dict(color="#94a3b8"), gridcolor=GRID_COLOR,
            range=[0, 105], zeroline=False
        ),
        yaxis=dict(tickfont=dict(color="#d1d5db", size=11), gridcolor='rgba(0,0,0,0)'),
        showlegend=False,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  5. SIMULATOR COMPARISON (Waterfall / grouped bar)
# ═══════════════════════════════════════════════════════════════════════════
def create_simulator_comparison_chart(summary):
    """Scenario A vs Scenario B grouped bar."""
    categories = ['Total AOI (Pre)', 'Baseline Residual', 'AI-Optimized Residual']
    baseline_vals = [summary['total_aoi_before'], summary.get('base_post_aoi', summary['total_aoi_before']*0.8), 0]
    optimized_vals = [0, 0, summary.get('opt_post_aoi', summary['total_aoi_before']*0.6)]

    fig = go.Figure(data=[
        go.Bar(name='Scenario A (Baseline)', x=categories,
               y=[summary['total_aoi_before'], summary.get('base_post_aoi', summary['total_aoi_before']*0.8), 0],
               marker=dict(color='#475569', line=dict(width=1, color='rgba(255,255,255,0.1)'))),
        go.Bar(name='Scenario B (AI-Optimized)', x=categories,
               y=[0, 0, summary.get('opt_post_aoi', summary['total_aoi_before']*0.6)],
               marker=dict(color=COLOR_PRIMARY, line=dict(width=1, color='rgba(255,255,255,0.1)'))),
    ])

    fig.update_layout(
        **_BASE_LAYOUT,
        margin=dict(l=50, r=20, t=50, b=40),
        legend=_DEFAULT_LEGEND,
        barmode='group',
        title=dict(text="⚖️ AOI Residual Comparison", font=dict(size=17, color="#ffffff", family="Outfit")),
        xaxis=dict(tickfont=dict(color="#94a3b8"), gridcolor='rgba(0,0,0,0)'),
        yaxis=dict(title=dict(text="Congestion Index", font=dict(color="#94a3b8")), tickfont=dict(color="#94a3b8"), gridcolor=GRID_COLOR),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  6. VEHICLE PIE (smart top-N + Others grouping, sunburst style)
# ═══════════════════════════════════════════════════════════════════════════
def create_vehicle_pie_chart(df_viols, max_categories=6):
    """Donut pie with top-N categories + Others, polished dark styling."""
    vc = df_viols['clean_vehicle_type'].value_counts()
    top = vc.head(max_categories)
    others = vc.iloc[max_categories:].sum()
    if others > 0:
        top = pd.concat([top, pd.Series({'Others': others})])

    palette = [COLOR_PRIMARY, COLOR_CYAN, COLOR_PURPLE, COLOR_PINK, COLOR_TEAL, COLOR_YELLOW, '#64748b']

    fig = go.Figure(go.Pie(
        labels=top.index, values=top.values,
        hole=0.55,
        marker=dict(colors=palette[:len(top)], line=dict(color='#0f172a', width=2)),
        textinfo='label+percent', textfont=dict(size=11, color='#e2e8f0'),
        hovertemplate='%{label}<br>Count: %{value:,}<br>Share: %{percent}<extra></extra>',
        sort=False
    ))

    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(text="🚗 Violations by Vehicle Type", font=dict(size=15, color="#ffffff", family="Outfit")),
        margin=dict(l=10, r=10, t=45, b=10),
        showlegend=False,
        annotations=[dict(text=f'<b>{int(vc.sum()):,}</b><br><span style="font-size:10px;color:#94a3b8">Total</span>',
                          x=0.5, y=0.5, font_size=18, font_color='#e2e8f0', showarrow=False)]
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  7. RISK TIER DISTRIBUTION (Horizontal stacked 100% bar)
# ═══════════════════════════════════════════════════════════════════════════
def create_risk_distribution_chart(df_forecast):
    """Horizontal stacked bar showing % of cells in each risk tier."""
    def classify(aoi):
        if aoi >= 75: return 'Critical'
        if aoi >= 50: return 'High'
        if aoi >= 25: return 'Moderate'
        return 'Low'

    tiers = df_forecast['AOI'].apply(classify).value_counts()
    total = tiers.sum()
    tier_order = ['Critical', 'High', 'Moderate', 'Low']

    fig = go.Figure()
    cumulative = 0
    for tier in tier_order:
        val = tiers.get(tier, 0)
        pct = val / total * 100 if total > 0 else 0
        fig.add_trace(go.Bar(
            y=['Risk Distribution'], x=[pct],
            name=f'{tier} ({val})',
            orientation='h',
            marker=dict(color=RISK_COLORS[tier]),
            text=f'{pct:.0f}%' if pct >= 5 else '',
            textposition='inside', textfont=dict(color='#ffffff', size=12, family='Inter'),
            hovertemplate=f'{tier}: {val} zones ({pct:.1f}%)<extra></extra>'
        ))
        cumulative += pct

    fig.update_layout(
        **_BASE_LAYOUT,
        barmode='stack',
        title=dict(text="🎯 Cell Risk Distribution", font=dict(size=15, color="#ffffff", family="Outfit")),
        margin=dict(l=10, r=10, t=45, b=10),
        xaxis=dict(
            title=dict(text="Percentage of Cells", font=dict(color="#94a3b8")),
            tickfont=dict(color="#94a3b8"), gridcolor=GRID_COLOR,
            range=[0, 100], ticksuffix='%'
        ),
        yaxis=dict(visible=False),
        height=140,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.15, xanchor="center", x=0.5),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  8. VIOLATIONS BY WEEKDAY (Radar/Polar bar chart)
# ═══════════════════════════════════════════════════════════════════════════
def create_weekday_radar_chart(df_violations):
    """Polar bar chart showing violation volume by day of week."""
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    df_v = df_violations.copy()
    df_v['weekday'] = df_v['created_datetime'].dt.weekday
    counts = df_v.groupby('weekday').size().reindex(range(7), fill_value=0)

    fig = go.Figure(go.Barpolar(
        r=counts.values,
        theta=day_names,
        marker=dict(
            color=counts.values,
            colorscale=[[0, '#1e3a5f'], [0.5, COLOR_PRIMARY], [1, COLOR_CYAN]],
            line=dict(color='rgba(255,255,255,0.15)', width=1),
        ),
        hovertemplate='%{theta}: %{r:,}<extra></extra>'
    ))

    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(text="📅 Violations by Day of Week", font=dict(size=15, color="#ffffff", family="Outfit")),
        margin=dict(l=40, r=40, t=55, b=30),
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(showticklabels=True, tickfont=dict(color="#94a3b8", size=9), gridcolor=GRID_COLOR),
            angularaxis=dict(tickfont=dict(color="#e2e8f0", size=12), gridcolor=GRID_COLOR),
        ),
        showlegend=False,
        height=340,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  9. HOURLY HEATMAP (Matrix chart – hour × weekday)
# ═══════════════════════════════════════════════════════════════════════════
def create_hourly_heatmap(df_violations):
    """Hour-of-day × day-of-week heatmap of violation intensity."""
    df_v = df_violations.copy()
    df_v['hour'] = df_v['created_datetime'].dt.hour
    df_v['weekday'] = df_v['created_datetime'].dt.weekday
    pivot = df_v.groupby(['weekday', 'hour']).size().reset_index(name='count')
    matrix = pivot.pivot(index='weekday', columns='hour', values='count').reindex(range(7)).fillna(0)

    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    fig = go.Figure(go.Heatmap(
        z=matrix.values,
        x=[f'{h:02d}:00' for h in range(24)],
        y=day_names,
        colorscale=[[0, '#0f172a'], [0.25, '#1e3a5f'], [0.5, '#3730a3'], [0.75, COLOR_PRIMARY], [1, COLOR_CYAN]],
        hovertemplate='%{y} %{x}<br>Violations: %{z:,}<extra></extra>',
        colorbar=dict(title=dict(text='Count', font=dict(color='#94a3b8')), tickfont=dict(color='#94a3b8'))
    ))

    fig.update_layout(
        **_BASE_LAYOUT,
        margin=dict(l=50, r=20, t=50, b=40),
        title=dict(text="🔥 Violation Intensity Heatmap (Hour × Day)", font=dict(size=17, color="#ffffff", family="Outfit")),
        xaxis=dict(title=dict(text="Hour", font=dict(color="#94a3b8")), tickfont=dict(color="#94a3b8", size=9), dtick=2),
        yaxis=dict(tickfont=dict(color="#e2e8f0"), autorange='reversed'),
        height=320,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 10. TOP STATIONS BAR (Violations per police station – horizontal)
# ═══════════════════════════════════════════════════════════════════════════
def create_top_stations_chart(df_violations, top_n=10):
    """Horizontal bar chart of top-N stations by violation volume."""
    vc = df_violations['police_station'].value_counts().head(top_n).sort_values()
    n = len(vc)
    colors = [f'rgba(99,102,241,{0.3 + 0.7*(i/max(n-1,1))})' for i in range(n)]

    fig = go.Figure(go.Bar(
        x=vc.values, y=vc.index, orientation='h',
        marker=dict(color=colors, line=dict(color='rgba(255,255,255,0.08)', width=1)),
        text=vc.values, textposition='outside', textfont=dict(color='#e2e8f0', size=11),
        hovertemplate='%{y}: %{x:,} violations<extra></extra>'
    ))

    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(text=f"🏢 Top {top_n} Busiest Police Stations", font=dict(size=17, color="#ffffff", family="Outfit")),
        margin=dict(l=180, r=60, t=50, b=30),
        xaxis=dict(title=dict(text="Violation Count", font=dict(color="#94a3b8")), tickfont=dict(color="#94a3b8"), gridcolor=GRID_COLOR),
        yaxis=dict(tickfont=dict(color="#d1d5db", size=11), gridcolor='rgba(0,0,0,0)'),
        showlegend=False,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 11. AOI GAUGE (Radial gauge for average AOI)
# ═══════════════════════════════════════════════════════════════════════════
def create_aoi_gauge(avg_aoi, label="Avg AOI Score"):
    """Semi-circle gauge for average AOI with gradient colour."""
    if avg_aoi >= 75:
        gauge_color = COLOR_RED
    elif avg_aoi >= 50:
        gauge_color = COLOR_ORANGE
    elif avg_aoi >= 25:
        gauge_color = COLOR_YELLOW
    else:
        gauge_color = COLOR_GREEN

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_aoi,
        number=dict(font=dict(size=38, color='#e2e8f0', family='Outfit'), suffix='/100'),
        title=dict(text=label, font=dict(size=14, color='#94a3b8', family='Outfit')),
        gauge=dict(
            axis=dict(range=[0, 100], tickfont=dict(color='#64748b', size=10), tickcolor='#334155'),
            bar=dict(color=gauge_color, thickness=0.8),
            bgcolor='#1e293b',
            borderwidth=0,
            steps=[
                dict(range=[0, 25], color='rgba(16,185,129,0.12)'),
                dict(range=[25, 50], color='rgba(245,158,11,0.10)'),
                dict(range=[50, 75], color='rgba(249,115,22,0.10)'),
                dict(range=[75, 100], color='rgba(239,68,68,0.12)'),
            ],
            threshold=dict(line=dict(color="#ffffff", width=2), thickness=0.75, value=avg_aoi)
        )
    ))

    fig.update_layout(
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        font=FONT_STYLE, margin=dict(l=30, r=30, t=40, b=20),
        height=230,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 12. VALIDATION STATUS PIE (Approved vs Rejected)
# ═══════════════════════════════════════════════════════════════════════════
def create_validation_pie(df_violations):
    """Simple donut for approved vs rejected violations."""
    vc = df_violations['validation_status'].value_counts()
    colors_map = {'approved': COLOR_GREEN, 'rejected': COLOR_RED}
    colors = [colors_map.get(v, '#64748b') for v in vc.index]

    fig = go.Figure(go.Pie(
        labels=[v.title() for v in vc.index], values=vc.values,
        hole=0.6,
        marker=dict(colors=colors, line=dict(color='#0f172a', width=2)),
        textinfo='label+percent', textfont=dict(size=12, color='#e2e8f0'),
        hovertemplate='%{label}: %{value:,}<extra></extra>',
    ))

    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(text="✅ Ticket Validation Status", font=dict(size=15, color="#ffffff", family="Outfit")),
        margin=dict(l=10, r=10, t=45, b=10),
        showlegend=False,
        height=280,
        annotations=[dict(text='Status', x=0.5, y=0.5, font_size=13, font_color='#94a3b8', showarrow=False)]
    )
    return fig
