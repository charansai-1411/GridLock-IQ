"""
GridLock IQ — Hotspot Intelligence Map
Leaflet.js + CartoDB Dark tiles (reliable in any iframe, no auth needed).
H3 hexagons as GeoJSON overlays. Operational dispatch UI.
"""
import streamlit as st
import pandas as pd
import numpy as np
import h3
import json
import os
import sys
import math

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.data_state import ensure_data_loaded
from src.patrol_optimizer import get_h3_to_station_map


# ─── POI tags ─────────────────────────────────────────────────────────────────
@st.cache_data
def _load_poi_tags():
    path = os.path.join(PROJECT_ROOT, "data", "processed", "cell_poi_tags.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


# ─── H3 → GeoJSON ring ────────────────────────────────────────────────────────
def _cell_geojson_ring(cell_id):
    """Return GeoJSON polygon ring [[lng,lat], ...] closed. GeoJSON is [lng,lat]."""
    try:
        verts = h3.h3_to_geo_boundary(cell_id)   # returns ((lat,lng), ...)
        ring = [[float(lng), float(lat)] for lat, lng in verts]
        ring.append(ring[0])                       # close the ring
        return ring
    except Exception:
        return []


def _zone_class(cell_id, df_all):
    cd = df_all[df_all["h3_cell"] == cell_id]
    if cd.empty:
        return "LOW RISK"
    active_pct = (cd["AOI"] > 5.0).sum() / max(len(cd), 1)
    mean_aoi   = float(cd["AOI"].mean())
    max_aoi    = float(cd["AOI"].max())
    if mean_aoi >= 12.0 and active_pct >= 0.35:
        return "STRUCTURAL"
    if max_aoi >= 50.0 and active_pct < 0.35:
        return "EVENT-DRIVEN"
    return "LOW RISK"


# ─── Build GeoJSON FeatureCollection ─────────────────────────────────────────
def build_geojson(df_fc_window, df_fc_all, df_viol, station_map, poi_tags):
    """
    Returns a GeoJSON FeatureCollection string for all active H3 cells.
    Each feature carries all zone properties needed by the JS zone card.
    """
    if df_fc_window is None or df_fc_window.empty:
        return '{"type":"FeatureCollection","features":[]}'

    agg = (
        df_fc_window.groupby("h3_cell")
        .agg(
            ieu     =("AOI",             "mean"),
            pred_t1 =("pred_t1",         "mean"),
            pred_t2 =("pred_t2",         "mean"),
            pred_t4 =("pred_t4",         "mean"),
            viol_cnt=("violation_count", "sum"),
        )
        .reset_index()
    )
    agg = agg[agg["ieu"] > 1.0].copy()

    # ── Vehicle breakdown ─────────────────────────────────────────────────
    cell_veh: dict = {}
    if df_viol is not None and not df_viol.empty:
        vc = (
            df_viol.groupby(["h3_cell", "clean_vehicle_type"])
            .size().reset_index(name="cnt")
        )
        for cid, grp in vc.groupby("h3_cell"):
            tot  = grp["cnt"].sum()
            top  = grp.nlargest(4, "cnt")
            rows = [{"type": r["clean_vehicle_type"],
                     "pct":  round(r["cnt"] / tot * 100)}
                    for _, r in top.iterrows()]
            others = 100 - sum(v["pct"] for v in rows)
            if others > 0:
                rows.append({"type": "Others", "pct": others})
            cell_veh[cid] = rows

    # ── Hourly profile (sparkline) ────────────────────────────────────────
    cell_hourly: dict = {}
    if df_viol is not None and not df_viol.empty:
        dv = df_viol.copy()
        if dv["created_datetime"].dt.tz is not None:
            dv["local_hour"] = dv["created_datetime"].dt.tz_convert("Asia/Kolkata").dt.hour
        else:
            dv["local_hour"] = dv["created_datetime"].dt.hour
        hc = dv.groupby(["h3_cell", "local_hour"]).size().reset_index(name="cnt")
        for cid, grp in hc.groupby("h3_cell"):
            profile = [0] * 24
            for _, r in grp.iterrows():
                profile[int(r["local_hour"])] = int(r["cnt"])
            cell_hourly[cid] = profile

    # ── Assemble features ─────────────────────────────────────────────────
    features = []
    for _, row in agg.iterrows():
        cid  = row["h3_cell"]
        ieu  = float(row["ieu"])
        pt1  = float(row["pred_t1"])
        pt2  = float(row["pred_t2"])
        pt4  = float(row["pred_t4"])

        if ieu >= 65:
            tier="Critical"; fc="#ff4444"; fo=0.65; bc="#ff0000"; bw=2.5
        elif ieu >= 50:
            tier="High";     fc="#ff8c00"; fo=0.50; bc="#ff6600"; bw=2.0
        elif ieu >= 25:
            tier="Moderate"; fc="#f0c040"; fo=0.35; bc="#d4a800"; bw=1.5
        elif ieu > 1:
            tier="Low";      fc="#2ecc71"; fo=0.20; bc="#27ae60"; bw=1.0
        else:
            continue

        poi   = poi_tags.get(cid, {})
        label = poi.get("poi_label", "")
        if not label:
            si    = station_map.get(cid, {})
            label = (si.get("police_station", "") if isinstance(si, dict) else "") or f"Zone {cid[:10]}"

        si    = station_map.get(cid, {})
        juris = si.get("police_station", "Unknown") if isinstance(si, dict) else "Unknown"
        zc    = _zone_class(cid, df_fc_all)

        ring  = _cell_geojson_ring(cid)
        if not ring:
            continue

        veh_list  = cell_veh.get(cid, [{"type": "CAR", "pct": 100}])
        dominant  = veh_list[0]["type"] if veh_list else "CAR"
        units     = max(1, math.ceil(pt1 / 25.0))

        props = {
            "cell_id":         cid,
            "junction_name":   label,
            "jurisdiction":    juris,
            "zone_class":      zc,
            "ieu":             round(ieu, 1),
            "pred_t1":         round(pt1, 1),
            "pred_t2":         round(pt2, 1),
            "pred_t4":         round(pt4, 1),
            "tier":            tier,
            "violations_total": int(row["viol_cnt"]),
            "vehicles":        veh_list,
            "dominant_vehicle": dominant,
            "hourly_profile":  cell_hourly.get(cid, [0] * 24),
            "units_needed":    units,
            "fill_color":      fc,
            "fill_opacity":    fo,
            "border_color":    bc,
            "border_width":    bw,
        }

        features.append({
            "type": "Feature",
            "properties": props,
            "geometry": {
                "type": "Polygon",
                "coordinates": [ring]
            }
        })

    features.sort(key=lambda f: f["properties"]["ieu"], reverse=True)
    fc_obj = {"type": "FeatureCollection", "features": features}
    return json.dumps(fc_obj, ensure_ascii=False)


# ─── HTML template (Leaflet) ──────────────────────────────────────────────────
_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0f1117;font-family:Inter,sans-serif;overflow:hidden;height:100vh}
#container{display:flex;width:100%;height:100vh}

/* MAP */
#map-side{position:relative;flex:0 0 68%;height:100%}
#map{width:100%;height:100%;background:#0f1117}
.leaflet-container{background:#0f1117!important}

/* FILTER CHIPS */
#filter-chips{position:absolute;top:10px;left:10px;z-index:1000;display:flex;gap:7px;flex-wrap:wrap}
.chip{display:flex;align-items:center;gap:5px;padding:6px 13px;border-radius:20px;
  cursor:pointer;font-size:12px;font-weight:700;user-select:none;
  backdrop-filter:blur(8px);transition:all .2s;border:1px solid rgba(255,255,255,.2);
  box-shadow:0 2px 8px rgba(0,0,0,.4)}
.chip.on{color:#fff}
.chip.off{background:rgba(10,12,20,.85)!important;color:rgba(255,255,255,.3)!important;
  border-color:rgba(255,255,255,.08)!important}
#chip-Critical{background:rgba(255,68,68,.85)}
#chip-High    {background:rgba(255,140,0,.85)}
#chip-Moderate{background:rgba(220,175,50,.9);color:#111}
#chip-Low     {background:rgba(46,204,113,.85)}

/* LEGEND */
#legend{position:absolute;bottom:36px;left:14px;z-index:1000;
  background:rgba(13,15,22,.9);backdrop-filter:blur(6px);
  border-radius:8px;padding:12px 14px;width:158px;
  border:1px solid rgba(255,255,255,.06)}
#legend h4{color:#8899aa;font-size:10px;font-weight:700;text-transform:uppercase;
  letter-spacing:.08em;margin-bottom:8px}
.lr{display:flex;align-items:center;gap:8px;color:#e2e8f0;font-size:11px;margin-bottom:5px}
.ld{width:10px;height:10px;border-radius:50%;flex-shrink:0}

/* RE-CENTER */
#recenter{position:absolute;bottom:36px;right:10px;z-index:1000;
  background:rgba(13,15,22,.9);border:1px solid rgba(255,255,255,.2);
  color:#fff;width:34px;height:34px;border-radius:6px;cursor:pointer;
  font-size:17px;display:flex;align-items:center;justify-content:center;
  backdrop-filter:blur(4px);box-shadow:0 2px 8px rgba(0,0,0,.4)}
#recenter:hover{background:rgba(79,142,247,.35);border-color:#4f8ef7}

/* TOOLTIP */
#tooltip{position:fixed;z-index:9999;pointer-events:none;display:none;width:266px;
  background:#1a1d26;border:1px solid rgba(255,255,255,.13);border-radius:8px;
  padding:14px;box-shadow:0 8px 32px rgba(0,0,0,.55)}
.tt-title{color:#fff;font-size:13px;font-weight:700;margin-bottom:9px;
  padding-bottom:8px;border-bottom:1px solid rgba(255,255,255,.09)}
.tt-row{display:flex;justify-content:space-between;align-items:center;
  margin-bottom:6px;font-size:12px}
.tt-lbl{color:#8899aa}
.tt-val{color:#fff;font-weight:600}
.tt-bw{display:flex;align-items:center;gap:6px}
.tt-bar-bg{width:78px;height:6px;background:rgba(255,255,255,.1);border-radius:3px;overflow:hidden}
.tt-bar-fg{height:100%;border-radius:3px}
.tt-hint{color:#4f8ef7;font-size:11px;margin-top:10px;text-align:center}

/* ZONE CARD */
#card-side{flex:0 0 32%;background:#12151f;border-left:1px solid rgba(255,255,255,.07);
  overflow-y:auto;padding:16px 14px;height:100%}
#card-prompt{display:flex;flex-direction:column;align-items:center;justify-content:center;
  min-height:180px;text-align:center;color:#8899aa;font-size:13px;line-height:1.6}
#card-prompt .icon{font-size:34px;margin-bottom:10px}
.dd-wrap{margin-top:12px}
.dd-lbl{color:#8899aa;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em}
#zone-dd{width:100%;margin-top:6px;padding:8px 10px;background:#1a1d26;
  border:1px solid rgba(255,255,255,.12);color:#fff;border-radius:6px;
  font-size:12px;font-family:Inter,sans-serif;cursor:pointer;outline:none}
#zone-dd:focus{border-color:#4f8ef7}
#zc{display:none;padding-top:4px}

/* Card elements */
.zc-name{color:#fff;font-size:15px;font-weight:800;font-family:Outfit,Inter,sans-serif;margin-bottom:2px}
.zc-sub{color:#8899aa;font-size:12px;margin-bottom:12px}
.zc-badge{border-radius:6px;padding:12px 14px;margin-bottom:12px}
.zc-badge-tier{font-size:11px;font-weight:800;letter-spacing:.06em;text-transform:uppercase;margin-bottom:5px}
.zc-badge-score{font-size:26px;font-weight:900;font-family:Outfit,sans-serif;margin-bottom:5px}
.bar-bg{width:100%;height:7px;background:rgba(255,255,255,.1);border-radius:4px;overflow:hidden;margin-top:2px}
.bar-fg{height:100%;border-radius:4px}
.sec{color:#8899aa;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;margin:12px 0 7px}
.pred-row{display:flex;align-items:center;gap:7px;margin-bottom:6px}
.pred-lbl{color:#8899aa;font-size:11px;font-weight:700;width:32px;flex-shrink:0}
.pred-bg{flex:1;height:7px;background:rgba(255,255,255,.08);border-radius:4px;overflow:hidden}
.pred-fg{height:100%;border-radius:4px}
.pred-num{color:#fff;font-size:11px;font-weight:700;width:52px;text-align:right}
.pred-arr{font-size:11px;width:12px}
.veh-row{display:flex;align-items:center;gap:6px;margin-bottom:5px}
.veh-name{color:#8899aa;font-size:11px;width:68px;flex-shrink:0}
.veh-bg{flex:1;height:6px;background:rgba(255,255,255,.07);border-radius:3px;overflow:hidden}
.veh-fg{height:100%;border-radius:3px}
.veh-pct{color:#e2e8f0;font-size:11px;font-weight:600;width:32px;text-align:right}
canvas#spark{width:100%;height:54px;margin-top:4px;display:block}
.action{background:rgba(255,68,68,.09);border-left:4px solid #ff4444;
  border-radius:6px;padding:12px 14px;margin-top:12px}
.action-t{color:#ff4444;font-size:10px;font-weight:800;text-transform:uppercase;
  letter-spacing:.07em;margin-bottom:6px}
.action-b{color:#fff;font-size:13px;font-weight:700;line-height:1.6}
.zc-footer{margin-top:12px;padding-top:10px;border-top:1px solid rgba(255,255,255,.06);
  color:#8899aa;font-size:11px;line-height:1.7}
.zc-cid{font-size:9px;color:rgba(255,255,255,.15);margin-top:4px;font-family:monospace;word-break:break-all}
</style>
</head>
<body>
<!-- TOOLTIP -->
<div id="tooltip">
  <div class="tt-title" id="tt-name"></div>
  <div class="tt-row">
    <span class="tt-lbl">Risk Score</span>
    <div class="tt-bw">
      <div class="tt-bar-bg"><div class="tt-bar-fg" id="tt-bf"></div></div>
      <span class="tt-val" id="tt-sc"></span>
    </div>
  </div>
  <div class="tt-row"><span class="tt-lbl">Status</span><span class="tt-val" id="tt-st"></span></div>
  <div class="tt-row"><span class="tt-lbl">Violations</span><span class="tt-val" id="tt-vi"></span></div>
  <div class="tt-row"><span class="tt-lbl">Primary</span><span class="tt-val" id="tt-vt"></span></div>
  <div class="tt-row"><span class="tt-lbl">Jurisdiction</span><span class="tt-val" id="tt-ju"></span></div>
  <div class="tt-hint">Click to open full zone card →</div>
</div>

<div id="container">
  <!-- LEFT MAP -->
  <div id="map-side">
    <div id="filter-chips">
      <div class="chip on" id="chip-Critical" onclick="toggleTier('Critical')">🔴 Critical <span id="cnt-Critical">0</span></div>
      <div class="chip on" id="chip-High"     onclick="toggleTier('High')"    >🟠 High <span id="cnt-High">0</span></div>
      <div class="chip on" id="chip-Moderate" onclick="toggleTier('Moderate')">🟡 Moderate <span id="cnt-Moderate">0</span></div>
      <div class="chip on" id="chip-Low"      onclick="toggleTier('Low')"     >🟢 Low <span id="cnt-Low">0</span></div>
    </div>
    <div id="map"></div>
    <button id="recenter" onclick="reCenter()" title="Re-center Bengaluru">⌖</button>
    <div id="legend">
      <h4>Risk Level</h4>
      <div class="lr"><div class="ld" style="background:#ff4444"></div><span>Critical &nbsp;≥65</span></div>
      <div class="lr"><div class="ld" style="background:#ff8c00"></div><span>High &nbsp;&nbsp;&nbsp;&nbsp;50–64</span></div>
      <div class="lr"><div class="ld" style="background:#f0c040"></div><span>Moderate 25–49</span></div>
      <div class="lr"><div class="ld" style="background:#2ecc71"></div><span>Low &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1–24</span></div>
      <div class="lr"><div class="ld" style="background:transparent;border:1px solid rgba(255,255,255,.25)"></div><span>Inactive &nbsp;0</span></div>
    </div>
  </div>

  <!-- RIGHT ZONE CARD -->
  <div id="card-side">
    <div id="card-prompt">
      <div class="icon">🗺️</div>
      <div>Click any zone on the map<br>to inspect it</div>
      <div style="color:#4f8ef7;font-size:11px;margin-top:6px">Or select from the dropdown below</div>
    </div>
    <div class="dd-wrap">
      <div class="dd-lbl">Top Hotspots by Risk Score</div>
      <select id="zone-dd" onchange="selectFromDd(this.value)">
        <option value="">— Select a zone —</option>
      </select>
    </div>
    <div id="zc"></div>
  </div>
</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
var FC = __GEOJSON__;
var TIME_WINDOW = "__TIME_WINDOW__";

var map = L.map('map', {
  center: [12.9716, 77.5946],
  zoom: 12,
  zoomControl: true,
  attributionControl: true
});

// CartoDB Dark Matter tiles — reliable globally, great contrast
L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
  subdomains: 'abcd',
  maxZoom: 19
}).addTo(map);

// ── State ──────────────────────────────────────────────────────────────
var _active = {Critical:true, High:true, Moderate:true, Low:true};
var _hexLayer = null;
var TIER_ICON = {Critical:'🔴', High:'🟠', Moderate:'🟡', Low:'🟢'};

// ── Draw hexagons ──────────────────────────────────────────────────────
_hexLayer = L.geoJSON(FC, {
  style: function(f) {
    var p = f.properties;
    return {
      fillColor:   p.fill_color,
      fillOpacity: p.fill_opacity,
      color:       p.border_color,
      weight:      p.border_width,
      opacity:     1
    };
  },
  onEachFeature: function(f, layer) {
    layer.on({
      mouseover: function(e) {
        _showTt(f.properties, e.originalEvent);
        layer.setStyle({weight: f.properties.border_width + 1.5, fillOpacity: Math.min(f.properties.fill_opacity + 0.15, 0.9)});
      },
      mouseout: function() {
        _hideTt();
        layer.setStyle({weight: f.properties.border_width, fillOpacity: f.properties.fill_opacity});
      },
      click: function() { openCard(f.properties); }
    });
  }
}).addTo(map);

// Tooltip follows mouse globally
document.addEventListener('mousemove', function(e) {
  var tt = document.getElementById('tooltip');
  if (tt.style.display === 'block') {
    var x = e.clientX + 18, y = e.clientY - 10;
    if (x + 278 > window.innerWidth)  x = e.clientX - 280;
    if (y + 230 > window.innerHeight) y = e.clientY - 232;
    tt.style.left = x + 'px';
    tt.style.top  = y + 'px';
  }
});

// ── Filter chips ───────────────────────────────────────────────────────
function toggleTier(tier) {
  _active[tier] = !_active[tier];
  var chip = document.getElementById('chip-' + tier);
  chip.classList.toggle('on',  _active[tier]);
  chip.classList.toggle('off', !_active[tier]);
  _hexLayer.eachLayer(function(l) {
    if (!l.feature) return;
    var p = l.feature.properties;
    if (p.tier !== tier) return;
    if (_active[tier]) {
      l.setStyle({fillOpacity: p.fill_opacity, opacity: 1, weight: p.border_width});
    } else {
      l.setStyle({fillOpacity: 0, opacity: 0});
    }
  });
}

// ── Chip counts ────────────────────────────────────────────────────────
(function() {
  var cnt = {Critical:0, High:0, Moderate:0, Low:0};
  FC.features.forEach(function(f) { var t=f.properties.tier; if(cnt[t]!==undefined)cnt[t]++; });
  Object.keys(cnt).forEach(function(t) {
    var el = document.getElementById('cnt-' + t);
    if (el) el.textContent = cnt[t];
  });
})();

// ── Dropdown ───────────────────────────────────────────────────────────
(function() {
  var dd = document.getElementById('zone-dd');
  FC.features.forEach(function(f) {
    var p = f.properties;
    if (p.ieu < 25) return;
    var opt = document.createElement('option');
    opt.value = p.cell_id;
    opt.textContent = p.junction_name + ' (' + p.tier + ')';
    dd.appendChild(opt);
  });
})();

function selectFromDd(cid) {
  if (!cid) return;
  var feat = FC.features.find(function(f) { return f.properties.cell_id === cid; });
  if (feat) openCard(feat.properties);
}

// ── Tooltip ────────────────────────────────────────────────────────────
function _showTt(p) {
  document.getElementById('tt-name').textContent = '📍 ' + p.junction_name;
  var bf = document.getElementById('tt-bf');
  bf.style.width = Math.min(Math.round(p.ieu), 100) + '%';
  bf.style.background = p.fill_color;
  document.getElementById('tt-sc').textContent = p.ieu.toFixed(0) + '/100';
  document.getElementById('tt-st').textContent = (TIER_ICON[p.tier]||'') + ' ' + p.tier.toUpperCase();
  document.getElementById('tt-vi').textContent = p.violations_total.toLocaleString() + ' this window';
  var v0 = p.vehicles[0] || {type:'CAR', pct:100};
  document.getElementById('tt-vt').textContent = v0.type + ' (' + v0.pct + '%)';
  document.getElementById('tt-ju').textContent = p.jurisdiction;
  document.getElementById('tooltip').style.display = 'block';
}
function _hideTt() { document.getElementById('tooltip').style.display = 'none'; }

// ── Zone Card ──────────────────────────────────────────────────────────
function openCard(p) {
  _hideTt();
  document.getElementById('card-prompt').style.display = 'none';
  document.getElementById('zone-dd').value = p.cell_id;
  var zc = document.getElementById('zc');
  zc.style.display = 'block';
  zc.innerHTML = _cardHTML(p);
  setTimeout(function() { _drawSpark(p); }, 40);
}

function _arr(a, b) {
  if (a > b + 3) return {s:'↑', c:'#ff4444'};
  if (a < b - 3) return {s:'↓', c:'#2ecc71'};
  return {s:'→', c:'#8899aa'};
}

function _cardHTML(p) {
  var col = p.fill_color;
  var a1 = _arr(p.pred_t1, p.ieu);
  var a2 = _arr(p.pred_t2, p.ieu);
  var a4 = _arr(p.pred_t4, p.ieu);

  var vehRows = p.vehicles.map(function(v) {
    return '<div class="veh-row">' +
      '<span class="veh-name">' + v.type + '</span>' +
      '<div class="veh-bg"><div class="veh-fg" style="width:' + v.pct + '%;background:' + col + '90"></div></div>' +
      '<span class="veh-pct">' + v.pct + '%</span>' +
    '</div>';
  }).join('');

  var predRows = [
    {lbl:'T+1h', val:p.pred_t1, a:a1, op:1.0},
    {lbl:'T+2h', val:p.pred_t2, a:a2, op:0.75},
    {lbl:'T+4h', val:p.pred_t4, a:a4, op:0.5}
  ].map(function(r) {
    return '<div class="pred-row">' +
      '<span class="pred-lbl">' + r.lbl + '</span>' +
      '<div class="pred-bg"><div class="pred-fg" style="width:' + Math.min(r.val,100) + '%;background:' + col + ';opacity:' + r.op + '"></div></div>' +
      '<span class="pred-num">' + r.val.toFixed(0) + '/100</span>' +
      '<span class="pred-arr" style="color:' + r.a.c + '">' + r.a.s + '</span>' +
    '</div>';
  }).join('');

  return (
    '<div class="zc-name">📍 ' + p.junction_name + '</div>' +
    '<div class="zc-sub">' + p.jurisdiction + ' Jurisdiction</div>' +

    '<div class="zc-badge" style="background:' + col + '18;border-left:4px solid ' + col + '">' +
      '<div class="zc-badge-tier" style="color:' + col + '">' + p.tier.toUpperCase() + ' — ' + p.zone_class + '</div>' +
      '<div class="zc-badge-score" style="color:' + col + '">' + p.ieu.toFixed(0) +
        '<span style="font-size:13px;color:#8899aa">/100</span></div>' +
      '<div class="bar-bg"><div class="bar-fg" style="width:' + Math.min(p.ieu,100) + '%;background:' + col + '"></div></div>' +
    '</div>' +

    '<div class="sec">Predicted Next Hour</div>' + predRows +
    '<div class="sec">Violation Breakdown</div>' + vehRows +

    '<div class="sec">Peak Hours (Historical)</div>' +
    '<canvas id="spark" width="300" height="54"></canvas>' +

    '<div class="action">' +
      '<div class="action-t">⚡ Recommended Action</div>' +
      '<div class="action-b">Deploy ' + p.units_needed + ' unit' + (p.units_needed!==1?'s':'') +
        ' from ' + p.jurisdiction + ' PS immediately<br>' +
        '<span style="color:#8899aa;font-size:12px;font-weight:400">Primary target: ' + p.dominant_vehicle + 's</span></div>' +
    '</div>' +

    '<div class="zc-footer">' +
      'Total violations (window): ' + p.violations_total.toLocaleString() + '<br>Window: ' + TIME_WINDOW +
    '</div>' +
    '<div class="zc-cid">' + p.cell_id + '</div>'
  );
}

function _drawSpark(p) {
  var cv = document.getElementById('spark');
  if (!cv) return;
  var W = cv.offsetWidth || 300, H = 54;
  cv.width = W; cv.height = H;
  var ctx = cv.getContext('2d');
  var data = p.hourly_profile;
  var max = Math.max.apply(null, data.concat([1]));
  var bw = W / 24;
  data.forEach(function(v, i) {
    var h = (v / max) * (H - 6);
    var isPeak = (i >= 7 && i <= 9) || (i >= 17 && i <= 19);
    ctx.fillStyle = isPeak ? 'rgba(240,192,64,.6)' : 'rgba(79,142,247,.45)';
    ctx.fillRect(i * bw + 1, H - h, bw - 2, h);
  });
  ctx.strokeStyle = 'rgba(255,255,255,.07)';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0, H-0.5); ctx.lineTo(W, H-0.5); ctx.stroke();
}

function reCenter() {
  map.setView([12.9716, 77.5946], 12);
}
</script>
</body>
</html>
"""


def _build_html(geojson_str: str, time_window: str) -> str:
    return (
        _HTML
        .replace("__GEOJSON__", geojson_str)
        .replace("__TIME_WINDOW__", time_window.replace('"', "'"))
    )


# ─── Page render ──────────────────────────────────────────────────────────────
def render_hotspot_map():
    ensure_data_loaded()
    poi_tags = _load_poi_tags()

    df_viol = st.session_state.get("df_violations")
    df_fc   = st.session_state.get("df_forecast")

    if df_viol is None or df_fc is None:
        st.error("Data not loaded. Please reload.")
        return

    if "station_map" not in st.session_state or not st.session_state["station_map"]:
        st.session_state["station_map"] = get_h3_to_station_map(df_viol)
    station_map = st.session_state["station_map"]

    df_fc = df_fc.copy()
    df_fc["hour_dt"] = pd.to_datetime(df_fc["hour_dt"])

    if "hotspot_time_window" not in st.session_state:
        st.session_state["hotspot_time_window"] = "Last 7 Days"

    # ── Header ────────────────────────────────────────────────────────────
    st.markdown(
        """<div style="background:#0f1117;border-bottom:1px solid rgba(255,255,255,.07);
        padding:0 6px;height:40px;display:flex;align-items:center;
        justify-content:space-between;margin:-1rem -1rem .8rem -1rem;
        font-family:Outfit,Inter,sans-serif;">
        <span style="font-size:15px;font-weight:800;color:#fff;">🗺️ &nbsp;Hotspot Intelligence Map</span>
        <span style="font-size:11px;color:#8899aa;">Bengaluru · Real-time dispatch view</span>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Controls ──────────────────────────────────────────────────────────
    col_tw, col_btn, _ = st.columns([1.4, 0.6, 5])
    window_options = ["Single Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days"]
    with col_tw:
        tw = st.selectbox(
            "Time Window", window_options,
            index=window_options.index(st.session_state["hotspot_time_window"]),
            key="hotspot_tw_sel", label_visibility="collapsed",
        )
        st.session_state["hotspot_time_window"] = tw

    with col_btn:
        if st.button("🔄 Reload", key="hotspot_reload"):
            st.cache_data.clear()
            st.rerun()

    # ── Filter forecast window ────────────────────────────────────────────
    peak_ts = df_fc.groupby("hour_dt")["AOI"].sum().idxmax()

    if tw == "Single Hour":
        fc_window = df_fc[df_fc["hour_dt"] == peak_ts]
    elif tw == "Last 6 Hours":
        fc_window = df_fc[(df_fc["hour_dt"] >= peak_ts - pd.Timedelta(hours=5)) &
                          (df_fc["hour_dt"] <= peak_ts)]
    elif tw == "Last 24 Hours":
        fc_window = df_fc[(df_fc["hour_dt"] >= peak_ts - pd.Timedelta(hours=23)) &
                          (df_fc["hour_dt"] <= peak_ts)]
    else:
        fc_window = df_fc[(df_fc["hour_dt"] >= peak_ts - pd.Timedelta(days=7)) &
                          (df_fc["hour_dt"] <= peak_ts)]

    if fc_window is None or fc_window.empty:
        fc_window = df_fc

    # ── Build + render ────────────────────────────────────────────────────
    geojson_str = build_geojson(fc_window, df_fc, df_viol, station_map, poi_tags)
    html_content = _build_html(geojson_str, tw)
    st.components.v1.html(html_content, height=700, scrolling=False)

    # ── Summary ───────────────────────────────────────────────────────────
    try:
        import json as _json
        features = _json.loads(geojson_str)["features"]
        tiers = {t: sum(1 for f in features if f["properties"]["tier"] == t)
                 for t in ["Critical", "High", "Moderate", "Low"]}
        st.caption(
            f"🔴 {tiers['Critical']} Critical · "
            f"🟠 {tiers['High']} High · "
            f"🟡 {tiers['Moderate']} Moderate · "
            f"🟢 {tiers['Low']} Low · "
            f"{len(features)} active zones · Window: {tw}"
        )
    except Exception:
        pass


render_hotspot_map()
