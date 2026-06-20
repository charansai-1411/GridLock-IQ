import os
from datetime import datetime
import json
import pandas as pd
import streamlit as st

from config import PROCESSED_DATA_PATH

PROCESSED_DIR = os.path.dirname(PROCESSED_DATA_PATH)
CLUSTERED_ZONES_PATH = os.path.join(PROCESSED_DIR, "h3_clustered_zones.parquet")
FORECAST_RESULTS_PATH = os.path.join(PROCESSED_DIR, "forecast_results.parquet")
REPEAT_OFFENDERS_PATH = os.path.join(PROCESSED_DIR, "repeat_offenders.parquet")
STATION_MAP_PATH = os.path.join(PROCESSED_DIR, "station_map.json")


def get_data_file_versions():
    """Build a cache key from data file modification times."""
    paths = (
        CLUSTERED_ZONES_PATH,
        FORECAST_RESULTS_PATH,
        REPEAT_OFFENDERS_PATH,
    )
    return tuple((path, os.path.getmtime(path) if os.path.exists(path) else None) for path in paths)


@st.cache_data
def load_all_data(data_file_versions):
    _ = data_file_versions
    df_clustered_zones = None
    df_forecast = None
    df_repeat_offenders = None
    station_map = {}

    # Load station map from static precompiled JSON to save startup computation and memory
    if os.path.exists(STATION_MAP_PATH):
        try:
            with open(STATION_MAP_PATH, "r") as f:
                station_map = json.load(f)
        except Exception as e:
            print(f"Error loading station_map.json: {e}")

    if os.path.exists(CLUSTERED_ZONES_PATH):
        df_clustered_zones = pd.read_parquet(CLUSTERED_ZONES_PATH)

    if os.path.exists(FORECAST_RESULTS_PATH):
        # Load only the 12 columns referenced by the app pages
        FORECAST_COLS = [
            "h3_cell", "hour_dt", "violation_count", "latitude", "longitude",
            "cell_vehicle_mass", "junction_flag", "AOI", "historical_density",
            "pred_t1", "pred_t2", "pred_t4"
        ]
        df_forecast = pd.read_parquet(FORECAST_RESULTS_PATH, columns=FORECAST_COLS)
        df_forecast["hour_dt"] = pd.to_datetime(df_forecast["hour_dt"])
        
        # Downcast floats to float32 and junction_flag to int8 to save over 120MB memory
        float_cols = [
            "violation_count", "latitude", "longitude", "cell_vehicle_mass",
            "AOI", "historical_density", "pred_t1", "pred_t2", "pred_t4"
        ]
        df_forecast["junction_flag"] = df_forecast["junction_flag"].astype("int8")
        df_forecast[float_cols] = df_forecast[float_cols].astype("float32")

    if os.path.exists(REPEAT_OFFENDERS_PATH):
        # Load only the top 5000 repeat offenders to save ~88MB memory
        df_repeat_offenders = pd.read_parquet(REPEAT_OFFENDERS_PATH).head(5000)

    return df_clustered_zones, df_forecast, df_repeat_offenders, station_map


def clear_data_cache():
    load_all_data.clear()


def ensure_data_loaded():
    """Populate session data when a Streamlit page is opened directly.
    
    Compares the current data file modification times against the version stored
    in session state. If the files have changed (e.g. pipeline was re-run), the
    cached data is discarded and reloaded so stale results don't persist mid-session.
    """
    current_versions = get_data_file_versions()
    if st.session_state.get("df_forecast") is not None:
        # Already loaded — check if files have changed since last load
        if st.session_state.get("_data_versions") == current_versions:
            return  # Files unchanged; session state is current
        # Files changed: clear stale state and reload
        for key in ("df_clustered_zones", "df_forecast",
                    "df_repeat_offenders", "station_map",
                    "filtered_forecast", "filtered_violations",
                    "selected_time", "time_window", "_data_versions"):
            st.session_state.pop(key, None)
        load_all_data.clear()

    df_clustered_zones, df_forecast, df_repeat_offenders, station_map = load_all_data(
        get_data_file_versions()
    )

    st.session_state["df_clustered_zones"] = df_clustered_zones
    st.session_state["df_forecast"] = df_forecast
    st.session_state["df_repeat_offenders"] = df_repeat_offenders
    st.session_state["station_map"] = station_map
    st.session_state["_data_versions"] = current_versions  # fingerprint for change detection

    if df_forecast is None:
        st.session_state["selected_time"] = datetime.now()
        st.session_state["filtered_forecast"] = None
        st.session_state["filtered_violations"] = None
        return

    if "selected_time" not in st.session_state:
        # Default to the single hour with the highest total city-wide violation count.
        hourly_totals = df_forecast.groupby("hour_dt")["violation_count"].sum()
        if not hourly_totals.empty:
            st.session_state["selected_time"] = hourly_totals.idxmax()
        else:
            st.session_state["selected_time"] = df_forecast["hour_dt"].max()

    if "time_window" not in st.session_state:
        st.session_state["time_window"] = "Single Hour"

    update_filtered_data()


def update_filtered_data():
    df_forecast = st.session_state.get("df_forecast")
    selected_time = st.session_state.get("selected_time")

    if df_forecast is None or selected_time is None:
        st.session_state["filtered_forecast"] = None
        st.session_state["filtered_violations"] = None
        return

    ref_ts = pd.to_datetime(selected_time)
    if df_forecast["hour_dt"].dt.tz is not None and ref_ts.tzinfo is None:
        ref_ts = ref_ts.tz_localize(df_forecast["hour_dt"].dt.tz)
    elif df_forecast["hour_dt"].dt.tz is None and ref_ts.tzinfo is not None:
        ref_ts = ref_ts.tz_localize(None)

    window_opt = st.session_state.get("time_window", "Single Hour")
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
    else:
        start_ts = df_forecast["hour_dt"].min()
        end_ts = ref_ts

    sub_forecast = df_forecast[(df_forecast["hour_dt"] >= start_ts) & (df_forecast["hour_dt"] <= end_ts)]

    if window_opt != "Single Hour" and not sub_forecast.empty:
        agg_forecast = sub_forecast.groupby("h3_cell").agg(
            AOI=("AOI", "mean"),
            AOI_max=("AOI", "max"),
            pred_t1=("pred_t1", "mean"),
            pred_t2=("pred_t2", "mean"),
            pred_t4=("pred_t4", "mean"),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            cell_vehicle_mass=("cell_vehicle_mass", "mean"),
            junction_flag=("junction_flag", "max"),
            violation_count=("violation_count", "sum"),
            historical_density=("historical_density", "mean"),
        ).reset_index()
        agg_forecast["hour_dt"] = ref_ts
    else:
        agg_forecast = sub_forecast.copy()
        agg_forecast["AOI_max"] = agg_forecast["AOI"]

    st.session_state["filtered_forecast"] = agg_forecast
    st.session_state["filtered_violations"] = None
