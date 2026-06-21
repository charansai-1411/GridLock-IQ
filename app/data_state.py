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
    df_forecast_hours = None
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
        try:
            import duckdb
            con = duckdb.connect()
            df_forecast_hours = con.execute(f"SELECT DISTINCT hour_dt FROM '{FORECAST_RESULTS_PATH}' ORDER BY hour_dt").df()
            df_forecast_hours["hour_dt"] = pd.to_datetime(df_forecast_hours["hour_dt"])
            con.close()
        except Exception as e:
            print(f"Error loading unique hours via DuckDB: {e}")

    if os.path.exists(REPEAT_OFFENDERS_PATH):
        # Load only the top 5000 repeat offenders to save ~88MB memory
        df_repeat_offenders = pd.read_parquet(REPEAT_OFFENDERS_PATH).head(5000)

    return df_clustered_zones, df_forecast_hours, df_repeat_offenders, station_map


def clear_data_cache():
    load_all_data.clear()


def ensure_data_loaded():
    """Populate session data when a Streamlit page is opened directly."""
    current_versions = get_data_file_versions()
    if st.session_state.get("df_forecast_hours") is not None:
        # Already loaded — check if files have changed since last load
        if st.session_state.get("_data_versions") == current_versions:
            # Files are unchanged, but check if user query parameters changed!
            selected_time = st.session_state.get("selected_time")
            time_window = st.session_state.get("time_window")
            
            # If the sliced data is missing or query parameters changed, run update_filtered_data()
            if (st.session_state.get("df_forecast") is None or
                selected_time != st.session_state.get("_last_selected_time") or
                time_window != st.session_state.get("_last_time_window")):
                
                update_filtered_data()
                st.session_state["_last_selected_time"] = selected_time
                st.session_state["_last_time_window"] = time_window
            return  # Files unchanged; session state is current
        # Files changed: clear stale state and reload
        for key in ("df_clustered_zones", "df_forecast_hours", "df_forecast",
                    "df_repeat_offenders", "station_map",
                    "filtered_forecast", "filtered_violations",
                    "selected_time", "time_window", "_data_versions",
                    "_last_selected_time", "_last_time_window"):
            st.session_state.pop(key, None)
        load_all_data.clear()

    df_clustered_zones, df_forecast_hours, df_repeat_offenders, station_map = load_all_data(
        get_data_file_versions()
    )

    st.session_state["df_clustered_zones"] = df_clustered_zones
    st.session_state["df_forecast_hours"] = df_forecast_hours
    st.session_state["df_repeat_offenders"] = df_repeat_offenders
    st.session_state["station_map"] = station_map
    st.session_state["_data_versions"] = current_versions  # fingerprint for change detection

    if df_forecast_hours is None or df_forecast_hours.empty:
        st.session_state["selected_time"] = datetime.now()
        st.session_state["filtered_forecast"] = None
        st.session_state["filtered_violations"] = None
        return

    if "selected_time" not in st.session_state:
        # Default to the default simulation target hour (2024-03-12 03:00:00 UTC) if available,
        # otherwise use the max available hour.
        available_datetimes = pd.to_datetime(df_forecast_hours["hour_dt"])
        default_dt = pd.Timestamp("2024-03-12 03:00:00", tz="UTC")
        if (available_datetimes == default_dt).any():
            st.session_state["selected_time"] = default_dt
        else:
            st.session_state["selected_time"] = available_datetimes.max()

    if "time_window" not in st.session_state:
        st.session_state["time_window"] = "Single Hour"

    update_filtered_data()
    st.session_state["_last_selected_time"] = st.session_state["selected_time"]
    st.session_state["_last_time_window"] = st.session_state["time_window"]


def update_filtered_data():
    selected_time = st.session_state.get("selected_time")
    window_opt = st.session_state.get("time_window", "Single Hour")
    df_forecast_hours = st.session_state.get("df_forecast_hours")

    if selected_time is None or df_forecast_hours is None or df_forecast_hours.empty:
        st.session_state["filtered_forecast"] = None
        st.session_state["df_forecast"] = None
        st.session_state["filtered_violations"] = None
        return

    ref_ts = pd.to_datetime(selected_time)
    
    # Align timezone of ref_ts to the parquet's timezone (tz)
    tz = df_forecast_hours["hour_dt"].dt.tz
    if tz is not None:
        if ref_ts.tzinfo is None:
            ref_ts = ref_ts.tz_localize(tz)
        else:
            ref_ts = ref_ts.tz_convert(tz)
    else:
        if ref_ts.tzinfo is not None:
            ref_ts = ref_ts.tz_localize(None)

    # Calculate start_ts and end_ts based on window_opt
    if window_opt == "Single Hour":
        # Query the entire day of ref_ts to support daily trend charts
        start_ts = ref_ts.normalize()
        end_ts = ref_ts.normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
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
        # All Time - query max available range of hours from metadata
        start_ts = pd.to_datetime(df_forecast_hours["hour_dt"].min())
        end_ts = ref_ts

    # Convert to ISO string format for SQL query (always query in UTC as stored in database)
    start_ts_utc = start_ts.tz_convert("UTC") if start_ts.tzinfo else start_ts
    end_ts_utc = end_ts.tz_convert("UTC") if end_ts.tzinfo else end_ts
    start_str = start_ts_utc.strftime("%Y-%m-%d %H:%M:%S")
    end_str = end_ts_utc.strftime("%Y-%m-%d %H:%M:%S")

    import duckdb
    con = duckdb.connect()
    
    # Read only required columns
    query = f"""
        SELECT h3_cell, hour_dt, violation_count, latitude, longitude,
               cell_vehicle_mass, junction_flag, AOI, historical_density,
               pred_t1, pred_t2, pred_t4
        FROM '{FORECAST_RESULTS_PATH}'
        WHERE hour_dt >= '{start_str}' AND hour_dt <= '{end_str}'
    """
    
    try:
        sub_forecast = con.execute(query).df()
        # Parse hour_dt to datetime
        sub_forecast["hour_dt"] = pd.to_datetime(sub_forecast["hour_dt"])
        if tz is not None:
            if sub_forecast["hour_dt"].dt.tz is None:
                sub_forecast["hour_dt"] = sub_forecast["hour_dt"].dt.tz_localize("UTC").dt.tz_convert(tz)
            else:
                sub_forecast["hour_dt"] = sub_forecast["hour_dt"].dt.tz_convert(tz)
        
        # Downcast floats to float32 and junction_flag to int8 to save memory
        float_cols = [
            "violation_count", "latitude", "longitude", "cell_vehicle_mass",
            "AOI", "historical_density", "pred_t1", "pred_t2", "pred_t4"
        ]
        sub_forecast["junction_flag"] = sub_forecast["junction_flag"].astype("int8")
        sub_forecast[float_cols] = sub_forecast[float_cols].astype("float32")
    except Exception as e:
        print(f"Error querying DuckDB: {e}")
        sub_forecast = pd.DataFrame()

    con.close()

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
        # If Single Hour, filter sub_forecast (which covers the full day) to only the active ref_ts hour
        if window_opt == "Single Hour":
            agg_forecast = sub_forecast[sub_forecast["hour_dt"] == ref_ts].copy()
        else:
            agg_forecast = sub_forecast.copy()
            
        if not agg_forecast.empty:
            agg_forecast["AOI_max"] = agg_forecast["AOI"]
        else:
            # If empty, return df with correct column schemas to prevent KeyErrors on subpages
            agg_forecast = pd.DataFrame(columns=[
                "h3_cell", "hour_dt", "violation_count", "latitude", "longitude",
                "cell_vehicle_mass", "junction_flag", "AOI", "historical_density",
                "pred_t1", "pred_t2", "pred_t4", "AOI_max"
            ])
            agg_forecast["h3_cell"] = agg_forecast["h3_cell"].astype(str)
            agg_forecast["hour_dt"] = pd.to_datetime(agg_forecast["hour_dt"])
            agg_forecast["junction_flag"] = agg_forecast["junction_flag"].astype("int8")
            for col in ["violation_count", "latitude", "longitude", "cell_vehicle_mass",
                        "AOI", "historical_density", "pred_t1", "pred_t2", "pred_t4", "AOI_max"]:
                agg_forecast[col] = agg_forecast[col].astype("float32")

    if sub_forecast.empty:
        sub_forecast = pd.DataFrame(columns=[
            "h3_cell", "hour_dt", "violation_count", "latitude", "longitude",
            "cell_vehicle_mass", "junction_flag", "AOI", "historical_density",
            "pred_t1", "pred_t2", "pred_t4"
        ])
        sub_forecast["h3_cell"] = sub_forecast["h3_cell"].astype(str)
        sub_forecast["hour_dt"] = pd.to_datetime(sub_forecast["hour_dt"])
        sub_forecast["junction_flag"] = sub_forecast["junction_flag"].astype("int8")
        for col in ["violation_count", "latitude", "longitude", "cell_vehicle_mass",
                    "AOI", "historical_density", "pred_t1", "pred_t2", "pred_t4"]:
            sub_forecast[col] = sub_forecast[col].astype("float32")

    st.session_state["df_forecast"] = sub_forecast
    st.session_state["filtered_forecast"] = agg_forecast
    st.session_state["filtered_violations"] = None
