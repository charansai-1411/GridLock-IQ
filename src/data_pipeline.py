import os
import pandas as pd
import numpy as np
import sys

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RAW_DATA_PATH, PROCESSED_DATA_DIR, PROCESSED_DATA_PATH, MASS_MAP, DEFAULT_VEHICLE_MASS, DURATION_TAIL_CAP_MINS

def load_raw_data(csv_path):
    """Loads the raw violation logs from CSV."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Raw data file not found at: {csv_path}")
    print(f"Loading raw data from: {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns.")
    return df

def clean_data(df):
    """
    Cleans raw dataframe, parses datetimes, computes duration, 
    maps vehicle masses, and prepares features.
    """
    print("Preprocessing timestamps...")
    # Parse datetimes using mixed format to handle sub-second formatting variations
    df['created_datetime'] = pd.to_datetime(df['created_datetime'], errors='coerce', format='mixed')
    df['closed_datetime'] = pd.to_datetime(df.get('closed_datetime'), errors='coerce', format='mixed')
    df['modified_datetime'] = pd.to_datetime(df['modified_datetime'], errors='coerce', format='mixed')

    # Calculate obstruction duration using the real close time when available.
    # Some exports leave closed_datetime empty, so modified_datetime remains the fallback proxy.
    print("Computing obstruction durations...")
    df['effective_end_datetime'] = df['closed_datetime'].combine_first(df['modified_datetime'])
    df['duration_mins'] = (df['effective_end_datetime'] - df['created_datetime']).dt.total_seconds() / 60.0
    
    # Identify duration anomalies: negative durations or durations > 24 hours (1440 mins)
    invalid_mask = (df['duration_mins'] < 0) | (df['duration_mins'] > 1440) | df['duration_mins'].isna()
    df['is_duration_imputed'] = invalid_mask
    
    # Compute station-wise median duration for imputation, falling back to overall median
    overall_median = df.loc[~invalid_mask, 'duration_mins'].median()
    if pd.isna(overall_median):
        overall_median = 15.0  # fallback default
    
    print(f"Overall median duration for valid entries: {overall_median:.2f} minutes.")
    
    station_medians = df[~invalid_mask].groupby('police_station')['duration_mins'].median()
    
    def impute_duration(row):
        if row['is_duration_imputed']:
            station = row['police_station']
            if pd.notna(station) and station in station_medians:
                return station_medians[station]
            return overall_median
        return row['duration_mins']
    
    df['duration_mins'] = df.apply(impute_duration, axis=1)
    print(f"Imputed {invalid_mask.sum()} anomalous/missing duration records.")
    
    # Apply tail cap to non-imputed records only.
    # Records where modified_datetime is hours/days later than created_datetime
    # are administrative contamination, not real parking duration.
    # Imputed records already use station medians (~13 min) — cap doesn't affect them.
    tail_mask = (~df['is_duration_imputed']) & (df['duration_mins'] > DURATION_TAIL_CAP_MINS)
    capped_count = tail_mask.sum()
    df.loc[tail_mask, 'duration_mins'] = DURATION_TAIL_CAP_MINS
    print(f"Tail-capped {capped_count} non-imputed records above {DURATION_TAIL_CAP_MINS:.0f} min.")
    
    # Vehicle Mass mapping (using vehicle_type, falling back to updated_vehicle_type)
    print("Mapping vehicle masses...")
    df['clean_vehicle_type'] = df['vehicle_type'].fillna(df['updated_vehicle_type']).fillna('SCOOTER').str.upper().str.strip()

    # Normalise common spelling/alias variants so they hit the correct MASS_MAP key
    type_replacements = {
        # Two-wheelers
        'TWO WHEELER'          : 'MOTOR CYCLE',
        'TWO-WHEELER'          : 'MOTOR CYCLE',
        'MOTERCYCLE'           : 'MOTOR CYCLE',
        'MOTORIZED CYCLE'      : 'MOTOR CYCLE',
        'MOTOR BIKE'           : 'MOTOR CYCLE',
        # Scooters / mopeds
        'E SCOOTER'            : 'SCOOTER',
        'E-SCOOTER'            : 'SCOOTER',
        # Autos / three-wheelers
        'THREE WHEELER'        : 'PASSENGER AUTO',
        'E-RICKSHAW'           : 'PASSENGER AUTO',
        'E RICKSHAW'           : 'PASSENGER AUTO',
        'GOODS AUTO'           : 'GOODS AUTO',   # keep — mapped separately
        # Cabs / taxis
        'CAB'                  : 'TAXI',
        'PRIVATE CAR'          : 'CAR',
        'CAR/PASSENGER'        : 'CAR',
        # Maxi-cab variants
        'MAXI CAB'             : 'MAXI-CAB',
        'MAXICAB'              : 'MAXI-CAB',
        # Heavy vehicles
        'HGV'                  : 'HGV',
        'LORRY'                : 'LORRY/GOODS VEHICLE',
        'LORRY/GOODS VEHICLE'  : 'LORRY/GOODS VEHICLE',
        'MINI LORRY'           : 'MINI LORRY',
        'LIGHT GOODS VEHICLE'  : 'LGV',
        'LIGHT MOTOR VEHICLE'  : 'LGV',
        # Buses
        'PRIVATE BUS'          : 'PRIVATE BUS',
        'TOURIST BUS'          : 'TOURIST BUS',
        'SCHOOL VEHICLE'       : 'SCHOOL VEHICLE',
        'SCHOOL BUS'           : 'SCHOOL VEHICLE',
        'MINI BUS'             : 'MINI LORRY',
        'BUS(BMTC/KSRTC)'     : 'BUS (BMTC/KSRTC)',
        'BMTC'                 : 'BUS (BMTC/KSRTC)',
        # Misc
        'OTHERS'               : 'OTHERS',
        'OTHER'                : 'OTHERS',
        'UNKNOWN'              : 'OTHERS',
    }
    df['clean_vehicle_type'] = df['clean_vehicle_type'].replace(type_replacements)
    
    df['vehicle_mass'] = df['clean_vehicle_type'].map(MASS_MAP).fillna(DEFAULT_VEHICLE_MASS)
    
    # Compute Junction Flag
    print("Computing junction flags...")
    # If junction_name is missing or is "No Junction", then junction_flag is 0, else 1
    df['junction_flag'] = df['junction_name'].apply(
        lambda x: 0 if pd.isna(x) or str(x).strip().lower() in ['no junction', 'nan', ''] else 1
    )
    
    # Drop columns that are completely null or not useful to save space
    drop_cols = ['description', 'action_taken_timestamp']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Filter out rejected and duplicate records — these are invalid tickets that
    # inflate local violation counts and corrupt density/IEU calculations.
    # Records with null validation_status (42%) are kept as unreviewed-but-valid.
    INVALID_STATUSES = {'rejected', 'duplicate'}
    before = len(df)
    df = df[~df['validation_status'].isin(INVALID_STATUSES)].copy()
    print(f"Filtered {before - len(df):,} rejected/duplicate records. "
          f"Remaining: {len(df):,}")

    return df

def run_pipeline():
    """Runs the full ETL data pipeline."""
    try:
        df = load_raw_data(RAW_DATA_PATH)
        df_cleaned = clean_data(df)
        
        # Ensure output directory exists
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        
        print(f"Saving cleaned dataset to Parquet format at: {PROCESSED_DATA_PATH}...")
        df_cleaned.to_parquet(PROCESSED_DATA_PATH, index=False, engine='pyarrow')
        print("Data pipeline completed successfully!")
        
        # Output summary stats
        print("\n--- Processed Dataset Summary ---")
        print(f"Total Rows: {len(df_cleaned)}")
        print(f"Mean Duration: {df_cleaned['duration_mins'].mean():.2f} mins")
        print(f"Median Duration: {df_cleaned['duration_mins'].median():.2f} mins")
        print(f"Junction Adjacent Violations: {df_cleaned['junction_flag'].sum()} ({df_cleaned['junction_flag'].mean()*100:.2f}%)")
        print(f"Validation Status Breakdown:\n{df_cleaned['validation_status'].value_counts(dropna=False)}")
        
    except Exception as e:
        print(f"Error executing data pipeline: {e}")
        raise

if __name__ == '__main__':
    run_pipeline()
