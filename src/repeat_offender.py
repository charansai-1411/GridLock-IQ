import os
import sys
import numpy as np
import pandas as pd

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROCESSED_DATA_PATH

# Output file path
REPEAT_OFFENDERS_PATH = os.path.join(os.path.dirname(PROCESSED_DATA_PATH), "repeat_offenders.parquet")

def compute_repeat_offenders():
    """
    Identifies repeat offenders, computes their Repeat Risk Score,
    and categorizes them into High, Medium, and Monitor priority tiers.
    Saves the output to repeat_offenders.parquet.
    """
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError(f"Cleaned violations parquet not found at {PROCESSED_DATA_PATH}. Run spatial engine first.")
        
    print(f"Loading cleaned violations dataset from: {PROCESSED_DATA_PATH}...")
    df = pd.read_parquet(PROCESSED_DATA_PATH)
    
    # 1. Clean vehicle numbers
    print("Cleaning vehicle numbers...")
    df['clean_vehicle_num'] = df['vehicle_number'].fillna('').str.strip().str.upper()
    
    # Filter out empty or noise vehicle numbers
    noise_patterns = ['', 'UNKNOWN', 'NAN', 'N/A', 'NULL', 'TEMP', 'TEST']
    valid_df = df[~df['clean_vehicle_num'].isin(noise_patterns) & (df['clean_vehicle_num'].str.len() > 3)]
    
    print(f"Valid violations with vehicle number: {len(valid_df)} out of {len(df)} total.")
    
    # 2. Determine the time window for recency (last 30 days based on maximum date in dataset)
    max_date = valid_df['created_datetime'].max()
    recency_threshold = max_date - pd.Timedelta(days=30)
    print(f"Recency time threshold (last 30 days): {recency_threshold} to {max_date}")
    
    # 3. Compute stats per vehicle
    print("Aggregating metrics per vehicle...")
    
    # Pre-calculate recent violations
    valid_df['is_recent'] = valid_df['created_datetime'] >= recency_threshold
    
    vehicle_groups = valid_df.groupby('clean_vehicle_num')
    
    offenders = vehicle_groups.agg(
        violation_count=('id', 'count'),
        unique_locations_count=('h3_cell', 'nunique'),
        recent_violations_count=('is_recent', 'sum'),
        last_violation_time=('created_datetime', 'max'),
        last_police_station=('police_station', 'last'),
        last_h3_cell=('h3_cell', 'last'),
        last_latitude=('latitude', 'last'),
        last_longitude=('longitude', 'last'),
        dominant_vehicle_type=('clean_vehicle_type', lambda x: x.mode()[0] if not x.mode().empty else 'SCOOTER')
    ).reset_index()
    
    # Calculate recency weight
    offenders['recency_weight'] = offenders['recent_violations_count'] / offenders['violation_count']
    
    # 4. Compute Risk Score
    # Formula: Repeat_Risk_Score = (violation_count * 0.5) + (unique_locations_count * 0.3) + (recency_weight * 0.2)
    print("Computing Risk Scores...")
    offenders['risk_score'] = (
        (offenders['violation_count'] * 0.5) + 
        (offenders['unique_locations_count'] * 0.3) + 
        (offenders['recency_weight'] * 0.2)
    )
    
    # 5. Classify Risk Tiers
    # Risk Tier:
    #   Score >= 10  =  HIGH PRIORITY (immediate towing on next detection)
    #   Score 5-9    =  MEDIUM PRIORITY (flagged for checkpoint stops)
    #   Score < 5    =  MONITOR
    def assign_priority_tier(score):
        if score >= 10.0:
            return 'HIGH PRIORITY'
        elif score >= 5.0:
            return 'MEDIUM PRIORITY'
        else:
            return 'MONITOR'
            
    offenders['priority_tier'] = offenders['risk_score'].apply(assign_priority_tier)
    
    # Sort by risk score descending
    offenders = offenders.sort_values('risk_score', ascending=False).reset_index(drop=True)
    
    # Save the repeat offenders database
    print(f"Saving {len(offenders)} repeat offender profiles to: {REPEAT_OFFENDERS_PATH}...")
    offenders.to_parquet(REPEAT_OFFENDERS_PATH, index=False, engine='pyarrow')
    
    print("\n--- Repeat Offender Summary ---")
    print(f"Total Unique Vehicles Tracked: {len(offenders)}")
    print(f"HIGH PRIORITY Vehicles: {len(offenders[offenders['priority_tier'] == 'HIGH PRIORITY'])}")
    print(f"MEDIUM PRIORITY Vehicles: {len(offenders[offenders['priority_tier'] == 'MEDIUM PRIORITY'])}")
    print(f"MONITOR Tiers: {len(offenders[offenders['priority_tier'] == 'MONITOR'])}")
    
    # Print top 5 offenders
    print("\nTop 5 Repeat Offenders:")
    print(offenders[['clean_vehicle_num', 'violation_count', 'unique_locations_count', 'risk_score', 'priority_tier']].head(5).to_string(index=False))
    
    return offenders

if __name__ == '__main__':
    compute_repeat_offenders()
