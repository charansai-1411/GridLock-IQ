import os
import sys
import numpy as np
import pandas as pd
import h3
from sklearn.cluster import HDBSCAN

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROCESSED_DATA_PATH, H3_RESOLUTION, AOI_SQUASH_K, IEU_ALPHA

def add_h3_cells(df, resolution=H3_RESOLUTION):
    """Maps GPS coordinates into Uber H3 hexagonal cell IDs using fast list comprehension."""
    print(f"Mapping coordinates to H3 cells at resolution {resolution}...")
    # Fast list comprehension to avoid pandas apply overhead
    df['h3_cell'] = [h3.geo_to_h3(lat, lng, resolution) for lat, lng in zip(df['latitude'], df['longitude'])]
    return df

def compute_mass_rolling_3h(df):
    """
    Computes per-record rolling 3-hour mass sum within each H3 cell.
    For each violation, sums the vehicle_mass of all violations recorded
    in the same cell within the preceding 3 hours (inclusive).
    This replaces local_density (violation count) — mass sum is an
    independent signal not inflated by enforcement deployment patterns.
    """
    print("Computing 3h rolling mass sum per H3 cell...")
    df = df.sort_values(['h3_cell', 'created_datetime']).reset_index(drop=True)

    WINDOW_SECS = 3 * 3600
    parts = []
    for cell, grp in df.groupby('h3_cell', sort=False):
        grp = grp.sort_values('created_datetime').copy()
        ts   = (grp['created_datetime'].astype('int64') // 1_000_000_000).values
        mass = grp['vehicle_mass'].values
        j, sums = 0, []
        for i in range(len(ts)):
            while ts[j] < ts[i] - WINDOW_SECS:
                j += 1
            sums.append(mass[j : i + 1].sum())
        grp['mass_3h_sum'] = sums
        parts.append(grp)

    df = pd.concat(parts).sort_index()
    print(f"  mass_3h_sum range: {df['mass_3h_sum'].min():.1f} – {df['mass_3h_sum'].max():.1f}")
    return df


def compute_ieu(df):
    """
    Computes Instantaneous Enforcement Urgency (IEU) on a 0-100 scale.

    Formula:
        IEU_raw = JunctionMultiplier x (mass_3h_sum)^alpha
        IEU     = 100 x (1 - exp(-IEU_raw / k))    [smooth-squash]

    where:
      mass_3h_sum  = total vehicle_mass in the 3h rolling window for this H3 cell
      alpha        = 0.8  (sublinear saturation — first vehicles hurt most)
      JunctionMult = 2.0 at intersections, 1.0 mid-block
      k            = p90_raw / ln(4)  so p90 → IEU=75 (High/Critical boundary)

    Output columns: AOI_raw (=IEU_raw), AOI (=IEU), risk_tier
    Column names kept as AOI_*/risk_tier for dashboard compatibility.
    """
    print("Computing Instantaneous Enforcement Urgency (IEU)...")
    junction_multiplier = df['junction_flag'].apply(lambda x: 2.0 if x == 1 else 1.0)

    # IEU_raw: junction-weighted sublinear mass footprint
    df['AOI_raw'] = junction_multiplier * (df['mass_3h_sum'] ** IEU_ALPHA)

    # Calibrate k: anchor p90 → IEU = 75
    active_raw = df.loc[df['AOI_raw'] > 0, 'AOI_raw']
    if len(active_raw) > 0:
        p50_raw = float(active_raw.quantile(0.50))
        p90_raw = float(active_raw.quantile(0.90))
        k = p90_raw / np.log(4.0) if p90_raw > 0 else AOI_SQUASH_K
    else:
        p50_raw = p90_raw = 0.0
        k = AOI_SQUASH_K

    print(f"IEU smooth-squash (p90->75): p50_raw={p50_raw:.3f}  p90_raw={p90_raw:.3f}  k={k:.4f}")
    p99_raw = float(active_raw.quantile(0.99)) if len(active_raw) > 0 else 0.0
    print(f"  Sanity: f(p50)={100*(1-np.exp(-p50_raw/k)):.1f}  "
          f"f(p90)={100*(1-np.exp(-p90_raw/k)):.1f}  "
          f"f(p99)={100*(1-np.exp(-p99_raw/k)):.1f}")

    df['AOI'] = (100.0 * (1.0 - np.exp(-df['AOI_raw'] / k))).clip(0, 100)

    # Risk tiers (same thresholds as before)
    def assign_risk_tier(aoi):
        if aoi <= 25:  return 'Low'
        elif aoi <= 50: return 'Moderate'
        elif aoi <= 75: return 'High'
        else:           return 'Critical'

    df['risk_tier'] = df['AOI'].apply(assign_risk_tier)
    return df, k


def run_spatial_clustering(df_aggregated, min_aoi_threshold=50.0, eps_meters=200.0):
    """
    Runs HDBSCAN on high-risk H3 cell centroids using haversine metric 
    to cluster corridor-level hotspots.
    """
    print(f"Running HDBSCAN corridor clustering for hotspots with AOI >= {min_aoi_threshold}...")
    
    # Filter high-risk H3 cells
    hotspots = df_aggregated[df_aggregated['max_aoi'] >= min_aoi_threshold].copy()
    if len(hotspots) < 5:
        print("Not enough high-AOI cells to cluster. Assigning noise label.")
        df_aggregated['cluster_label'] = -1
        return df_aggregated

    # Convert coordinates to radians for haversine
    coords_rad = np.radians(hotspots[['latitude', 'longitude']].values)
    
    # Convert epsilon distance in meters to radians (Earth radius = 6,371,000 meters)
    epsilon_rad = eps_meters / 6371000.0
    
    # Instantiate and fit HDBSCAN
    # Use min_cluster_size=5 to identify smaller local corridor groups
    db = HDBSCAN(min_cluster_size=5, min_samples=3, metric='haversine', cluster_selection_epsilon=epsilon_rad)
    cluster_labels = db.fit_predict(coords_rad)
    
    hotspots['cluster_label'] = cluster_labels
    
    # Merge cluster labels back to the main aggregated dataframe
    df_aggregated = df_aggregated.merge(
        hotspots[['h3_cell', 'cluster_label']], on='h3_cell', how='left'
    )
    df_aggregated['cluster_label'] = df_aggregated['cluster_label'].fillna(-1).astype(int)
    
    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"Discovered {num_clusters} high-risk corridors/clusters.")
    
    return df_aggregated

def aggregate_h3_zones(df):
    """Aggregates violation details per H3 cell for maps and dashboard visualizations."""
    print("Aggregates data per H3 cell...")
    
    # Find dominant vehicle type for each H3 cell
    print("Calculating dominant vehicle type per cell...")
    cell_vehicle_counts = df.groupby(['h3_cell', 'clean_vehicle_type']).size().reset_index(name='count')
    dominant_vehicles = cell_vehicle_counts.sort_values(['h3_cell', 'count'], ascending=[True, False])\
                                           .groupby('h3_cell').first().reset_index()
    dominant_vehicle_map = dict(zip(dominant_vehicles['h3_cell'], dominant_vehicles['clean_vehicle_type']))
    
    # Group by cell to compute aggregated metrics
    agg_funcs = {
        'AOI': ['max', 'mean'],
        'duration_mins': 'mean',
        'junction_flag': 'mean',
        'id': 'count'
    }
    
    agg_df = df.groupby('h3_cell').agg(agg_funcs).reset_index()
    # Flatten columns
    agg_df.columns = ['h3_cell', 'max_aoi', 'mean_aoi', 'mean_duration', 'junction_rate', 'violation_count']
    
    # Add coordinates of H3 cell centers
    print("Calculating H3 cell centroids...")
    centroids = [h3.h3_to_geo(cell) for cell in agg_df['h3_cell']]
    agg_df['latitude'] = [c[0] for c in centroids]
    agg_df['longitude'] = [c[1] for c in centroids]
    
    # Map dominant vehicle type
    agg_df['dominant_vehicle_type'] = agg_df['h3_cell'].map(dominant_vehicle_map)
    
    # Classify Risk Tier for the cell based on its max AOI
    def assign_risk_tier(aoi):
        if aoi <= 25:
            return 'Low'
        elif aoi <= 50:
            return 'Moderate'
        elif aoi <= 75:
            return 'High'
        else:
            return 'Critical'
    agg_df['max_risk_tier'] = agg_df['max_aoi'].apply(assign_risk_tier)
    
    return agg_df

def run_spatial_pipeline():
    """Ingests processed Parquet, applies H3 mapping, computes density & AOI, runs HDBSCAN, and saves output."""
    try:
        if not os.path.exists(PROCESSED_DATA_PATH):
            raise FileNotFoundError(f"Cleaned violations data not found at: {PROCESSED_DATA_PATH}. Please run data pipeline first.")
            
        print(f"Loading cleaned violations Parquet from: {PROCESSED_DATA_PATH}...")
        df = pd.read_parquet(PROCESSED_DATA_PATH)
        
        # Step 1: Add H3 hexagons
        df = add_h3_cells(df)

        # Step 2: Compute 3h rolling mass sum per H3 cell (replaces local_density)
        df = compute_mass_rolling_3h(df)

        # Step 3: Calculate IEU (stored as AOI_raw / AOI for dashboard compatibility)
        df, aoi_k = compute_ieu(df)
        
        # Step 4: Save detail records (with calculated H3 cell and AOI) back to cleaned parquet for modeling
        print(f"Overwriting details file with H3 and AOI columns: {PROCESSED_DATA_PATH}...")
        df.to_parquet(PROCESSED_DATA_PATH, index=False, engine='pyarrow')
        
        # Step 5: Aggregate per H3 cell for spatial dashboard metrics
        agg_df = aggregate_h3_zones(df)
        
        # Step 6: Cluster the aggregated hotspots
        agg_df = run_spatial_clustering(agg_df)
        
        # Save aggregated H3 clusters file
        output_agg_path = os.path.join(os.path.dirname(PROCESSED_DATA_PATH), "h3_clustered_zones.parquet")
        print(f"Saving aggregated H3 clustered zones to: {output_agg_path}...")
        agg_df.to_parquet(output_agg_path, index=False, engine='pyarrow')
        
        print("Spatial pipeline completed successfully!")
        
        print("\n--- Spatial Engine Summary ---")
        print(f"Total Unique H3 Cells: {len(agg_df)}")
        print(f"AOI squash k value:    {aoi_k:.2f}")
        print(f"Critical Risk Cells (max AOI > 75): {len(agg_df[agg_df['max_aoi'] > 75])}")
        print(f"High Risk Cells (max AOI 51-75): {len(agg_df[(agg_df['max_aoi'] > 50) & (agg_df['max_aoi'] <= 75)])}")
        
    except Exception as e:
        print(f"Error running spatial pipeline: {e}")
        raise

if __name__ == '__main__':
    run_spatial_pipeline()
