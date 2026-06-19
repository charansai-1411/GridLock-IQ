"""
Build POI tags for each H3 cell using ONLY provided dataset columns:
  - junction_name  (named junction / metro / market / theatre)
  - location       (free-text address — keyword extraction fallback)
  - police_station (area-level fallback)

Output: data/processed/cell_poi_tags.json
  {h3_cell: {poi_label, poi_type, confidence, police_station}}
"""
import pandas as pd
import numpy as np
import h3
import json
import re
from collections import Counter

RAW_CSV = 'data/raw/jan to may police violation_anonymized791b166 (1).csv'

print("Loading raw data...")
raw = pd.read_csv(RAW_CSV,
    usecols=['latitude','longitude','location','junction_name','police_station'],
    low_memory=False)
print(f"  {len(raw):,} total rows")

# Add H3 cell (resolution 7, same as pipeline)
RESOLUTION = 7
raw['h3_cell'] = raw.apply(
    lambda r: h3.geo_to_h3(r['latitude'], r['longitude'], RESOLUTION)
    if pd.notna(r['latitude']) and pd.notna(r['longitude']) else None,
    axis=1
)
raw = raw[raw['h3_cell'].notna()]

# ── TIER 1: junction_name ─────────────────────────────────────────────────────
# Clean: strip the BTP code prefix, keep only the descriptive name
def clean_junction(name):
    if pd.isna(name) or name.strip() == 'No Junction':
        return None
    # Remove "BTPxxx - " prefix
    name = re.sub(r'^BTP\d+\s*-\s*', '', str(name)).strip()
    return name if name else None

raw['junction_clean'] = raw['junction_name'].apply(clean_junction)

# For each cell, get the most frequent non-null junction name
junction_by_cell = (
    raw[raw['junction_clean'].notna()]
    .groupby('h3_cell')['junction_clean']
    .agg(lambda x: x.value_counts().index[0] if len(x) > 0 else None)
    .reset_index()
    .rename(columns={'junction_clean': 'junction_tag'})
)

# Classify junction type from keywords in its name
def classify_junction(name):
    if name is None:
        return 'junction'
    n = name.lower()
    if 'metro' in n or 'station' in n:
        return 'metro_station'
    if 'market' in n or 'bazaar' in n:
        return 'market'
    if 'mall' in n or 'plaza' in n or 'forum' in n:
        return 'mall'
    if 'hospital' in n or 'medical' in n:
        return 'hospital'
    if 'college' in n or 'university' in n or 'institute' in n:
        return 'educational'
    if 'theatre' in n or 'cinema' in n or 'multiplex' in n:
        return 'entertainment'
    if 'circle' in n or 'square' in n:
        return 'traffic_circle'
    if 'bridge' in n:
        return 'bridge'
    return 'junction'

junction_by_cell['junction_type'] = junction_by_cell['junction_tag'].apply(classify_junction)

# ── TIER 2: location keyword extraction ───────────────────────────────────────
LOCATION_KEYWORDS = [
    ('metro',      'metro_station'),
    ('station',    'transit_station'),
    ('mall',       'mall'),
    ('forum',      'mall'),
    ('plaza',      'commercial'),
    ('market',     'market'),
    ('hospital',   'hospital'),
    ('medical',    'hospital'),
    ('college',    'educational'),
    ('university', 'educational'),
    ('institute',  'educational'),
    ('school',     'educational'),
    ('stadium',    'stadium'),
    ('park',       'park'),
    ('bus stand',  'bus_terminal'),
    ('junction',   'junction'),
    ('circle',     'traffic_circle'),
]

def extract_location_keyword(loc_series):
    """Get dominant POI type from a series of location strings for one cell."""
    text = ' '.join(loc_series.dropna().str.lower().tolist())
    for kw, poi_type in LOCATION_KEYWORDS:
        if kw in text:
            # Find the most common snippet containing the keyword
            matches = [l for l in loc_series.dropna() if kw in l.lower()]
            if matches:
                # Take first component of the address that contains the keyword
                for m in matches[:5]:
                    parts = m.split(',')
                    for p in parts:
                        if kw in p.lower():
                            return p.strip(), poi_type
    return None, None

loc_keywords = []
for cell, grp in raw.groupby('h3_cell'):
    label, ptype = extract_location_keyword(grp['location'])
    if label:
        loc_keywords.append({'h3_cell': cell, 'loc_label': label, 'loc_type': ptype})
loc_df = pd.DataFrame(loc_keywords)

# ── TIER 3: police station (always available) ─────────────────────────────────
station_by_cell = (
    raw.groupby('h3_cell')['police_station']
    .agg(lambda x: x.value_counts().index[0] if len(x) > 0 else None)
    .reset_index()
    .rename(columns={'police_station': 'police_station'})
)

# ── MERGE: prefer Tier 1 > Tier 2 > Tier 3 ───────────────────────────────────
all_cells = raw['h3_cell'].unique()
result = pd.DataFrame({'h3_cell': all_cells})
result = result.merge(junction_by_cell, on='h3_cell', how='left')
result = result.merge(loc_df, on='h3_cell', how='left') if len(loc_df) > 0 else result.assign(loc_label=None, loc_type=None)
result = result.merge(station_by_cell, on='h3_cell', how='left')

TYPE_ORD = {
    'metro_station':   4,
    'market':          3,
    'mall':            3,
    'commercial':      3,
    'entertainment':   2,
    'transit_station': 2,
    'junction':        2,
    'traffic_circle':  2,
    'bridge':          2,
    'bus_terminal':    2,
    'hospital':        1,
    'educational':     1,
    'park':            1,
    'stadium':         1,
    'area':            1,
}

def build_poi_entry(row):
    if pd.notna(row.get('junction_tag')):
        ptype = row['junction_type']
        return {
            'poi_label': row['junction_tag'],
            'poi_type':  ptype,
            'junction_type_ord': TYPE_ORD.get(ptype, 1),
            'source':    'junction_name',
            'police_station': row.get('police_station')
        }
    elif pd.notna(row.get('loc_label')):
        ptype = row.get('loc_type', 'landmark')
        return {
            'poi_label': row['loc_label'],
            'poi_type':  ptype,
            'junction_type_ord': TYPE_ORD.get(ptype, 1),
            'source':    'location',
            'police_station': row.get('police_station')
        }
    else:
        return {
            'poi_label': row.get('police_station', 'Unknown Area'),
            'poi_type':  'area',
            'junction_type_ord': 1,
            'source':    'police_station',
            'police_station': row.get('police_station')
        }

poi_tags = {}
for _, row in result.iterrows():
    poi_tags[row['h3_cell']] = build_poi_entry(row)

print(f"\nPOI tags built for {len(poi_tags):,} cells")
print(f"  Tier 1 (junction_name): {junction_by_cell['junction_tag'].notna().sum():,} cells")
print(f"  Tier 2 (location kw):   {len(loc_df):,} cells")
print(f"  Tier 3 (police stn):    {station_by_cell['police_station'].notna().sum():,} cells")

# ── POI type breakdown ────────────────────────────────────────────────────────
type_counts = Counter(v['poi_type'] for v in poi_tags.values())
print("\nPOI type distribution:")
for t, n in type_counts.most_common():
    print(f"  {t:<20}: {n:,}")

# Save
with open('data/processed/cell_poi_tags.json', 'w', encoding='utf-8') as f:
    json.dump(poi_tags, f, indent=2, ensure_ascii=False)
print(f"\nSaved → data/processed/cell_poi_tags.json")

# ── Show top hotspot cells with their tags ────────────────────────────────────
print("\n=== Top cells by violation count with POI tags ===")
cell_counts = raw['h3_cell'].value_counts().head(20)
for cell, count in cell_counts.items():
    tag = poi_tags.get(cell, {})
    label = tag.get('poi_label', '?')
    ptype = tag.get('poi_type', '?')
    src   = tag.get('source', '?')
    print(f"  {cell}  {count:>6} violations  [{ptype}] {label}  (src={src})")
