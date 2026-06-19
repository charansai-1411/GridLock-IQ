import os

# Project Path Configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "jan to may police violation_anonymized791b166 (1).csv")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "cleaned_violations.parquet")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Spatial index parameters
H3_RESOLUTION = 7  # average edge length ~1.22km (approx 1km radius)

# Physics-inspired Vehicle Mass Map
# Based on urban traffic lane obstruction capacity:
#   5.0 = tanker/heavy multi-axle (blocks entire lane + turn radius)
#   4.5 = full-size bus (articulated; high dwell time at stops)
#   4.0 = HGV/lorry/heavy goods (rigid body, wide)
#   3.5 = LGV/mini lorry/tractor (mid-size goods)
#   3.0 = car/auto/cab/jeep (standard obstruction unit)
#   2.5 = van/taxi (slightly narrower footprint)
#   2.0 = goods auto (three-wheeler cargo)
#   1.5 = motor cycle (half-lane footprint)
#   1.0 = scooter/moped (minimal footprint)
MASS_MAP = {
    # Heavy vehicles
    'TANKER'              : 5.0,
    'BUS'                 : 4.5,
    'PRIVATE BUS'         : 4.5,
    'TOURIST BUS'         : 4.5,
    'FACTORY BUS'         : 4.5,
    'SCHOOL VEHICLE'      : 4.5,
    'BUS (BMTC/KSRTC)'   : 4.5,
    'HGV'                 : 4.0,
    'LORRY/GOODS VEHICLE' : 4.0,
    # Medium-heavy
    'LGV'                 : 3.5,
    'MINI LORRY'          : 3.5,
    'TRACTOR'             : 3.5,
    'TEMPO'               : 3.0,
    # Standard
    'CAR'                 : 3.0,
    'PASSENGER AUTO'      : 2.0,
    'MAXI-CAB'            : 3.0,
    'JEEP'                : 3.0,
    # Light
    'TAXI'                : 2.5,
    'VAN'                 : 2.5,
    'GOODS AUTO'          : 2.0,
    # Two-wheelers
    'MOTOR CYCLE'         : 1.5,
    'SCOOTER'             : 1.0,
    'MOPED'               : 1.0,
    # Fallback for unknown multi-word types
    'OTHERS'              : 2.0,
}

# Default mass for any type not in MASS_MAP above.
# Set to 2.0 (light four-wheeler) rather than 1.0 (scooter),
# since unknown types are more likely light vehicles than two-wheelers.
DEFAULT_VEHICLE_MASS = 2.0

# Density Calculation parameters
DENSITY_WINDOW_HOURS = 3

# Duration tail cap for non-imputed records.
# p85 of the clean (non-imputed) population is ~90 minutes.
# Records with modified_datetime many hours/days after created_datetime
# are administrative contamination, not real parking duration.
# Cap is applied AFTER imputation — only affects non-imputed records above this threshold.
DURATION_TAIL_CAP_MINS = 90.0

# AOI scaling parameters
# Smooth-squashing normalization: AOI = 100 * (1 - exp(-AOI_raw / k))
# k is calibrated so the median active AOI_raw maps to ~25/100 (Low/Moderate boundary).
# This avoids hard clipping — extreme values asymptotically approach 100 rather than jumping to it.
# k is computed fresh from the data in spatial_engine; this constant is the fallback.
AOI_SQUASH_K = 3000.0

# IEU formula parameters
# Alpha (saturation exponent): sublinear power on mass sum creates diminishing
# marginal disruption — first few parked vehicles degrade capacity severely,
# adding more to an already blocked road yields less marginal harm.
IEU_ALPHA = 0.8

# Prediction horizons in hours
FORECAST_HORIZONS = [1, 2, 4]


