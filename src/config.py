"""
Shared configuration for CycloVision AI.

Central place for:
  - IMD (India Meteorological Department) cyclone intensity categories
  - File paths
  - Model/image hyperparameters

Keeping these in one file means every script (preprocessing, training,
dashboard) agrees on the same category boundaries and paths.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

INSAT_DIR = DATA_RAW / "insat"
ERA5_DIR = DATA_RAW / "era5"
IBTRACS_DIR = DATA_RAW / "ibtracs"

# ---------------------------------------------------------------------------
# IMD cyclone intensity categories
# Source: India Meteorological Department (RSMC New Delhi) classification,
# based on maximum sustained surface wind speed (3-minute average), in km/h.
#
# TERMINOLOGY NOTE FOR PRESENTATIONS & PAPERS:
# IMD formally defines a 7-tier scale for cyclonic disturbances:
#   Tier 1: Depression (D)
#   Tier 2: Deep Depression (DD)
#   Tier 3: Cyclonic Storm (CS)
#   Tier 4: Severe Cyclonic Storm (SCS)
#   Tier 5: Very Severe Cyclonic Storm (VSCS)
#   Tier 6: Extremely Severe Cyclonic Storm (ESCS)
#   Tier 7: Super Cyclonic Storm (SuCS)
#
# In this codebase, we also include Tier 0: "Low Pressure Area" (pre-cyclonic
# baseline, <31 km/h) so the system can track early-stage disturbances before
# they reach Depression grade. This yields 8 discrete classification bins
# (NUM_CATEGORIES = 8). When presenting to judges, refer to this as:
# "IMD's 7-tier cyclone intensity scale, plus a pre-cyclonic Low Pressure Area baseline".
# ---------------------------------------------------------------------------
IMD_CATEGORIES = [
    # (label,                          min_kmh, max_kmh)
    ("Low Pressure Area",                0,   30),
    ("Depression",                      31,   50),
    ("Deep Depression",                 51,   62),
    ("Cyclonic Storm",                  63,   88),
    ("Severe Cyclonic Storm",           89,  117),
    ("Very Severe Cyclonic Storm",     118,  167),
    ("Extremely Severe Cyclonic Storm",168,  221),
    ("Super Cyclonic Storm",           222, 10_000),
]

# Convenience lookups
CATEGORY_NAMES = [c[0] for c in IMD_CATEGORIES]
NUM_CATEGORIES = len(IMD_CATEGORIES)

# ---------------------------------------------------------------------------
# IMD 7-Tier Scale — Full Official Metadata
# Each entry corresponds to category index 0-7 (index 0 = Low Pressure Area,
# indices 1-7 = the official IMD 7 tiers).
#
# Fields:
#   wind_kmph   : official sustained wind range string (km/h)
#   wind_knots  : official sustained wind range string (knots)
#   t_number    : Dvorak T-number
#   sea_condition: state of sea
#   wave_height  : significant wave height (metres)
#   action       : official action advisory
# ---------------------------------------------------------------------------
IMD_SCALE = {
    0: {  # Low Pressure Area (pre-cyclonic, not in official 7-tier)
        "wind_kmph":    "< 31",
        "wind_knots":   "< 17",
        "t_number":     "< 1.5",
        "sea_condition": "Slight / Moderate",
        "wave_height":  "< 1.25 m",
        "action":       "Weather watch bulletin issued. Routine monitoring.",
    },
    1: {  # Tier 1 — Depression
        "wind_kmph":    "31 – 49",
        "wind_knots":   "17 – 27",
        "t_number":     "1.5",
        "sea_condition": "Moderate to Rough",
        "wave_height":  "1.25 – 4.0 m",
        "action":       "Fishermen advised not to venture into the open seas.",
    },
    2: {  # Tier 2 — Deep Depression
        "wind_kmph":    "50 – 61",
        "wind_knots":   "28 – 33",
        "t_number":     "2.0",
        "sea_condition": "Very Rough",
        "wave_height":  "4.0 – 6.0 m",
        "action":       "Fishermen advised not to venture into the open seas.",
    },
    3: {  # Tier 3 — Cyclonic Storm
        "wind_kmph":    "62 – 87",
        "wind_knots":   "34 – 47",
        "t_number":     "2.5 – 3.0",
        "sea_condition": "High",
        "wave_height":  "6.0 – 9.0 m",
        "action":       "Total suspension of fishing operations.",
    },
    4: {  # Tier 4 — Severe Cyclonic Storm
        "wind_kmph":    "88 – 117",
        "wind_knots":   "48 – 63",
        "t_number":     "3.5",
        "sea_condition": "Very High",
        "wave_height":  "9.0 – 14.0 m",
        "action":       "Total suspension of fishing operations.",
    },
    5: {  # Tier 5 — Very Severe Cyclonic Storm
        "wind_kmph":    "118 – 167",
        "wind_knots":   "64 – 90",
        "t_number":     "4.0 – 4.5",
        "sea_condition": "Phenomenal",
        "wave_height":  "Over 14.0 m",
        "action":       "Total suspension of fishing operations.",
    },
    6: {  # Tier 6 — Extremely Severe Cyclonic Storm
        "wind_kmph":    "168 – 221",
        "wind_knots":   "91 – 119",
        "t_number":     "5.0 – 6.0",
        "sea_condition": "Phenomenal",
        "wave_height":  "Over 14.0 m",
        "action":       "Total suspension of fishing operations.",
    },
    7: {  # Tier 7 — Super Cyclonic Storm
        "wind_kmph":    "222 and above",
        "wind_knots":   "120 and above",
        "t_number":     "> 6.5",
        "sea_condition": "Phenomenal",
        "wave_height":  "Over 14.0 m",
        "action":       "Total suspension of fishing operations.",
    },
}


def wind_speed_to_category(wind_kmh: float) -> int:
    """
    Map a sustained wind speed (km/h) to an IMD category index (0-7).
    Uses continuous threshold boundaries to prevent floating-point gap drop-through
    (e.g., 90 knots = 166.68 km/h falling between 166 and 167 km/h).
    """
    if wind_kmh < 30.5:
        return 0  # Low Pressure Area
    elif wind_kmh < 50.5:
        return 1  # Depression
    elif wind_kmh < 62.5:
        return 2  # Deep Depression
    elif wind_kmh < 88.5:
        return 3  # Cyclonic Storm
    elif wind_kmh < 117.5:
        return 4  # Severe Cyclonic Storm
    elif wind_kmh < 167.5:
        return 5  # Very Severe Cyclonic Storm
    elif wind_kmh < 221.5:
        return 6  # Extremely Severe Cyclonic Storm
    else:
        return 7  # Super Cyclonic Storm


def category_name(idx: int) -> str:
    return CATEGORY_NAMES[idx]



# ---------------------------------------------------------------------------
# Image / sequence hyperparameters
# ---------------------------------------------------------------------------
CROP_SIZE = 128            # pixels; frame is cropped to CROP_SIZE x CROP_SIZE around storm centre
SEQUENCE_LENGTH_IN = 8     # number of past frames fed to the prediction model
SEQUENCE_LENGTH_OUT = 12   # 12 × 6h = +72h full neural-network track forecast
FRAME_INTERVAL_MINUTES = 30  # nominal INSAT frame cadence

# ERA5 physical features fused alongside imagery for classification/prediction
ERA5_FEATURES = [
    "sea_surface_temperature",
    "wind_shear_850_200hpa",
    "relative_humidity_700hpa",
    "mean_sea_level_pressure",
]
