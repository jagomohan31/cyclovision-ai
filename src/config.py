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
    ("Very Severe Cyclonic Storm",     118,  166),
    ("Extremely Severe Cyclonic Storm",167,  221),
    ("Super Cyclonic Storm",           222, 10_000),
]

# Convenience lookups
CATEGORY_NAMES = [c[0] for c in IMD_CATEGORIES]
NUM_CATEGORIES = len(IMD_CATEGORIES)


def wind_speed_to_category(wind_kmh: float) -> int:
    """Map a sustained wind speed (km/h) to an IMD category index (0-7)."""
    for idx, (_, lo, hi) in enumerate(IMD_CATEGORIES):
        if lo <= wind_kmh <= hi:
            return idx
    # Anything above the last bracket's max still counts as Super Cyclonic Storm
    return NUM_CATEGORIES - 1


def category_name(idx: int) -> str:
    return CATEGORY_NAMES[idx]


# ---------------------------------------------------------------------------
# Image / sequence hyperparameters
# ---------------------------------------------------------------------------
CROP_SIZE = 128            # pixels; frame is cropped to CROP_SIZE x CROP_SIZE around storm centre
SEQUENCE_LENGTH_IN = 8     # number of past frames fed to the prediction model
SEQUENCE_LENGTH_OUT = 4    # number of future frames/steps predicted
FRAME_INTERVAL_MINUTES = 30  # nominal INSAT frame cadence

# ERA5 physical features fused alongside imagery for classification/prediction
ERA5_FEATURES = [
    "sea_surface_temperature",
    "wind_shear_850_200hpa",
    "relative_humidity_700hpa",
    "mean_sea_level_pressure",
]
