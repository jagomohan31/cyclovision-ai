"""
Load the real IMD RSMC New Delhi best-track record (see data/raw/ibtracs/SOURCE.md)
and reconcile it with our IMD_CATEGORIES scheme from src/config.py.

This is your ground-truth source for:
  - training targets for the classification model (IMD grade -> category index)
  - training targets for the prediction model (lat/lon/wind at each future step)
  - picking which storms to prioritize once you start pulling INSAT imagery,
    since you'll want good coverage across all 7 categories, not just the
    common Cyclonic-Storm-and-below cases.

Run directly (`python -m src.data.load_besttrack`) to print a category
distribution sanity check.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd

from src.config import wind_speed_to_category, category_name, IBTRACS_DIR

KNOTS_TO_KMH = 1.852

# IMD's short grade codes, in case you want to cross-check against the
# wind-speed-derived category instead of (or alongside) trusting wind alone.
GRADE_CODE_TO_CATEGORY_INDEX = {
    "D": 1,      # Depression
    "DD": 2,     # Deep Depression
    "CS": 3,     # Cyclonic Storm
    "SCS": 4,    # Severe Cyclonic Storm
    "VSCS": 5,   # Very Severe Cyclonic Storm
    "ESCS": 6,   # Extremely Severe Cyclonic Storm
    "SuCS": 7,   # Super Cyclonic Storm
}


def load_observations(path: Path | None = None) -> pd.DataFrame:
    """Load per-fix observations with wind converted to km/h and both
    wind-derived and grade-derived category indices attached."""
    path = path or (IBTRACS_DIR / "imd_besttrack_observations.csv")
    df = pd.read_csv(path, parse_dates=["time"])

    df["wind_kmh"] = df["wind"] * KNOTS_TO_KMH
    df["category_from_wind"] = df["wind_kmh"].apply(
        lambda w: wind_speed_to_category(w) if pd.notnull(w) else None
    )
    df["category_from_grade"] = df["grade"].map(GRADE_CODE_TO_CATEGORY_INDEX)
    return df


def load_storms(path: Path | None = None) -> pd.DataFrame:
    path = path or (IBTRACS_DIR / "imd_besttrack_storms.csv")
    return pd.read_csv(path, parse_dates=["start_time", "end_time"])


if __name__ == "__main__":
    obs = load_observations()

    # Sanity check: how often does the wind-derived category agree with
    # IMD's own reported grade? (Some disagreement is expected right at
    # category boundaries — IMD sometimes holds a grade steady briefly
    # around the threshold — but it should agree the large majority of the time.)
    both = obs.dropna(subset=["category_from_wind", "category_from_grade"])
    agree = (both["category_from_wind"] == both["category_from_grade"]).mean()
    print(f"Wind-derived vs IMD-grade category agreement: {agree:.1%} "
          f"({len(both)} observations with both fields present)")

    print("\nCategory distribution (by IMD grade, all 7,585 observations):")
    counts = obs["category_from_grade"].value_counts().sort_index()
    for idx, n in counts.items():
        print(f"  {int(idx)}  {category_name(int(idx)):<32} {n:>5} observations")

    print(f"\nTotal storms: {obs['storm_id'].nunique()}  "
          f"|  date range: {obs['time'].min().date()} to {obs['time'].max().date()}")
