"""
Bulk ERA5 downloader for ALL 425 IMD storms (1982-2026).

Reads storm_id, date range, and bounding box from the IBTrACS best-track
record and downloads one ERA5 .nc file per storm into data/raw/era5/.

Run:
    python -m src.data.download_era5_all

Skips storms already downloaded (safe to re-run / resume after interruption).
"""

from __future__ import annotations
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import ERA5_DIR
from src.data.load_besttrack import load_observations

PADDING_DEG = 3.0   # degrees of bounding-box padding around storm track


def download_all(skip_existing: bool = True):
    import cdsapi

    df = load_observations()

    storms = df.groupby("storm_id").agg(
        start=("time", "min"),
        end=("time", "max"),
        lat_min=("lat", "min"),
        lat_max=("lat", "max"),
        lon_min=("lon", "min"),
        lon_max=("lon", "max"),
    ).reset_index()

    ERA5_DIR.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()

    total = len(storms)
    skipped = 0
    downloaded = 0
    failed = []

    for i, row in storms.iterrows():
        sid = row["storm_id"]
        out_path = ERA5_DIR / f"{sid}_era5.nc"

        if skip_existing and out_path.exists():
            print(f"[{i+1}/{total}] SKIP  {sid}  (already exists)")
            skipped += 1
            continue

        # Bounding box with padding, clamped to valid ranges
        north = min(round(row["lat_max"] + PADDING_DEG, 1),  90.0)
        south = max(round(row["lat_min"] - PADDING_DEG, 1), -90.0)
        west  = max(round(row["lon_min"] - PADDING_DEG, 1), -180.0)
        east  = min(round(row["lon_max"] + PADDING_DEG, 1),  180.0)

        start_date = row["start"].strftime("%Y-%m-%d")
        end_date   = row["end"].strftime("%Y-%m-%d")

        print(f"[{i+1}/{total}] {sid}  {start_date} -> {end_date}  bbox=[N{north} W{west} S{south} E{east}]")

        try:
            client.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "format": "netcdf",
                    "variable": [
                        "sea_surface_temperature",
                        "mean_sea_level_pressure",
                        "10m_u_component_of_wind",
                        "10m_v_component_of_wind",
                    ],
                    "date": f"{start_date}/{end_date}",
                    "time": [f"{h:02d}:00" for h in range(0, 24, 6)],
                    "area": [north, west, south, east],
                },
                str(out_path),
            )
            size_kb = out_path.stat().st_size // 1024
            print(f"         -> Saved {size_kb} KB to {out_path.name}")
            downloaded += 1

        except Exception as e:
            print(f"         -> FAILED: {e}")
            failed.append(sid)
            time.sleep(5)   # brief pause before next request on error

    print()
    print("=" * 60)
    print(f"Done.  Downloaded={downloaded}  Skipped={skipped}  Failed={len(failed)}")
    if failed:
        print("Failed storm IDs:", failed)


if __name__ == "__main__":
    download_all()
