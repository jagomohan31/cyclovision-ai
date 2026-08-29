"""
Download ERA5 reanalysis features for fusion into the classification model
(sea-surface temperature, wind shear, humidity, MSLP — see
config.ERA5_FEATURES).

Requires a free Copernicus Climate Data Store account + API key:
    1. Register at https://cds.climate.copernicus.eu
    2. Get your API key from your account page
    3. Save it to ~/.cdsapirc as:
           url: https://cds.climate.copernicus.eu/api
           key: <your-key>
    4. pip install cdsapi

This talks to an external API (not reachable from this sandbox), so run
it on your own machine / Colab — this file is the ready-to-go template
for that, not something to run in here.
"""

from __future__ import annotations
from pathlib import Path
from src.config import ERA5_DIR


def download_era5_for_storm(
    storm_id: str,
    start_date: str,   # "YYYY-MM-DD"
    end_date: str,     # "YYYY-MM-DD"
    area: list[float],  # [North, West, South, East] in degrees
    out_dir: Path = ERA5_DIR,
):
    """
    Pull the four fusion features for one storm's lifecycle + a bounding
    box around its track. Keeping requests storm-scoped (rather than
    downloading global ERA5) is what makes this practical on a student
    GPU/laptop — full global ERA5 is enormous.
    """
    import cdsapi  # imported lazily so this file can be read/tested without the package installed

    client = cdsapi.Client()
    out_path = out_dir / f"{storm_id}_era5.nc"

    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": [
                "sea_surface_temperature",
                "mean_sea_level_pressure",
                "relative_humidity",   # requested at 700 hPa via pressure_level param on the pressure-levels dataset if needed
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
            ],
            "date": f"{start_date}/{end_date}",
            "time": [f"{h:02d}:00" for h in range(0, 24, 6)],  # 6-hourly, matching IBTrACS cadence
            "area": area,  # [N, W, S, E]
        },
        str(out_path),
    )
    print(f"Saved ERA5 data for {storm_id} -> {out_path}")
    return out_path


if __name__ == "__main__":
    # Example: Cyclone Biparjoy (2023), Arabian Sea, roughly 6-19 June 2023.
    # Bounding box is generous around its actual track — tighten once you
    # have the real lat/lon range from load_besttrack.load_observations().
    download_era5_for_storm(
        storm_id="2023-003",
        start_date="2023-06-06",
        end_date="2023-06-19",
        area=[25, 60, 10, 75],  # North, West, South, East
    )
