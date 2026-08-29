# Data source: IMD RSMC New Delhi official best-track record

**Files here are REAL data** (425 storms, 1982-2026, 7,585 six/three-hourly
observations) — not synthetic placeholders. This is exactly the kind of
ground truth your classification and prediction models train against.

- `imd_besttrack_observations.csv` — one row per storm fix: time, lat, lon,
  wind (**knots**), pressure, and IMD `grade` code.
- `imd_besttrack_storms.csv` — one row per storm: name, basin, start/end
  time, peak wind, peak grade.
- `imd_besttrack_manifest.json` — provenance metadata (source file, hash,
  generation date).

## Provenance

Original data: **India Meteorological Department, RSMC New Delhi** — the
official cyclone warning agency for the North Indian Ocean — published as a
best-track Excel workbook at rsmcnewdelhi.imd.gov.in (see `source_url` in
the manifest for the exact file).

Accessed via the open-source **imdtrack** Python library, which parses
IMD's workbook into tidy tables and is kept up to date automatically:

> Syed, H. A. (2026). *imdtrack: A Python library for the IMD North Indian
> Ocean cyclone best-track record.* https://doi.org/10.5281/zenodo.21301659
> Code: https://github.com/syedhamidali/imdtrack (BSD-3-Clause)

If you use this data in your SIH submission/report, cite both IMD (as the
originating agency) and the imdtrack library (as the access path) — this is
good practice and something judges notice.

## Important: grade codes vs. our 7-tier config

This file uses IMD's short grade codes, which map onto
`src/config.py::IMD_CATEGORIES` like this:

| Code   | Meaning                          |
|--------|-----------------------------------|
| D      | Depression                        |
| DD     | Deep Depression                   |
| CS     | Cyclonic Storm                    |
| SCS    | Severe Cyclonic Storm             |
| VSCS   | Very Severe Cyclonic Storm        |
| ESCS   | Extremely Severe Cyclonic Storm   |
| SuCS   | Super Cyclonic Storm              |

Wind speed in this file is in **knots** (IMD's native unit) — `src/data/load_besttrack.py`
converts to km/h (1 kt = 1.852 km/h) before applying `config.wind_speed_to_category()`,
so everything downstream is consistent in km/h.

## What's still missing (needs MOSDAC / ERA5)

This gives you real storm **tracks and intensities** — enough to build and
sanity-check the prediction and classification *targets* right now. You
still need the actual **satellite imagery** (INSAT) and **ERA5 reanalysis**
features to train the real models — see `docs/mosdac_guide.md` and
`src/data/download_era5.py`.
