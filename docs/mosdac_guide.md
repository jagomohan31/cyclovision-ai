# Getting real INSAT satellite imagery from MOSDAC

This is the one step in the whole pipeline that needs a human to click
through a registration form — I can't do it for you, and I can't browse
the live site to confirm exact button labels (it blocks automated
access), so treat the steps below as the right *sequence*, but double
check the exact wording on-screen since ISRO does update the UI.

## 1. Register

- Go to **https://www.mosdac.gov.in/registration**
- Sign up with your **institutional email** if you have one (`@college.ac.in`
  etc.) — this is generally the fastest path for approval on data centres
  like this; a personal Gmail address usually still works but can take
  longer.
- You'll typically need to state your purpose — write something like:
  *"Academic project for Smart India Hackathon 2026 (PS SIH26070, Ministry
  of Earth Sciences) — AI/ML-based tropical cyclone identification,
  classification, and prediction using INSAT multi-spectral imagery."*
  Being specific like this tends to speed up manual approval queues.
- Approval is not always instant — **do this today**, even before you've
  written any more code, since it's the one dependency on someone else's
  timeline.

## 2. Find cyclone-specific data (don't boil the ocean)

Once approved, log in and look for:
- **INSAT 3D/3DR/3DS → Cyclone / eAtlas** section — MOSDAC has historically
  hosted a cyclone-specific product catalogue organised by named storm,
  which is much easier to work with than hunting through raw full-disk
  imagery by date.
- If that specific section has moved, search/browse for **"INSAT 3D
  Cyclone eAtlas"** from the MOSDAC home/open-data pages — it's worth
  checking **mosdac.gov.in/open-data** too, since some derived products
  are listed there with no extra approval step at all.

## 3. Pick your first storms

Use `python -m src.data.load_besttrack` (already in this repo, running on
**real IMD data** — see `data/raw/ibtracs/SOURCE.md`) to shortlist storms.
Good first picks, across a spread of categories so you're not only
training on common Depressions:

```
python3 -c "
from src.data.load_besttrack import load_storms
s = load_storms()
print(s[s['year'] >= 2019].sort_values('max_wind', ascending=False)[['storm_id','year','name','basin','peak_grade','max_wind']].head(15))
"
```

Download the IR (and, if available, water-vapour) channel for each
chosen storm's full lifecycle, at whatever cadence MOSDAC offers
(typically 30 min for INSAT) from genesis to dissipation.

## 4. Where files go in this project

Save each frame as a `.npy` array of raw brightness temperature (Kelvin),
one file per timestamp, named to match what `src/data/dataset.py` expects:

```
data/raw/insat/<storm_id>/<YYYYMMDDHHMM>.npy
```

Example: `data/raw/insat/2023-002/202305102130.npy` for a MOCHA (2023)
frame. `storm_id` must match the `storm_id` column in
`data/raw/ibtracs/imd_besttrack_observations.csv` so the loader can join
imagery to real IMD labels automatically.

If MOSDAC gives you HDF5/NetCDF instead of a simple array, that's fine —
just add a small conversion step (extract the IR band as a 2D array,
`np.save(...)`) before dropping it in the folder above. `xarray` (already
in requirements.txt) reads INSAT's HDF5 products directly:

```python
import xarray as xr
ds = xr.open_dataset("your_insat_file.h5")
print(ds)  # inspect variable names — the IR band is usually named
           # something like IMG_TIR1 or similar; exact name varies by product
```

## 5. Once you have even ~5-10 storms downloaded

Set `use_synthetic_fallback=False` in `CycloneClassificationDataset` (or
just leave it `True` — real files are always preferred automatically over
the synthetic fallback whenever a matching `.npy` exists) and re-run:

```
python -m src.training.train_classifier --subset 500
```

You're now training on real signal.

## Also worth doing in parallel (doesn't block on MOSDAC approval)

**ERA5 reanalysis** (sea-surface temp, wind shear, humidity — the physical
features the classifier fuses in) comes from Copernicus Climate Data
Store, which typically approves API-key requests near-instantly:
**https://cds.climate.copernicus.eu** → create account → API key → see
`src/data/download_era5.py` for the request template.
