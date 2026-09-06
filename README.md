# CycloVision AI

AI/ML system for **identification, classification, and prediction** of
tropical cyclone patterns from multi-source satellite data — built for
Smart India Hackathon 2026, **PS SIH26070** (Ministry of Earth Sciences).

This repo is the working codebase behind the idea submission PPT. Every
piece of code in here has been run and verified — nothing is untested
boilerplate. See **"What's already working"** below before you read
another line of docs.

---

## What's already working (verified)

| Piece | Status | Proof |
|---|---|---|
| Real IMD cyclone data | ✅ 425 storms, 1982-2026, 7,585 observations | `data/raw/ibtracs/SOURCE.md` |
| Real ERA5 reanalysis data | ✅ 425/425 storms (100%), SST, MSLP, U/V wind | `data/raw/era5/` (NetCDF files) |
| Preprocessing (crop/normalize/sequence) | ✅ unit tested | `pytest tests/` |
| Detection model (U-Net) | ✅ forward pass verified | `python src/models/detection.py` |
| Classification model (CNN + ERA5 fusion) | ✅ IMD 7-tier scale + LPA baseline (8 output bins) | `python -m src.models.classification` |
| Prediction model (ConvLSTM) | ✅ forward pass verified | `python src/models/prediction.py` |
| Full training loop (leak-free) | ✅ storm-ID level train/val split (zero leakage) | `python -m src.training.train_classifier` |
| Evaluation metrics & tests | ✅ 15 unit tests passing (incl. NaN guards & split) | `pytest tests/ -v` |
| Interactive Web Dashboard | ✅ live interactive UI with map, telemetry, AI | `streamlit run app.py` |

Run everything at once:
```bash
pip install -r requirements.txt
pytest tests/ -v
streamlit run app.py
```

## What's NOT done yet (the real next step)

The one thing no code can do for you: **real INSAT satellite imagery**
needs a MOSDAC account (manual approval) — see `docs/mosdac_guide.md`.
Until then, the classifier trains on real IMD labels paired with a
synthetic placeholder image and **100% real ERA5 reanalysis physical features**.
Do the MOSDAC signup **first** — it's the one step on someone else's clock.

---

## Project layout

```
cyclovision-ai/
├── app.py                 # Live interactive Streamlit dashboard (map, telemetry, AI inference)
├── data/
│   ├── raw/
│   │   ├── ibtracs/       # REAL DATA — IMD best-track record (425 storms, 1982-2026)
│   │   ├── insat/         # empty — your MOSDAC downloads go here
│   │   └── era5/          # REAL DATA — 425/425 downloaded Copernicus NetCDF reanalyses
│   └── processed/
├── docs/
│   ├── mosdac_guide.md    # step-by-step: getting real satellite imagery
│   └── pitch_presentation.md # SIH presentation script & judges Q&A guide
├── src/
│   ├── config.py          # IMD 7-tier scale + LPA baseline thresholds, paths, hyperparams
│   ├── data/
│   │   ├── preprocessing.py    # crop / normalize / sequence-building
│   │   ├── load_besttrack.py   # loads real IMD data, validates categories
│   │   ├── dataset.py          # PyTorch Dataset (real ERA5 fusion + synthetic image fallback)
│   │   ├── download_ibtracs.py # refresh/extend the best-track data
│   │   ├── download_era5.py    # single storm ERA5 downloader
│   │   └── download_era5_all.py# bulk 425-storm ERA5 downloader (100% complete)
│   ├── models/
│   │   ├── detection.py        # Model A — U-Net, finds storm + eye
│   │   ├── classification.py   # Model B — hybrid CNN + real ERA5 fusion
│   │   └── prediction.py       # Model C — ConvLSTM, track & intensity forecast
│   └── training/
│       ├── metrics.py          # IoU, per-class F1, track error (km)
│       └── train_classifier.py # leak-free training loop (split by storm ID)
└── tests/
    └── test_models.py     # 15 tests: models, metrics, NaN-guards, storm split
```

## Day-1 checklist for the team

1. **Someone registers for MOSDAC right now** (`docs/mosdac_guide.md`) — approval can take time, so this cannot be the last thing you do.
2. Everyone else: `pip install -r requirements.txt && pytest tests/ -v` — confirm your machine runs everything.
3. Read `src/config.py` — the IMD category thresholds are the single source of truth every model trains against.
4. Skim `python -m src.data.load_besttrack` output — get a feel for the real class imbalance (2,776 Depressions vs. 40 Super Cyclonic Storms) before you write a single line of training code assuming balanced classes.
5. Start with **classification** (`src/training/train_classifier.py`) — it's the most tractable of the three models and gives you a working demo fastest. Detection and prediction follow the same pattern once this one clicks.

## Why a synthetic-image fallback instead of waiting for real data

`src/data/dataset.py` generates a placeholder image when no real INSAT
file exists yet at the expected path, so `train_classifier.py` runs today
and proves the *plumbing* — data loading, class weighting, loss,
backprop, metrics — works, before a single real satellite file is
downloaded. The moment you drop real `.npy` crops into
`data/raw/insat/<storm_id>/`, the exact same code starts training on real
signal — zero changes needed. Treat any accuracy number from
synthetic-image training as meaningless; it exists to catch bugs, not to
report in your final results.

## Citing the data

If your report/demo cites data sources (it should — judges notice):
- **IMD, RSMC New Delhi** — original best-track record (see `data/raw/ibtracs/SOURCE.md` for the exact source file)
- **imdtrack** library (Syed, H. A., 2026) — https://doi.org/10.5281/zenodo.21301659 — used to access the IMD record in tidy form
- **MOSDAC / ISRO-SAC** — INSAT-3D/3DR/3DS imagery (once downloaded)
- **Copernicus Climate Data Store** — ERA5 reanalysis (once downloaded)

## Matching the idea submission PPT

The three-model pipeline here (Detect → Classify → Predict) and the
4-stage architecture diagram in the PPT are the same pipeline — this repo
is that diagram, made runnable.
