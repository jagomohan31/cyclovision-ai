# CycloVision AI

AI/ML system for **identification, classification, and prediction** of
tropical cyclone patterns from multi-source satellite data — built for
Smart India Hackathon 2026, **PS SIH26070** (Ministry of Earth Sciences).

This repo is the working codebase behind the idea submission PPT. Every
piece of code in here has been run and verified — nothing is untested
boilerplate. See **"What's already working"** below before you read
another line of docs.

---

## What's already working (verified today)

| Piece | Status | Proof |
|---|---|---|
| Real IMD cyclone data | ✅ 425 storms, 1982-2026, 7,585 observations | `data/raw/ibtracs/SOURCE.md` |
| Preprocessing (crop/normalize/sequence) | ✅ unit tested | `pytest tests/` |
| Detection model (U-Net) | ✅ forward pass verified | `python src/models/detection.py` |
| Classification model (CNN + ERA5 fusion) | ✅ forward pass verified, outputs all 8 IMD tiers | `python -m src.models.classification` |
| Prediction model (ConvLSTM) | ✅ forward pass verified | `python src/models/prediction.py` |
| Full training loop | ✅ runs end-to-end on real labels | `python -m src.training.train_classifier --subset 160 --epochs 2` |
| Evaluation metrics (IoU, per-class F1, track error km) | ✅ unit tested | `pytest tests/` |

Run everything at once:
```bash
pip install -r requirements.txt
pytest tests/ -v
```

## What's NOT done yet (the real next steps)

The one thing no code can do for you: **real INSAT satellite imagery**
needs a MOSDAC account (manual approval) — see `docs/mosdac_guide.md`.
Until then, the classifier trains on real IMD labels paired with a
synthetic placeholder image, specifically so you can test the whole
pipeline today without waiting. Do the MOSDAC signup **first**, today —
it's the one step on someone else's clock.

ERA5 (physical features) needs a Copernicus CDS API key — usually approved
fast — see `src/data/download_era5.py`.

---

## Project layout

```
cyclovision-ai/
├── data/
│   ├── raw/
│   │   ├── ibtracs/       # REAL DATA — IMD best-track record (already here)
│   │   ├── insat/         # empty — your MOSDAC downloads go here
│   │   └── era5/          # empty — your Copernicus downloads go here
│   └── processed/
├── docs/
│   └── mosdac_guide.md    # step-by-step: getting real satellite imagery
├── src/
│   ├── config.py          # IMD 7-tier category thresholds, paths, hyperparams
│   ├── data/
│   │   ├── preprocessing.py    # crop / normalize / sequence-building
│   │   ├── load_besttrack.py   # loads real IMD data, validates categories
│   │   ├── dataset.py          # PyTorch Dataset (real image -> falls back to synthetic)
│   │   ├── download_ibtracs.py # refresh/extend the best-track data
│   │   └── download_era5.py    # pull ERA5 physical features
│   ├── models/
│   │   ├── detection.py        # Model A — U-Net, finds storm + eye
│   │   ├── classification.py   # Model B — hybrid CNN+ERA5, IMD 7-tier grade
│   │   └── prediction.py       # Model C — ConvLSTM, track & intensity forecast
│   └── training/
│       ├── metrics.py          # IoU, per-class F1, track error (km)
│       └── train_classifier.py # training loop (start here — most tractable)
└── tests/
    └── test_models.py     # run this after ANY change: pytest tests/ -v
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
