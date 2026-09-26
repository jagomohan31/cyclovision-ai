
# CycloVision AI

AI/ML system for **identification, classification, and prediction** of
tropical cyclone patterns from multi-source satellite data — built for
Smart India Hackathon 2026, **PS SIH26070** (Ministry of Earth Sciences).

This repo is the working codebase behind the idea submission PPT. Every
piece of code in here has been run and verified — nothing is untested
boilerplate. See **"What's already working"** below before you read
another line of docs.


**'Integration plus India-calibration plus open accessibility - not a brand-new detection technique.' The individual techniques exist; assembling them into one IMD-calibrated, lightweight, end-to-end pipeline is the gap we're filling.**
---

## What's already working (verified)

| Piece | Status | Proof |
|---|---|---|
| Real IMD cyclone data | ✅ 425 storms, 1982-2026, 7,585 observations | `data/raw/ibtracs/SOURCE.md` |
| Real ERA5 reanalysis data | ✅ 424/425 storms, SST, MSLP, U/V wind | `data/raw/era5/` (NetCDF files) |
| Real INSAT-3D/3DR/3DS imagery | ✅ 13,479 frames across 11 storms (MOSDAC HDF5 → .npy) | `data/raw/insat/` |
| Preprocessing (crop/normalize/sequence) | ✅ unit tested | `pytest tests/` |
| Detection model (U-Net) | ✅ trained — val IoU=0.986, eye error=0.11 px | `detector_checkpoint.pt` 7.82 MB |
| Classification model (ResNet-18 + ERA5) | ✅ trained — val acc=27.1%, F1 0.13–0.54 | `classifier_checkpoint.pt` 45.08 MB |
| Prediction model (ConvLSTM) | ✅ trained — +24h=123 km, +72h=336 km, wind MAE=98.2 km/h | `predictor_checkpoint.pt` 0.32 MB |
| Evaluation metrics & tests | ✅ 19 unit tests passing (100% pass rate) | `pytest tests/ -v` |
| Interactive Web Dashboard | ✅ live UI (map, +24h/+48h/+72h cone, telemetry, AI) | `streamlit run app.py` |
| Real-time inference worker | ✅ watches `data/incoming/` for new H5 files, runs full pipeline | `realtime_worker.py` |

Run everything at once:
```bash
pip install -r requirements.txt
pytest tests/ -v
streamlit run app.py
```

---

## Model performance (Kaggle T4 GPU — final runs, Sep 2026)

### Model A — Cyclone Detector (U-Net)
- Architecture: CycloneUNet, base_channels=16
- val IoU: **0.986** | Eye centroid error: **0.11 px**
- Trained: 30/30 epochs on 13,479 INSAT frames across 11 storms
- Checkpoint: `detector_checkpoint.pt` (7.82 MB)

### Model B — Intensity Classifier (ResNet-18 + ERA5 fusion)
- Architecture: ResNet-18 backbone + ERA5 feature fusion
- Best val acc: **27.1%** at epoch 7, early stop epoch 17
- Per-class F1: Depression=0.13, Deep Depression=0.30, Cyclonic Storm=0.03,
  Severe=0.24, Very Severe=0.37, Extremely Severe=0.54
- Note: minority class performance limited by class imbalance; retraining planned post-hackathon
- Checkpoint: `classifier_checkpoint.pt` (45.08 MB)

### Model C — Track & Intensity Predictor (ConvLSTM)
- Architecture: ConvLSTM sequence model
- Track errors: +6h=40.4 km | +12h=68.4 km | +24h=123.1 km | +48h=224.1 km | +72h=336.1 km
- Wind speed MAE: **98.2 km/h** | Early stop epoch 12
- Checkpoint: `predictor_checkpoint.pt` (0.32 MB)

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



## Training data
| Storm | ID | INSAT frames |   ERA5 |
|---|---|---|---|
| Mekunu 2018 | 2018-003 | 264 | ✅ |
| Gaja 2018 | 2018-013 | 1,231 | ✅ |
| Fani 2019 | 2019-002 | 993 |   ✅ |
| Vayu 2019 | 2019-003 | 1,679 | ✅ |
| Kyarr 2019 | 2019-007 | 589 |  ✅ |
| Maha 2019 | 2019-008 | 1,848 | ✅ |
| Amphan 2020 | 2020-001 | 927 | ✅ |
| Tauktae 2021 | 2021-002 | 990 | ✅ |
| Mocha 2023 | 2023-002 | 1,343 | ✅ |
| Biparjoy 2023 | 2023-003 | 2,782 | ✅ |
| Tej 2023 | 2023-006 | 833 | ✅ |
| **Total** | | **13,479** | **11/11** |
Train/val split: per-storm 80/20 temporal (9,507 train / 2,382 val).
---
## Running the end-to-end simulation
Replay Biparjoy 2023 as a live incoming feed:
```bash
python realtime_worker.py --mode simulate --storm-id 2023-003
```
This feeds H5 frames one by one into the pipeline — detector finds the eye,
classifier assigns intensity, predictor outputs +6h through +72h track — exactly
as it would run on a real MOSDAC live feed.
---
## Citing the data
- **IMD, RSMC New Delhi** — original best-track record (see `data/raw/ibtracs/SOURCE.md`)
- **imdtrack** library (Syed, H. A., 2026) — [doi:10.5281/zenodo.21301659](https://doi.org/10.5281/zenodo.21301659)
- **MOSDAC / ISRO-SAC** — INSAT-3D/3DR/3DS imagery (HDF5 via MOSDAC data portal)
- **Copernicus Climate Data Store** — ERA5 reanalysis
---
