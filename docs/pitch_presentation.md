# CycloVision AI — Pitch & Presentation Guide
**Smart India Hackathon 2026 | Problem Statement PS SIH26070 (Ministry of Earth Sciences)**

---

## 🎤 Updated Speaker Script (Reflecting Current Verified State)

### Speaker 1: The Problem & Regional Reality (1.5 mins)
> *"Every cyclone in the North Indian Ocean threatens millions of lives along the Indian coastline. But Indian forecasters at the IMD and disaster response teams at the NDMA do not use the American Saffir-Simpson Hurricane scale. They use the **IMD 7-Tier Cyclone Intensity Scale** — ranging from Depression to Super Cyclonic Storm. Furthermore, cyclones in the Bay of Bengal and Arabian Sea often undergo rapid intensification over warm sea waters before satellite cloud patterns fully organize. A pure computer vision model looking only at cloud tops misses the underlying ocean thermodynamics."*

### Speaker 2: Multi-Source Architecture & Data Engineering (2 mins)
> *"This is where **CycloVision AI** innovates. Instead of relying purely on imagery, our system integrates a **hybrid multimodal fusion pipeline**:*
> 1. *Spatial imagery from INSAT-3D/3DR geostationary satellites (IR brightness temperature).*
> 2. *Physical environmental reanalysis from ECMWF Copernicus ERA5 — specifically Sea Surface Temperature (SST), Mean Sea Level Pressure (MSLP), and 10-meter U and V atmospheric wind vectors.*
> 
> *We have already retrieved and verified **100% of all 425 historical cyclones in the IMD best-track record (1982–2026)** with full 6-hourly ERA5 physical reanalysis data stored locally.*
> 
> *Crucially, we engineered our pipeline to prevent a common machine learning pitfall: **data leakage**. Consecutive 6-hour satellite fixes of the same cyclone are highly correlated. A naive random split lets the model 'peek' at neighboring frames. We implemented a strict **Storm-ID level split**, ensuring validation storms are completely held-out so our evaluation metrics reflect true real-world generalization."*

### Speaker 3: The 3 Deep Learning Models & Live Demo (2.5 mins)
> *"Our architecture features three specialized models:*
> - * **Model A (U-Net):** Performs cyclone segmentation and locates the eye center.*
> - * **Model B (CNN + ERA5 Fusion):** Classifies the disturbance across IMD's 7 cyclone tiers plus a pre-cyclonic Low Pressure Area baseline.*
> - * **Model C (ConvLSTM):** Predicts spatial cloud vortex evolution and forecasts future trajectory coordinates.*
> 
> *Today, our system is fully interactive. Through our live **Streamlit dashboard**, forecasters can scrub through any historical cyclone — such as Super Cyclone Amphan or Cyclone Biparjoy — monitor real-time ocean thermodynamic triggers (like SST > 29°C), observe the neural network's category confidence distribution, and automatically generate NDMA-compliant Early Warning Red/Orange/Yellow disaster advisories."*

---

## ⚖️ The "7-Tier vs. 8-Tier" Question: Exact Answer for Judges

**Judge:** *"Is your model an 8-tier or a 7-tier classifier? Why do you have 8 classes?"*

**Answer:**
> *"The India Meteorological Department (IMD) officially defines **7 intensity tiers for cyclonic disturbances**:
> 1. Depression (31–50 km/h)
> 2. Deep Depression (51–62 km/h)
> 3. Cyclonic Storm (63–88 km/h)
> 4. Severe Cyclonic Storm (89–117 km/h)
> 5. Very Severe Cyclonic Storm (118–166 km/h)
> 6. Extremely Severe Cyclonic Storm (167–221 km/h)
> 7. Super Cyclonic Storm (≥222 km/h)
>
> In our machine learning architecture, we added **Tier 0: Low Pressure Area (<31 km/h)** as an explicit baseline class. This allows the model to monitor early-stage tropical low-pressure systems before they intensify into a cyclonic depression. Therefore, while our classification covers the **IMD 7-tier cyclone scale**, our neural network outputs across **8 discrete classification bins**."*

---

## 🛡️ Technical Defense & "Gotcha" Answers

### Q1: *"How do you handle coastal boundary artifacts in ERA5 data?"*
> *"ERA5 uses land/ocean masks with fill values that can introduce NaNs along coastlines. If passed into a loss function, these NaNs silently corrupt backpropagation. In `src/data/dataset.py`, we extract valid atmospheric features using `.compressed()` array extractions with `np.nanmean()` fallbacks and z-score normalization based on physical atmospheric reference bounds, thoroughly verified in our unit test suite."*

### Q2: *"How do you handle severe class imbalance?"*
> *"In the 44-year IMD best-track record, there are 2,776 Depressions but only 40 Super Cyclonic Storms (a 69:1 imbalance). Training with unweighted cross-entropy would cause the model to collapse and never predict the most catastrophic cyclones. We address this using **inverse-frequency class weighting** in the loss function, and we evaluate performance using **per-class F1-scores** rather than raw overall accuracy."*

### Q3: *"What is the real-time inference latency?"*
> *"Once trained, inference takes under **50 milliseconds per frame** on a standard CPU. In operational deployment, the pipeline triggers every 30 minutes to ingest incoming INSAT frames and Copernicus updates, delivering real-time predictions well within forecasters' decision windows."*
