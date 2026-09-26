"""
PyTorch Dataset for the classification model.

TIMESTAMP FIX (Sep 2026):
    IMD best-track fixes land on 3-hourly marks (0000, 0300, 0600...).
    MOSDAC INSAT-3DR frames land at :15/:45 offsets (0015, 0045, 0115...).
    They never exactly coincide. The fix: scan real .npy files first,
    then linearly interpolate the intensity label between the two surrounding
    best-track fixes. This turns 97 label-points into up to 2,818 real
    training examples instead of silently falling back to synthetic.

Design goal: fall back to synthetic images when no real INSAT files exist,
so the training loop works immediately even before data is downloaded.
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import collections
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import INSAT_DIR, ERA5_DIR, CROP_SIZE, ERA5_FEATURES, NUM_CATEGORIES
from src.config import wind_speed_to_category
from src.data.preprocessing import crop_to_storm_center, normalize_brightness_temperature
from src.data.load_besttrack import load_observations

_ERA5_CACHE: dict[tuple[str, str], np.ndarray] = {}


def _load_era5_features(storm_id: str, obs_time) -> np.ndarray:
    """Load ERA5 features; falls back to zeros if file not downloaded."""
    cache_key = (str(storm_id), str(obs_time))
    if cache_key in _ERA5_CACHE:
        return _ERA5_CACHE[cache_key]

    nc_path = ERA5_DIR / f"{storm_id}_era5.nc"
    if not nc_path.exists():
        fallback = np.zeros(len(ERA5_FEATURES), dtype=np.float32)
        _ERA5_CACHE[cache_key] = fallback
        return fallback

    try:
        import netCDF4 as nc
        ds = nc.Dataset(str(nc_path))
        times = ds.variables["valid_time"][:]
        obs_ts = obs_time.timestamp() if hasattr(obs_time, "timestamp") else float(obs_time)
        idx = int(np.argmin(np.abs(np.array(times, dtype=float) - obs_ts)))

        features = []
        for var in ["sst", "msl", "u10", "v10"]:
            arr = ds.variables[var][idx]
            valid = arr.compressed() if hasattr(arr, "compressed") else np.asarray(arr).flatten()
            val = float(np.nanmean(valid)) if len(valid) > 0 else 0.0
            features.append(0.0 if np.isnan(val) else val)
        ds.close()

        # Normalise to ~zero-mean unit-variance
        features[0] = (features[0] - 290.0) / 15.0    # SST K
        features[1] = (features[1] - 101325.0) / 1500.0  # MSLP Pa
        features[2] = features[2] / 10.0               # U10 m/s
        features[3] = features[3] / 10.0               # V10 m/s

        out = np.array(features, dtype=np.float32)
        _ERA5_CACHE[cache_key] = out
        return out
    except Exception:
        fallback = np.zeros(len(ERA5_FEATURES), dtype=np.float32)
        _ERA5_CACHE[cache_key] = fallback
        return fallback


_SYNTH_BASE_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _get_synthetic_base(category_idx: int, size: int) -> np.ndarray:
    key = (category_idx, size)
    if key not in _SYNTH_BASE_CACHE:
        yy, xx = np.mgrid[0:size, 0:size]
        cy, cx = size / 2.0, size / 2.0
        r2 = (yy - cy) ** 2 + (xx - cx) ** 2
        depth = 60 + category_idx * 15
        _SYNTH_BASE_CACHE[key] = (300.0 - depth * np.exp(-r2 / (2.0 * (size / 5.0) ** 2))).astype(np.float32)
    return _SYNTH_BASE_CACHE[key]


def _synthetic_frame(category_idx: int, size: int, rng: np.random.Generator) -> np.ndarray:
    base = _get_synthetic_base(category_idx, size)
    noise = rng.normal(0, 3, size=(size, size)).astype(np.float32)
    return base + noise


def _build_real_frame_index(storm_id: str) -> list[dict]:
    """
    Scan all .npy files for a storm; interpolate wind speed labels from
    surrounding best-track fixes so every real satellite frame becomes a
    training example (up to 2,818 for Biparjoy instead of just 97).

    Frames outside the best-track time window are dropped silently.
    """
    obs_all = load_observations()
    storm_obs = (
        obs_all[(obs_all["storm_id"] == storm_id) &
                obs_all["wind_kmh"].notna() &
                obs_all["category_from_grade"].notna()]
        .copy().sort_values("time").reset_index(drop=True)
    )
    if storm_obs.empty:
        return []

    insat_dir = INSAT_DIR / storm_id
    if not insat_dir.exists():
        return []

    npy_files = sorted(insat_dir.glob("*.npy"))
    if not npy_files:
        return []

    # Convert best-track times to float seconds for interpolation
    bt_ts = np.array([pd.Timestamp(t).timestamp() for t in storm_obs["time"].values], dtype=float)
    bt_winds = storm_obs["wind_kmh"].values.astype(float)

    records = []
    for npy_path in npy_files:
        try:
            frame_dt = datetime.strptime(npy_path.stem, "%Y%m%d%H%M")
        except ValueError:
            continue

        frame_ts = frame_dt.replace(tzinfo=timezone.utc).timestamp()

        # Drop frames outside best-track window
        if frame_ts < bt_ts[0] or frame_ts > bt_ts[-1]:
            continue

        # Linear interpolation between surrounding fixes
        idx_after = int(np.searchsorted(bt_ts, frame_ts))
        if idx_after == 0:
            interp_wind = bt_winds[0]
        elif idx_after >= len(bt_ts):
            interp_wind = bt_winds[-1]
        else:
            t0, t1 = bt_ts[idx_after - 1], bt_ts[idx_after]
            w0, w1 = bt_winds[idx_after - 1], bt_winds[idx_after]
            alpha = (frame_ts - t0) / (t1 - t0) if (t1 - t0) > 0 else 0.0
            interp_wind = w0 + alpha * (w1 - w0)

        if np.isnan(interp_wind):
            continue

        records.append({
            "npy_path": npy_path,
            "time": frame_dt,
            "wind_kmh": float(interp_wind),
            "category": int(wind_speed_to_category(interp_wind)),
            "storm_id": storm_id,
        })

    return records


class CycloneClassificationDataset(Dataset):
    def __init__(
        self,
        observations=None,
        crop_size: int = CROP_SIZE,
        use_synthetic_fallback: bool = True,
        seed: int = 42,
        storm_ids: list[str] | None = None,
    ):
        """
        Parameters
        ----------
        observations : DataFrame | None
            Best-track observations. If None, loaded automatically.
            Used for label interpolation (real mode) or as direct labels (synthetic mode).
        crop_size : int
            Spatial crop size for the IR image patch.
        use_synthetic_fallback : bool
            If True, use synthetic images when no real .npy files found.
            Set False to raise FileNotFoundError when real data is missing.
        seed : int
            RNG seed for synthetic noise reproducibility.
        storm_ids : list[str] | None
            Restrict to specific storm IDs (for train/val splits).
        """
        self.crop_size = crop_size
        self.use_synthetic_fallback = use_synthetic_fallback
        self._rng = np.random.default_rng(seed)

        obs_all = observations if observations is not None else load_observations()
        if storm_ids is not None:
            obs_all = obs_all[obs_all["storm_id"].isin(storm_ids)]
        all_storms = obs_all["storm_id"].unique().tolist()

        # Attempt real-data mode: scan .npy files and interpolate labels
        self._real_records: list[dict] = []
        for sid in all_storms:
            self._real_records.extend(_build_real_frame_index(str(sid)))

        if self._real_records:
            self._mode = "real"
        else:
            # Synthetic fallback: one example per best-track fix
            self._mode = "synthetic"
            self.obs = obs_all.dropna(
                subset=["category_from_grade", "wind_kmh"]
            ).reset_index(drop=True)
            if not self.use_synthetic_fallback and len(self.obs) > 0:
                raise FileNotFoundError(
                    "No real INSAT .npy files found and use_synthetic_fallback=False.\n"
                    f"Expected: {INSAT_DIR}/<storm_id>/<YYYYMMDDHHMM>.npy"
                )

    def __len__(self) -> int:
        return len(self._real_records) if self._mode == "real" else len(self.obs)

    def __getitem__(self, idx: int):
        return self._getitem_real(idx) if self._mode == "real" else self._getitem_synthetic(idx)

    def _getitem_real(self, idx: int, augment: bool = False):
        rec = self._real_records[idx]
        arr = np.load(str(rec["npy_path"])).astype(np.float32)
        arr = np.nan_to_num(arr, nan=270.0)  # fill masked ocean pixels

        h, w = arr.shape
        cropped = crop_to_storm_center(arr, center_row=h // 2, center_col=w // 2, crop_size=self.crop_size)
        normalized = normalize_brightness_temperature(cropped)

        # Augmentation: random flips + rotation + brightness jitter
        # (satellite imagery has no preferred orientation; cyclones are symmetric)
        if augment:
            if self._rng.random() > 0.5:
                normalized = np.fliplr(normalized).copy()
            if self._rng.random() > 0.5:
                normalized = np.flipud(normalized).copy()
            # Random 90-degree rotation (cyclone vortex is rotationally symmetric)
            k = self._rng.integers(0, 4)
            if k > 0:
                normalized = np.rot90(normalized, k=k).copy()
            # Small brightness jitter (sensor calibration variation)
            normalized = (normalized + self._rng.normal(0, 0.02)).astype(np.float32)

        image = torch.from_numpy(normalized).unsqueeze(0).float()

        era5_np = _load_era5_features(rec["storm_id"], rec["time"])
        era5 = torch.from_numpy(era5_np).float()

        return {
            "image": image,
            "era5": era5,
            "label": int(rec["category"]),
            "is_real_image": True,
            "storm_id": rec["storm_id"],
            "time": str(rec["time"]),
        }

    def _getitem_synthetic(self, idx: int):
        row = self.obs.iloc[idx]
        frame = _synthetic_frame(int(row["category_from_grade"]), self.crop_size * 2, self._rng)

        h, w = frame.shape
        cropped = crop_to_storm_center(frame, center_row=h // 2, center_col=w // 2, crop_size=self.crop_size)
        normalized = normalize_brightness_temperature(cropped)
        image = torch.from_numpy(normalized).unsqueeze(0).float()

        era5_np = _load_era5_features(row["storm_id"], row["time"])
        era5 = torch.from_numpy(era5_np).float()

        return {
            "image": image,
            "era5": era5,
            "label": int(row["category_from_grade"]),
            "is_real_image": False,
            "storm_id": row["storm_id"],
            "time": str(row["time"]),
        }


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    ds = CycloneClassificationDataset()
    print(f"Dataset mode  : {ds._mode}")
    print(f"Dataset size  : {len(ds)} examples")

    sample = ds[0]
    print("Sample image shape :", sample["image"].shape)
    print("Label              :", sample["label"])
    print("Is real image      :", sample["is_real_image"])

    loader = DataLoader(ds, batch_size=8, shuffle=True)
    batch = next(iter(loader))
    print("Batch image shape  :", batch["image"].shape)
    print("Batch labels       :", batch["label"].tolist())

    if ds._mode == "real":
        cats = [r["category"] for r in ds._real_records]
        print("\nCategory distribution across real frames:")
        for cat, count in sorted(collections.Counter(cats).items()):
            print(f"  Category {cat}: {count:>5} frames")
    else:
        n_real = sum(ds[i]["is_real_image"] for i in range(min(200, len(ds))))
        print(f"\nReal INSAT files found (first 200 checked): {n_real}/200")