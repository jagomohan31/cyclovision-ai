"""
PyTorch Dataset for the classification model.

Design goal: let the team run the *entire* training loop today — before
MOSDAC access/downloads are sorted out — by falling back to a synthetic
image when a real INSAT file isn't found at the expected path yet. This
catches shape/plumbing bugs immediately instead of a week from now.

Expected real-image path once you have INSAT downloads (see
docs/mosdac_guide.md):
    data/raw/insat/<storm_id>/<YYYYMMDDHHMM>.npy
    (a single-channel float32 array of IR brightness temperature, any
    size >= config.CROP_SIZE — preprocessing.crop_to_storm_center handles
    cropping it down.)

Swap `use_synthetic_fallback=False` once every observation you train on
has a real file — that flag existing at all is what should disappear
first as your data collection progresses; it's a development aid, not
something to ship.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import INSAT_DIR, ERA5_DIR, CROP_SIZE, ERA5_FEATURES, NUM_CATEGORIES
from src.data.preprocessing import crop_to_storm_center, normalize_brightness_temperature
from src.data.load_besttrack import load_observations

_ERA5_CACHE: dict[tuple[str, str], np.ndarray] = {}


def _load_era5_features(storm_id: str, obs_time) -> np.ndarray:
    """
    Load ERA5 features for a given storm observation from the downloaded .nc file.
    Returns a float32 array of shape (4,): [SST_mean, MSLP_mean, U10_mean, V10_mean].
    Falls back to zeros if the ERA5 file hasn't been downloaded yet for this storm.
    Caches results in-memory to make multi-epoch training fast.
    """
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
        from datetime import timezone

        ds = nc.Dataset(str(nc_path))
        times = ds.variables["valid_time"][:]  # seconds since 1970-01-01

        # Find the closest time step to the observation time
        obs_ts = obs_time.timestamp() if hasattr(obs_time, "timestamp") else float(obs_time)
        idx = int(np.argmin(np.abs(np.array(times, dtype=float) - obs_ts)))

        features = []
        for var in ["sst", "msl", "u10", "v10"]:
            arr = ds.variables[var][idx]
            # compressed() returns only the non-masked values as a plain ndarray
            if hasattr(arr, "compressed"):
                valid = arr.compressed()
            else:
                valid = np.asarray(arr).flatten()
            val = float(np.nanmean(valid)) if len(valid) > 0 else 0.0
            features.append(0.0 if np.isnan(val) else val)

        ds.close()

        # Normalise to roughly zero-mean unit-variance using known physical ranges
        # SST: ~270-310 K  → subtract 290, divide by 15
        # MSLP: ~95000-102000 Pa → subtract 101325, divide by 1500
        # U10, V10: ~-20 to 20 m/s → divide by 10
        features[0] = (features[0] - 290.0) / 15.0
        features[1] = (features[1] - 101325.0) / 1500.0
        features[2] = features[2] / 10.0
        features[3] = features[3] / 10.0

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
    """
    Generate a placeholder brightness-temperature-like frame whose
    'coldness' (cloud-top height proxy) scales with category.
    Optimized with precomputed base grids for high-throughput training.
    """
    base = _get_synthetic_base(category_idx, size)
    noise = rng.normal(0, 3, size=(size, size)).astype(np.float32)
    return base + noise


class CycloneClassificationDataset(Dataset):
    def __init__(
        self,
        observations=None,
        crop_size: int = CROP_SIZE,
        use_synthetic_fallback: bool = True,
        seed: int = 42,
    ):
        obs = observations if observations is not None else load_observations()
        # Keep only fixes that have both a resolved category and a wind reading
        self.obs = obs.dropna(subset=["category_from_grade", "wind_kmh"]).reset_index(drop=True)
        self.crop_size = crop_size
        self.use_synthetic_fallback = use_synthetic_fallback
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.obs)

    def _load_frame(self, row) -> tuple[np.ndarray, bool]:
        real_path = INSAT_DIR / str(row["storm_id"]) / f"{row['time']:%Y%m%d%H%M}.npy"
        if real_path.exists():
            return np.load(real_path).astype(np.float32), True
        if not self.use_synthetic_fallback:
            raise FileNotFoundError(
                f"No INSAT file at {real_path} and use_synthetic_fallback=False. "
                "See docs/mosdac_guide.md to download real imagery."
            )
        return _synthetic_frame(int(row["category_from_grade"]), self.crop_size * 2, self._rng), False

    def __getitem__(self, idx: int):
        row = self.obs.iloc[idx]
        frame, is_real = self._load_frame(row)

        # crop_to_storm_center expects the storm already roughly centred for
        # synthetic frames (it is, by construction); for real INSAT tiles you'll
        # pass the true pixel row/col from preprocessing.latlon_to_pixel(...)
        # using that frame's actual lat/lon grid — see docs/mosdac_guide.md.
        h, w = frame.shape
        cropped = crop_to_storm_center(frame, center_row=h // 2, center_col=w // 2, crop_size=self.crop_size)
        normalized = normalize_brightness_temperature(cropped)

        image = torch.from_numpy(normalized).unsqueeze(0).float()  # (1, H, W)

        # Load real ERA5 features if downloaded, else fall back to zeros.
        era5_np = _load_era5_features(row["storm_id"], row["time"])
        era5 = torch.from_numpy(era5_np).float()

        label = int(row["category_from_grade"])
        return {
            "image": image,
            "era5": era5,
            "label": label,
            "is_real_image": is_real,
            "storm_id": row["storm_id"],
            "time": str(row["time"]),
        }


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    ds = CycloneClassificationDataset()
    print(f"Dataset size: {len(ds)} observations")

    sample = ds[0]
    print("Sample image shape:", sample["image"].shape, "| label:", sample["label"],
          "| is_real_image:", sample["is_real_image"])

    loader = DataLoader(ds, batch_size=8, shuffle=True)
    batch = next(iter(loader))
    print("Batch image shape:", batch["image"].shape, "| batch labels:", batch["label"].tolist())
    n_real = sum(ds[i]["is_real_image"] for i in range(min(200, len(ds))))
    print(f"Real INSAT files found (first 200 checked): {n_real}/200 "
          f"— {'0 is expected until you download real INSAT data' if n_real == 0 else ''}")
