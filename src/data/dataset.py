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

from src.config import INSAT_DIR, CROP_SIZE, ERA5_FEATURES, NUM_CATEGORIES
from src.data.preprocessing import crop_to_storm_center, normalize_brightness_temperature
from src.data.load_besttrack import load_observations


def _synthetic_frame(category_idx: int, size: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate a placeholder brightness-temperature-like frame whose
    "coldness" (cloud-top height proxy) loosely scales with category, so a
    model training on synthetic data can at least show non-trivial
    learning behaviour during pipeline smoke-tests. NOT real signal —
    replace with actual INSAT crops before trusting any reported accuracy.
    """
    yy, xx = np.mgrid[0:size, 0:size]
    cy, cx = size / 2, size / 2
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    depth = 60 + category_idx * 15  # stronger storms -> colder, deeper signature
    base = 300 - depth * np.exp(-(r ** 2) / (2 * (size / 5) ** 2))
    noise = rng.normal(0, 3, size=(size, size))
    return (base + noise).astype(np.float32)


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

        # ERA5 features aren't wired up yet (needs Copernicus CDS access — see
        # src/data/download_era5.py); zeros keep the fusion model's forward
        # pass correct so training runs end-to-end today. Swap in real ERA5
        # values as soon as you have them, same way as the imagery above.
        era5 = torch.zeros(len(ERA5_FEATURES), dtype=torch.float32)

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
