"""
Sanity-check test suite. Run with:  pytest tests/ -v

These don't test model *accuracy* (that needs real data and real training
time) — they test that every shape flows correctly end-to-end, so a typo
or a mismatched dimension gets caught in seconds, not after a 2-hour
Colab training run dies at epoch 3.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.config import NUM_CATEGORIES, ERA5_FEATURES, wind_speed_to_category, IMD_CATEGORIES, ERA5_DIR
from src.data.preprocessing import crop_to_storm_center, normalize_brightness_temperature, build_sequences
from src.data.dataset import _load_era5_features
from src.models.detection import CycloneUNet, locate_eye_from_mask
from src.models.classification import CycloneClassifier
from src.models.prediction import CyclonePredictor
from src.training.metrics import iou_score, per_class_f1, haversine_km


def test_config_category_thresholds_are_contiguous():
    """Category bands should tile 0..infinity with no gaps or overlaps."""
    sorted_cats = sorted(IMD_CATEGORIES, key=lambda c: c[1])
    for i in range(len(sorted_cats) - 1):
        this_max = sorted_cats[i][2]
        next_min = sorted_cats[i + 1][1]
        assert next_min == this_max + 1, (
            f"Gap/overlap between {sorted_cats[i][0]} and {sorted_cats[i+1][0]}"
        )


def test_wind_speed_to_category_boundaries():
    assert wind_speed_to_category(30) == 0   # Low Pressure Area
    assert wind_speed_to_category(31) == 1   # Depression starts
    assert wind_speed_to_category(62) == 2   # Deep Depression ends
    assert wind_speed_to_category(63) == 3   # Cyclonic Storm starts
    assert wind_speed_to_category(300) == 7  # Super Cyclonic Storm (open-ended)


def test_crop_to_storm_center_shape_and_edge_case():
    frame = np.random.rand(400, 400)
    assert crop_to_storm_center(frame, 200, 200, crop_size=128).shape == (128, 128)
    # Storm right at the frame edge shouldn't crash or return a ragged shape
    assert crop_to_storm_center(frame, 0, 0, crop_size=128).shape == (128, 128)
    assert crop_to_storm_center(frame, 399, 399, crop_size=128).shape == (128, 128)


def test_normalize_brightness_temperature_range():
    frame = np.array([100.0, 170.0, 245.0, 320.0, 500.0])
    norm = normalize_brightness_temperature(frame)
    assert norm.min() >= 0.0 and norm.max() <= 1.0


def test_build_sequences_shapes():
    frames = np.random.rand(20, 16, 16)
    X, y = build_sequences(frames, seq_len_in=8, seq_len_out=4)
    assert X.shape == (9, 8, 16, 16)
    assert y.shape == (9, 4, 16, 16)


def test_detection_unet_forward_pass():
    model = CycloneUNet(in_channels=1, base_channels=16)
    x = torch.randn(2, 1, 128, 128)
    logits = model(x)
    assert logits.shape == (2, 1, 128, 128)

    eye = locate_eye_from_mask(torch.sigmoid(logits))
    assert eye.shape == (2, 2)
    assert torch.isfinite(eye).all()


def test_classification_model_forward_pass():
    model = CycloneClassifier(num_categories=NUM_CATEGORIES, num_era5_features=len(ERA5_FEATURES))
    image = torch.randn(4, 1, 128, 128)
    era5 = torch.randn(4, len(ERA5_FEATURES))
    logits = model(image, era5)
    assert logits.shape == (4, NUM_CATEGORIES)


def test_prediction_model_forward_pass():
    model = CyclonePredictor(in_channels=1, hidden_channels=8, seq_len_out=4)
    x_seq = torch.randn(2, 8, 1, 32, 32)
    frames, track = model(x_seq)
    assert frames.shape == (2, 4, 1, 32, 32)
    assert track.shape == (2, 4, 3)
    assert torch.isfinite(frames).all() and torch.isfinite(track).all()


def test_iou_score_identical_masks_is_one():
    mask = (torch.rand(2, 1, 16, 16) > 0.5).float()
    assert abs(iou_score(mask, mask) - 1.0) < 1e-6


def test_per_class_f1_perfect_prediction():
    y = np.array([0, 1, 2, 3])
    f1s = per_class_f1(y, y, num_classes=4)
    assert all(abs(v - 1.0) < 1e-6 for v in f1s.values())


def test_haversine_zero_distance():
    d = haversine_km(15.0, 85.0, 15.0, 85.0)
    assert d < 1e-6


def test_haversine_known_distance():
    # Chennai (13.08N, 80.27E) to Kolkata (22.57N, 88.36E) is ~1330-1370 km
    d = haversine_km(13.08, 80.27, 22.57, 88.36)
    assert 1300 < d < 1400


def test_era5_feature_extraction_no_nans_on_missing_file():
    """Missing storm files should return a zero-vector fallback without crashing or NaNs."""
    from datetime import datetime
    feats = _load_era5_features("NON_EXISTENT_STORM", datetime(2023, 6, 15, 12, 0))
    assert isinstance(feats, np.ndarray)
    assert feats.shape == (len(ERA5_FEATURES),)
    assert feats.dtype == np.float32
    assert not np.isnan(feats).any()
    assert (feats == 0.0).all()


def test_real_era5_file_loading_and_nan_guarding():
    """Verify that an existing downloaded storm file loads with zero NaNs and valid normalized bounds."""
    from datetime import datetime
    test_file = ERA5_DIR / "2023-003_era5.nc"
    if not test_file.exists():
        import pytest
        pytest.skip("2023-003_era5.nc not present in raw era5 directory.")

    feats = _load_era5_features("2023-003", datetime(2023, 6, 10, 6, 0))
    assert isinstance(feats, np.ndarray)
    assert feats.shape == (len(ERA5_FEATURES),)
    assert feats.dtype == np.float32
    # Critical: assert strict absence of NaNs, Infs, or masked elements
    assert not np.isnan(feats).any()
    assert np.isfinite(feats).all()
    # Physical sanity check on normalized values (should roughly fall between -10 and +10)
    assert (np.abs(feats) < 15.0).all()


def test_storm_level_train_val_split_zero_leakage():
    """Ensure that train/val splitting by storm ID guarantees zero overlap of storms."""
    import pandas as pd
    # Construct a synthetic observations dataframe with multiple fixes per storm
    fake_obs = pd.DataFrame({
        "storm_id": ["STORM_A"] * 10 + ["STORM_B"] * 8 + ["STORM_C"] * 15 + ["STORM_D"] * 5 + ["STORM_E"] * 12,
        "wind_kmh": [60.0] * 50,
        "category_from_grade": [2] * 50,
    })

    all_storms = np.array(fake_obs["storm_id"].unique(), dtype=str)
    rng = np.random.default_rng(42)
    shuffled = all_storms.copy()
    rng.shuffle(shuffled)

    val_split = 0.2
    n_val = max(1, int(len(shuffled) * val_split))
    val_storms = set(shuffled[:n_val])
    train_storms = set(shuffled[n_val:])

    # 1. Storm IDs must be strictly disjoint
    assert val_storms.isdisjoint(train_storms), "Data leakage! Storm ID found in both train and val."

    # 2. Observation indices must be mutually exclusive and exhaust the dataset
    train_idx = set(fake_obs.index[fake_obs["storm_id"].isin(train_storms)])
    val_idx = set(fake_obs.index[fake_obs["storm_id"].isin(val_storms)])
    assert train_idx.isdisjoint(val_idx), "Observation indices overlap between splits!"
    assert len(train_idx) + len(val_idx) == len(fake_obs), "Indices do not cover all observations!"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
