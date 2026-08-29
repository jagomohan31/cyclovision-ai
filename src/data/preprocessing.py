"""
Preprocessing utilities for INSAT satellite frames.

These are pure, well-tested numpy functions with no I/O, so they can be
unit-tested with synthetic arrays before you ever touch real satellite
files. Once you have real INSAT NetCDF files (see src/data/download_ibtracs.py
and docs/mosdac_guide.md for how to get them), src/data/dataset.py wires
these functions into a PyTorch Dataset.
"""

from __future__ import annotations
import numpy as np


def crop_to_storm_center(
    frame: np.ndarray,
    center_row: int,
    center_col: int,
    crop_size: int = 128,
) -> np.ndarray:
    """
    Crop a 2D (or 3D, channels-last) satellite frame to a square window
    centred on the storm's reported lat/lon (already converted to pixel
    row/col by the caller). Pads with the frame's own edge values if the
    storm is near the boundary, so the output is always exactly
    (crop_size, crop_size[, channels]).
    """
    half = crop_size // 2
    h, w = frame.shape[0], frame.shape[1]

    pad = half + 1
    padded = np.pad(
        frame,
        [(pad, pad), (pad, pad)] + ([(0, 0)] if frame.ndim == 3 else []),
        mode="edge",
    )
    r = center_row + pad
    c = center_col + pad
    cropped = padded[r - half : r + half, c - half : c + half]
    return cropped


def normalize_brightness_temperature(
    frame: np.ndarray,
    t_min: float = 170.0,
    t_max: float = 320.0,
) -> np.ndarray:
    """
    Normalize INSAT infrared brightness-temperature values (Kelvin) to [0, 1].

    Defaults (170K-320K) cover the range from the coldest cyclone cloud tops
    to warm ocean/land surface. Values are clipped before scaling so a few
    noisy/out-of-range pixels can't blow up the whole normalization.
    """
    clipped = np.clip(frame, t_min, t_max)
    return (clipped - t_min) / (t_max - t_min)


def fill_missing_frame(frame: np.ndarray, prev_frame: np.ndarray | None) -> np.ndarray:
    """
    Handle a cloud-obscured or missing satellite pass.

    Strategy: if NaNs are present and we have a previous valid frame of the
    same shape, fill gaps from the previous frame (temporal interpolation).
    Otherwise fall back to the frame's own mean so training doesn't crash on
    a fully-missing tile.
    """
    if not np.isnan(frame).any():
        return frame

    filled = frame.copy()
    nan_mask = np.isnan(filled)

    if prev_frame is not None and prev_frame.shape == frame.shape:
        filled[nan_mask] = prev_frame[nan_mask]
        nan_mask = np.isnan(filled)  # in case prev_frame also had NaNs there

    if nan_mask.any():
        fill_value = np.nanmean(frame) if not np.isnan(frame).all() else 0.0
        filled[nan_mask] = fill_value

    return filled


def build_sequences(
    frames: np.ndarray,
    seq_len_in: int,
    seq_len_out: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slide a window over a time-ordered stack of frames to build
    (input_sequence, target_sequence) pairs for the prediction model.

    frames: array of shape (T, H, W) or (T, H, W, C), ordered by time.
    Returns X of shape (N, seq_len_in, H, W[, C]) and
            y of shape (N, seq_len_out, H, W[, C]).
    """
    total_len = seq_len_in + seq_len_out
    t = frames.shape[0]
    n_windows = t - total_len + 1
    if n_windows <= 0:
        raise ValueError(
            f"Need at least {total_len} frames to build one sequence, got {t}."
        )

    xs, ys = [], []
    for start in range(n_windows):
        xs.append(frames[start : start + seq_len_in])
        ys.append(frames[start + seq_len_in : start + total_len])

    return np.stack(xs), np.stack(ys)


def latlon_to_pixel(
    lat: float,
    lon: float,
    grid_lat_min: float,
    grid_lat_max: float,
    grid_lon_min: float,
    grid_lon_max: float,
    grid_height: int,
    grid_width: int,
) -> tuple[int, int]:
    """
    Convert a storm's (lat, lon) — as reported in IBTrACS — into a
    (row, col) pixel index on a satellite frame's fixed lat/lon grid.
    Row 0 is assumed to be the northernmost row (standard image convention).
    """
    lat_frac = (grid_lat_max - lat) / (grid_lat_max - grid_lat_min)
    lon_frac = (lon - grid_lon_min) / (grid_lon_max - grid_lon_min)

    row = int(np.clip(lat_frac * grid_height, 0, grid_height - 1))
    col = int(np.clip(lon_frac * grid_width, 0, grid_width - 1))
    return row, col
