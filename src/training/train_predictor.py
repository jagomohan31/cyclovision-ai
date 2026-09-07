"""
Training loop for Model C (Prediction) — ConvLSTM Spatio-Temporal Forecasting.

Forecasts cyclone track coordinates (lat, lon offsets), maximum sustained wind speed,
and future cloud vortex morphology over a 12-24h forecast horizon given past satellite
observations.

Usage:
    python -m src.training.train_predictor --epochs 5 --batch-size 16
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

# Project imports
from src.config import (
    SEQUENCE_LENGTH_IN,
    SEQUENCE_LENGTH_OUT,
    CROP_SIZE,
    INSAT_DIR,
    wind_speed_to_category,
)
from src.data.load_besttrack import load_observations
from src.data.preprocessing import crop_to_storm_center, normalize_brightness_temperature
from src.data.dataset import _get_synthetic_base
from src.models.prediction import CyclonePredictor
from src.training.metrics import haversine_km


class CycloneSequenceDataset(Dataset):
    """
    Builds sliding-window sequences of past and future cyclone observations
    for ConvLSTM spatio-temporal forecasting.
    """

    def __init__(
        self,
        observations: pd.DataFrame | None = None,
        seq_len_in: int = SEQUENCE_LENGTH_IN,
        seq_len_out: int = SEQUENCE_LENGTH_OUT,
        crop_size: int = 64,
        subset_storms: int | None = None,
        seed: int = 42,
    ):
        df = observations if observations is not None else load_observations()
        # Filter valid coordinates and wind readings
        df = df.dropna(subset=["lat", "lon", "wind_kmh"]).sort_values(
            by=["storm_id", "time"]
        ).reset_index(drop=True)

        if subset_storms is not None:
            all_sids = df["storm_id"].unique()[:subset_storms]
            df = df[df["storm_id"].isin(all_sids)].reset_index(drop=True)

        self.seq_len_in = seq_len_in
        self.seq_len_out = seq_len_out
        self.total_len = seq_len_in + seq_len_out
        self.crop_size = crop_size
        self._rng = np.random.default_rng(seed)

        # Build sequence metadata per storm
        self.sequences = []
        self.storm_ids = []

        grouped = df.groupby("storm_id")
        for storm_id, group in grouped:
            n_fixes = len(group)
            if n_fixes >= self.total_len:
                indices = group.index.values
                n_windows = n_fixes - self.total_len + 1
                for start in range(n_windows):
                    self.sequences.append({
                        "storm_id": storm_id,
                        "in_indices": indices[start : start + seq_len_in],
                        "out_indices": indices[start + seq_len_in : start + self.total_len],
                    })
                    self.storm_ids.append(storm_id)

        self.df = df
        self.unique_storms = np.array(list(set(self.storm_ids)), dtype=str)

    def __len__(self) -> int:
        return len(self.sequences)

    def _get_frame(self, row: pd.Series) -> np.ndarray:
        real_path = INSAT_DIR / str(row["storm_id"]) / f"{row['time']:%Y%m%d%H%M}.npy"
        if real_path.exists():
            frame = np.load(real_path).astype(np.float32)
        else:
            cat = int(row.get("category_from_grade", 1)) if pd.notna(row.get("category_from_grade")) else wind_speed_to_category(row["wind_kmh"])
            base = _get_synthetic_base(cat, self.crop_size * 2)
            noise = self._rng.normal(0, 3, size=base.shape).astype(np.float32)
            frame = base + noise

        h, w = frame.shape
        cropped = crop_to_storm_center(frame, h // 2, w // 2, crop_size=self.crop_size)
        return normalize_brightness_temperature(cropped)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | float | str]:
        seq_meta = self.sequences[idx]
        in_rows = self.df.iloc[seq_meta["in_indices"]]
        out_rows = self.df.iloc[seq_meta["out_indices"]]

        # Past frames: (seq_len_in, 1, H, W)
        in_frames = [self._get_frame(row) for _, row in in_rows.iterrows()]
        x_seq = torch.from_numpy(np.stack(in_frames)).unsqueeze(1).float()

        # Future target frames: (seq_len_out, 1, H, W)
        out_frames = [self._get_frame(row) for _, row in out_rows.iterrows()]
        y_frames = torch.from_numpy(np.stack(out_frames)).unsqueeze(1).float()

        # Reference position at last observed step (t_in - 1)
        ref_row = in_rows.iloc[-1]
        ref_lat = float(ref_row["lat"])
        ref_lon = float(ref_row["lon"])

        # Target track offsets: (seq_len_out, 3) -> [delta_lat, delta_lon, wind_kmh]
        track_targets = []
        true_lats = []
        true_lons = []
        true_winds = []

        for _, row in out_rows.iterrows():
            d_lat = float(row["lat"]) - ref_lat
            d_lon = float(row["lon"]) - ref_lon
            wind = float(row["wind_kmh"])
            track_targets.append([d_lat, d_lon, wind])
            true_lats.append(float(row["lat"]))
            true_lons.append(float(row["lon"]))
            true_winds.append(wind)

        return {
            "x_seq": x_seq,
            "y_frames": y_frames,
            "y_track": torch.tensor(track_targets, dtype=torch.float32),
            "ref_lat": torch.tensor(ref_lat, dtype=torch.float32),
            "ref_lon": torch.tensor(ref_lon, dtype=torch.float32),
            "true_lats": torch.tensor(true_lats, dtype=torch.float32),
            "true_lons": torch.tensor(true_lons, dtype=torch.float32),
            "true_winds": torch.tensor(true_winds, dtype=torch.float32),
            "storm_id": seq_meta["storm_id"],
        }


def compute_loss(
    pred_frames: torch.Tensor,
    pred_track: torch.Tensor,
    target_frames: torch.Tensor,
    target_track: torch.Tensor,
    loss_fn: nn.MSELoss,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Multi-task loss: Coordinate offset MSE + Scaled Wind MSE + Frame Reconstruction MSE.
    """
    # Position offsets: dlat, dlon
    pos_loss = loss_fn(pred_track[:, :, :2], target_track[:, :, :2])

    # Intensity loss: scale wind by 100 for balance
    wind_loss = loss_fn(pred_track[:, :, 2] / 100.0, target_track[:, :, 2] / 100.0)

    # Future frame reconstruction loss
    frame_loss = loss_fn(pred_frames, target_frames)

    total_loss = pos_loss + 0.1 * wind_loss + 0.5 * frame_loss
    return total_loss, pos_loss, wind_loss


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    loss_fn: nn.MSELoss,
    train: bool = True,
) -> tuple[float, float, float, dict[int, float], float]:
    """Run one epoch of training or validation."""
    model.train() if train else model.eval()

    total_loss = 0.0
    total_pos_loss = 0.0
    total_wind_loss = 0.0
    n_samples = 0

    all_pred_lats = []
    all_pred_lons = []
    all_true_lats = []
    all_true_lons = []
    all_pred_winds = []
    all_true_winds = []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            x_seq = batch["x_seq"].to(device)
            y_frames = batch["y_frames"].to(device)
            y_track = batch["y_track"].to(device)
            bs = x_seq.size(0)

            pred_frames, pred_track = model(x_seq)
            loss, p_loss, w_loss = compute_loss(pred_frames, pred_track, y_frames, y_track, loss_fn)

            if train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * bs
            total_pos_loss += p_loss.item() * bs
            total_wind_loss += w_loss.item() * bs
            n_samples += bs

            # Accumulate predictions for Haversine error computation
            ref_lat = batch["ref_lat"].numpy()[:, None]  # (B, 1)
            ref_lon = batch["ref_lon"].numpy()[:, None]

            pred_track_cpu = pred_track.detach().cpu().numpy()
            pred_lats = ref_lat + pred_track_cpu[:, :, 0]
            pred_lons = ref_lon + pred_track_cpu[:, :, 1]
            pred_winds = pred_track_cpu[:, :, 2]

            all_pred_lats.append(pred_lats)
            all_pred_lons.append(pred_lons)
            all_true_lats.append(batch["true_lats"].numpy())
            all_true_lons.append(batch["true_lons"].numpy())
            all_pred_winds.append(pred_winds)
            all_true_winds.append(batch["true_winds"].numpy())

    avg_loss = total_loss / max(1, n_samples)
    avg_pos_loss = total_pos_loss / max(1, n_samples)
    avg_wind_loss = total_wind_loss / max(1, n_samples)

    # Compute step-wise Haversine distance errors
    pred_lats = np.concatenate(all_pred_lats, axis=0)
    pred_lons = np.concatenate(all_pred_lons, axis=0)
    true_lats = np.concatenate(all_true_lats, axis=0)
    true_lons = np.concatenate(all_true_lons, axis=0)

    pred_winds = np.concatenate(all_pred_winds, axis=0)
    true_winds = np.concatenate(all_true_winds, axis=0)

    seq_len_out = pred_lats.shape[1]
    step_track_errors = {}
    for step in range(seq_len_out):
        errs = haversine_km(
            pred_lats[:, step], pred_lons[:, step],
            true_lats[:, step], true_lons[:, step]
        )
        step_track_errors[step + 1] = float(np.mean(errs))

    mean_wind_err = float(np.mean(np.abs(pred_winds - true_winds)))
    return avg_loss, avg_pos_loss, avg_wind_loss, step_track_errors, mean_wind_err


def main():
    parser = argparse.ArgumentParser(description="Train Model C (ConvLSTM) Track & Intensity Predictor")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--subset-storms", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build sequence dataset
    dataset = CycloneSequenceDataset(
        seq_len_in=SEQUENCE_LENGTH_IN,
        seq_len_out=SEQUENCE_LENGTH_OUT,
        crop_size=args.crop_size,
        subset_storms=args.subset_storms,
    )
    print(f"Total valid sequences: {len(dataset)} from {len(dataset.unique_storms)} storms")

    if len(dataset) == 0:
        print("No sequences met the minimum length requirements.")
        return

    # Leak-free storm-level split
    all_storms = dataset.unique_storms.copy()
    rng = np.random.default_rng(42)
    rng.shuffle(all_storms)

    n_val = max(1, int(len(all_storms) * args.val_split))
    val_storms = set(all_storms[:n_val])
    train_storms = set(all_storms[n_val:])

    train_idx = [i for i, meta in enumerate(dataset.sequences) if meta["storm_id"] in train_storms]
    val_idx = [i for i, meta in enumerate(dataset.sequences) if meta["storm_id"] in val_storms]

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    print(f"Train sequences: {len(train_ds)} ({len(train_storms)} storms) | "
          f"Val sequences: {len(val_ds)} ({len(val_storms)} storms) — Split by Storm ID")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = CyclonePredictor(
        in_channels=1,
        hidden_channels=args.hidden_channels,
        seq_len_out=SEQUENCE_LENGTH_OUT,
        track_features=3,
    ).to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    print("\nStarting ConvLSTM Trajectory & Intensity Training...")

    for epoch in range(1, args.epochs + 1):
        t_loss, _, _, _, _ = run_epoch(
            model, train_loader, optimizer, device, loss_fn, train=True
        )
        v_loss, _, _, val_track_errs, val_wind_err = run_epoch(
            model, val_loader, None, device, loss_fn, train=False
        )

        saved = ""
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), "predictor_checkpoint.pt")
            saved = " [Saved Best Checkpoint]"

        t6h = val_track_errs.get(1, 0.0)
        t24h = val_track_errs.get(SEQUENCE_LENGTH_OUT, 0.0)
        print(f"Epoch {epoch}/{args.epochs} | train_loss={t_loss:.4f} val_loss={v_loss:.4f} | "
              f"Track Error: +6h={t6h:.1f}km, +24h={t24h:.1f}km | Wind MAE={val_wind_err:.1f}km/h{saved}")

    print("\n" + "=" * 60)
    print("Final Model C Validation Report (Held-Out Storms):")
    for step, err in val_track_errs.items():
        hours = step * 6
        print(f"  Forecast +{hours:02d}h Lead Time: Mean Track Error = {err:.1f} km")
    print(f"  Intensity Forecast:     Mean Absolute Error = {val_wind_err:.1f} km/h")
    print("=" * 60)
    print("Trained Model C weights saved to predictor_checkpoint.pt")


if __name__ == "__main__":
    main()
