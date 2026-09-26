"""
convert_pending.py  --  Convert ONLY the H5 files that don't yet have a
corresponding .npy in data/raw/insat/.

This is a targeted wrapper around the same projection+crop logic used by
convert_insat_batch.py, but it:
  1. Scans MOSDAC_DATA/ recursively to find every .h5 file
  2. Derives the expected .npy path for each one (same naming as batch converter)
  3. Skips files whose .npy already exists  (fast -- just a Path.exists() check)
  4. Only processes truly pending files, so re-running is always safe and fast
  5. Prints a per-storm summary BEFORE converting so you can see what's pending
  6. Shows a live progress bar with ETA during conversion

Usage:
    cd "D:\\Hackathon Cyclovision AI\\cyclovision-ai"
    python convert_pending.py

Optional flags:
    --dry-run          Show what would be converted without doing it
    --storm 2019-007   Restrict to a single storm (by storm_id)
    --raw-dir PATH     Override default MOSDAC_DATA path
    --out-dir PATH     Override default data/raw/insat path
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# ── same constants as convert_insat_batch.py ──────────────────────────────────
CROP_SIZE = 128
PAD_HOURS = 6


# ── projection helpers (verbatim from convert_insat_batch.py) ─────────────────

def latlon_to_mercator_xy(lat_deg, lon_deg, a, b, lat_std_deg, lon0_deg):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    lat_std, lon0 = np.radians(lat_std_deg), np.radians(lon0_deg)
    e2 = 1 - (b**2 / a**2)
    e = np.sqrt(e2)
    k0 = np.cos(lat_std) / np.sqrt(1 - e2 * np.sin(lat_std)**2)
    x = a * k0 * (lon - lon0)
    y = a * k0 * np.log(
        np.tan(np.pi / 4 + lat / 2)
        * ((1 - e * np.sin(lat)) / (1 + e * np.sin(lat))) ** (e / 2)
    )
    return x, y


def interpolate_position(obs_storm: pd.DataFrame, target_time: pd.Timestamp):
    obs_storm = obs_storm.sort_values("time")
    if target_time <= obs_storm["time"].iloc[0]:
        row = obs_storm.iloc[0]
        return row["lat"], row["lon"]
    if target_time >= obs_storm["time"].iloc[-1]:
        row = obs_storm.iloc[-1]
        return row["lat"], row["lon"]
    idx = obs_storm["time"].searchsorted(target_time)
    t0 = obs_storm["time"].iloc[idx - 1]
    t1 = obs_storm["time"].iloc[idx]
    frac = (target_time - t0) / (t1 - t0)
    lat = obs_storm["lat"].iloc[idx - 1] + frac * (
        obs_storm["lat"].iloc[idx] - obs_storm["lat"].iloc[idx - 1]
    )
    lon = obs_storm["lon"].iloc[idx - 1] + frac * (
        obs_storm["lon"].iloc[idx] - obs_storm["lon"].iloc[idx - 1]
    )
    return lat, lon


def parse_acquisition_time(f: h5py.File) -> pd.Timestamp:
    date_str = f.attrs["Acquisition_Date"]
    time_str = f.attrs["Acquisition_Time_in_GMT"]
    if isinstance(date_str, bytes):
        date_str = date_str.decode()
    if isinstance(time_str, bytes):
        time_str = time_str.decode()
    return pd.to_datetime(f"{date_str} {time_str}", format="%d%b%Y %H%M")


def match_storm(ts: pd.Timestamp, storms: pd.DataFrame) -> tuple[str | None, str]:
    pad = pd.Timedelta(hours=PAD_HOURS)
    matches = storms[
        (storms["start_time"] - pad <= ts) & (ts <= storms["end_time"] + pad)
    ]
    if len(matches) == 1:
        return matches.iloc[0]["storm_id"], "ok"
    elif len(matches) > 1:
        return None, "ambiguous: " + ", ".join(matches["storm_id"].tolist())
    else:
        return None, "no_match"


def convert_one_file(
    h5_path: Path,
    storm_obs: pd.DataFrame,
    storm_id: str,
    out_dir: Path,
    crop_size: int = CROP_SIZE,
) -> str:
    with h5py.File(h5_path, "r") as f:
        ts = parse_acquisition_time(f)
        bt = f["TIR1_BT"][0]
        fill_value = f["TIR1_BT"].attrs.get("_FillValue", -999.0)
        bt = np.where(bt == fill_value, np.nan, bt)

        proj = f["Projection_Information"]
        a = proj.attrs["semi_major_axis"][0]
        b = proj.attrs["semi_minor_axis"][0]
        lat_std = proj.attrs["standard_parallel"][0]
        lon0 = proj.attrs["longitude_of_projection_origin"][0]
        X, Y = f["X"][:], f["Y"][:]

    lat, lon = interpolate_position(storm_obs, ts)
    storm_x, storm_y = latlon_to_mercator_xy(lat, lon, a, b, lat_std, lon0)
    col = int(np.argmin(np.abs(X - storm_x)))
    row = int(np.argmin(np.abs(Y - storm_y)))

    half = crop_size // 2
    pad_amt = half + 1
    padded = np.pad(bt, pad_amt, mode="edge")
    r, c = row + pad_amt, col + pad_amt
    crop = padded[r - half : r + half, c - half : c + half]

    out_subdir = out_dir / storm_id
    out_subdir.mkdir(parents=True, exist_ok=True)
    out_path = out_subdir / f"{ts:%Y%m%d%H%M}.npy"
    np.save(out_path, crop.astype(np.float32))
    return str(out_path)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Convert only pending (unconverted) INSAT H5 files to .npy")
    ap.add_argument(
        "--raw-dir",
        default=r"D:\Hackathon Cyclovision AI\cyclovision-ai\MOSDAC_DATA",
        help="Root folder containing per-storm H5 files (default: MOSDAC_DATA)",
    )
    ap.add_argument(
        "--besttrack-csv",
        default="data/raw/ibtracs/imd_besttrack_observations.csv",
    )
    ap.add_argument(
        "--storms-csv",
        default="data/raw/ibtracs/imd_besttrack_storms.csv",
    )
    ap.add_argument("--out-dir", default="data/raw/insat")
    ap.add_argument(
        "--storm",
        default=None,
        help="Only process a specific storm_id, e.g. --storm 2019-007",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be converted without writing any files",
    )
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    if not raw_dir.exists():
        print(f"ERROR: raw-dir not found: {raw_dir}")
        sys.exit(1)

    print(f"Loading best-track data...")
    storms = pd.read_csv(args.storms_csv, parse_dates=["start_time", "end_time"])
    obs = pd.read_csv(args.besttrack_csv, parse_dates=["time"])

    # ── Step 1: Scan all H5 files ──────────────────────────────────────────
    print(f"Scanning {raw_dir} for .h5 files...")
    all_h5 = sorted(raw_dir.rglob("*.h5"))
    print(f"Found {len(all_h5)} total .h5 files\n")

    # ── Step 2: Quick-check which ones are already converted ───────────────
    # We peek at each file's timestamp attribute to determine the expected .npy
    # path. This is faster than re-running the full projection math.
    print("Checking which files are already converted (this may take ~30s for large sets)...")

    pending: list[tuple[Path, str]] = []   # (h5_path, storm_id)
    done_count = 0
    no_match_count = 0
    ambiguous_count = 0

    t0 = time.time()
    for i, h5_path in enumerate(all_h5, 1):
        try:
            with h5py.File(h5_path, "r") as f:
                ts = parse_acquisition_time(f)

            storm_id, status = match_storm(ts, storms)

            if status == "no_match":
                no_match_count += 1
                continue
            if status.startswith("ambiguous"):
                ambiguous_count += 1
                continue

            if args.storm and storm_id != args.storm:
                continue  # skip storms not requested

            expected_npy = out_dir / storm_id / f"{ts:%Y%m%d%H%M}.npy"
            if expected_npy.exists():
                done_count += 1
            else:
                pending.append((h5_path, storm_id))

        except Exception as e:
            print(f"  WARN: could not read {h5_path.name}: {e}")

        if i % 1000 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (len(all_h5) - i) / rate if rate > 0 else 0
            print(f"  Scanned {i}/{len(all_h5)} ({rate:.0f}/s, ETA {eta:.0f}s)...")

    # ── Step 3: Per-storm summary of pending work ──────────────────────────
    pending_by_storm: dict[str, list[Path]] = {}
    for h5_path, storm_id in pending:
        pending_by_storm.setdefault(storm_id, []).append(h5_path)

    print(f"\n{'='*60}")
    print(f"PENDING CONVERSION SUMMARY")
    print(f"{'='*60}")
    print(f"  Already converted (skipped): {done_count}")
    print(f"  No matching storm (skipped): {no_match_count}")
    print(f"  Ambiguous storm (skipped):   {ambiguous_count}")
    print(f"  PENDING (to convert):        {len(pending)}")
    print()

    if pending_by_storm:
        print("  Pending by storm:")
        for sid in sorted(pending_by_storm):
            name_row = storms.loc[storms["storm_id"] == sid, "name"]
            name = name_row.values[0] if len(name_row) else "?"
            print(f"    {sid} ({name}): {len(pending_by_storm[sid])} files")
    else:
        print("  ✅ Nothing to convert -- all H5 files already have .npy counterparts.")

    print()

    if not pending:
        return

    if args.dry_run:
        print("DRY RUN -- no files written. Remove --dry-run to convert.")
        return

    # ── Step 4: Convert pending files ─────────────────────────────────────
    print(f"Converting {len(pending)} pending file(s)...")
    converted = 0
    failed: list[tuple[str, str]] = []

    t0 = time.time()
    for i, (h5_path, storm_id) in enumerate(pending, 1):
        try:
            storm_obs = obs[obs["storm_id"] == storm_id]
            convert_one_file(h5_path, storm_obs, storm_id, out_dir)
            converted += 1
        except Exception as e:
            failed.append((h5_path.name, str(e)))

        if i % 100 == 0 or i == len(pending):
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            pct = 100 * i / len(pending)
            eta = (len(pending) - i) / rate if rate > 0 else 0
            bar_width = 30
            filled = int(bar_width * i / len(pending))
            bar = "█" * filled + "░" * (bar_width - filled)
            print(
                f"\r  [{bar}] {i}/{len(pending)} ({pct:.0f}%) "
                f"| {rate:.1f}/s | ETA {eta:.0f}s",
                end="",
                flush=True,
            )

    print()  # newline after progress bar

    # ── Step 5: Final summary ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"{'='*60}")
    print(f"  Converted:  {converted}")
    print(f"  Failed:     {len(failed)}")

    if failed:
        print("\nFailed files:")
        for name, err in failed[:20]:
            print(f"  {name}: {err}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")

    print(f"\nNew .npy files are in: {out_dir.resolve()}")
    print("You can now retrain the classifier/predictor to use the extra data.")


if __name__ == "__main__":
    main()
