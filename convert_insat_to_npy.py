"""
Convert raw INSAT-3DR L1C HDF5 files (MOSDAC_DATA/*.h5) into the
128x128, storm-centred .npy crops your models actually train on, saved as
data/raw/insat/<storm_id>/<timestamp>.npy — exactly what
src/data/dataset.py's CycloneClassificationDataset looks for.

Verified independently before being handed to you (see chat) —
- The Mercator projection math reproduces this exact file's own documented
  corner coordinates to within millimetres.
- The best-track time-interpolation logic is checked against known inputs.
What is NOT yet tested: running against your actual best-track CSV and
your actual bulk file set, since neither exists in the environment this
was written in. Run it on ONE file first (instructions at the bottom)
before pointing it at all 2,818.

Usage:
    python convert_insat_to_npy.py --storm-name BIPARJOY \
        --raw-dir "D:\\Hackathon Cyclovision AI\\cyclovision-ai\\MOSDAC_DATA" \
        --besttrack-csv "data\\raw\\ibtracs\\imd_besttrack_observations.csv" \
        --storms-csv "data\\raw\\ibtracs\\imd_besttrack_storms.csv"
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

CROP_SIZE = 128

# --- Mercator projection: verified against this exact product's own
# documented corner coordinates (upper_left_xy) to sub-millimetre accuracy ---
def latlon_to_mercator_xy(lat_deg, lon_deg, a, b, lat_std_deg, lon0_deg):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    lat_std = np.radians(lat_std_deg)
    lon0 = np.radians(lon0_deg)
    e2 = 1 - (b**2 / a**2)
    e = np.sqrt(e2)
    k0 = np.cos(lat_std) / np.sqrt(1 - e2 * np.sin(lat_std)**2)
    x = a * k0 * (lon - lon0)
    y = a * k0 * np.log(np.tan(np.pi/4 + lat/2) * ((1 - e*np.sin(lat)) / (1 + e*np.sin(lat)))**(e/2))
    return x, y


def interpolate_position(obs_storm: pd.DataFrame, target_time: pd.Timestamp):
    """Linear-interpolate the storm's lat/lon at an arbitrary timestamp
    between the two bracketing best-track fixes."""
    obs_storm = obs_storm.sort_values("time")
    if target_time <= obs_storm["time"].iloc[0]:
        row = obs_storm.iloc[0]
        return row["lat"], row["lon"]
    if target_time >= obs_storm["time"].iloc[-1]:
        row = obs_storm.iloc[-1]
        return row["lat"], row["lon"]
    idx = obs_storm["time"].searchsorted(target_time)
    t0, t1 = obs_storm["time"].iloc[idx - 1], obs_storm["time"].iloc[idx]
    frac = (target_time - t0) / (t1 - t0)
    lat = obs_storm["lat"].iloc[idx-1] + frac * (obs_storm["lat"].iloc[idx] - obs_storm["lat"].iloc[idx-1])
    lon = obs_storm["lon"].iloc[idx-1] + frac * (obs_storm["lon"].iloc[idx] - obs_storm["lon"].iloc[idx-1])
    return lat, lon


def parse_acquisition_time(f: h5py.File) -> pd.Timestamp:
    date_str = f.attrs["Acquisition_Date"]
    time_str = f.attrs["Acquisition_Time_in_GMT"]
    if isinstance(date_str, bytes):
        date_str = date_str.decode()
    if isinstance(time_str, bytes):
        time_str = time_str.decode()
    # e.g. "06JUN2023" + "0015" -> 2023-06-06 00:15
    return pd.to_datetime(f"{date_str} {time_str}", format="%d%b%Y %H%M")


def convert_one_file(h5_path: Path, storm_obs: pd.DataFrame, storm_id: str, out_dir: Path,
                      crop_size: int = CROP_SIZE, verbose: bool = False) -> str:
    with h5py.File(h5_path, "r") as f:
        ts = parse_acquisition_time(f)

        bt = f["TIR1_BT"][0]  # (1616, 1737) after dropping the singleton time dim
        fill_value = f["TIR1_BT"].attrs.get("_FillValue", -999.0)
        bt = np.where(bt == fill_value, np.nan, bt)

        proj = f["Projection_Information"]
        a = proj.attrs["semi_major_axis"][0]
        b = proj.attrs["semi_minor_axis"][0]
        lat_std = proj.attrs["standard_parallel"][0]
        lon0 = proj.attrs["longitude_of_projection_origin"][0]

        X = f["X"][:]  # (1737,) projection x-coords, metres
        Y = f["Y"][:]  # (1616,) projection y-coords, metres

    lat, lon = interpolate_position(storm_obs, ts)
    storm_x, storm_y = latlon_to_mercator_xy(lat, lon, a, b, lat_std, lon0)

    col = int(np.argmin(np.abs(X - storm_x)))
    row = int(np.argmin(np.abs(Y - storm_y)))

    # Bounds sanity check -- if the storm centre is near/outside this
    # frame's edge, warn rather than silently return a bad crop.
    half = crop_size // 2
    if row - half < 0 or row + half > bt.shape[0] or col - half < 0 or col + half > bt.shape[1]:
        if verbose:
            print(f"  WARNING: {h5_path.name}: storm centre near frame edge "
                  f"(row={row}/{bt.shape[0]}, col={col}/{bt.shape[1]}) -- crop will be edge-padded.")

    from numpy import pad as np_pad
    pad_amt = half + 1
    padded = np_pad(bt, pad_amt, mode="edge")
    r, c = row + pad_amt, col + pad_amt
    crop = padded[r - half: r + half, c - half: c + half]

    out_subdir = out_dir / storm_id
    out_subdir.mkdir(parents=True, exist_ok=True)
    out_path = out_subdir / f"{ts:%Y%m%d%H%M}.npy"
    np.save(out_path, crop.astype(np.float32))
    return str(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--storm-name", required=True, help="e.g. BIPARJOY (matched case-insensitively)")
    ap.add_argument("--raw-dir", required=True, help="Folder containing the downloaded .h5 files")
    ap.add_argument("--besttrack-csv", default="data/raw/ibtracs/imd_besttrack_observations.csv")
    ap.add_argument("--storms-csv", default="data/raw/ibtracs/imd_besttrack_storms.csv")
    ap.add_argument("--out-dir", default="data/raw/insat")
    ap.add_argument("--limit", type=int, default=None, help="Process only the first N files -- use this for a first test run")
    args = ap.parse_args()

    storms = pd.read_csv(args.storms_csv)
    match = storms[storms["name"].str.upper() == args.storm_name.upper()]
    if match.empty:
        print(f"No storm named '{args.storm_name}' found in {args.storms_csv}. "
              f"Available names (sample): {storms['name'].dropna().unique()[:20]}")
        return
    storm_id = match.iloc[0]["storm_id"]
    print(f"Matched '{args.storm_name}' -> storm_id = {storm_id}")

    obs = pd.read_csv(args.besttrack_csv, parse_dates=["time"])
    storm_obs = obs[obs["storm_id"] == storm_id]
    if storm_obs.empty:
        print(f"No best-track observations found for storm_id {storm_id} in {args.besttrack_csv}")
        return
    print(f"Found {len(storm_obs)} best-track fixes for this storm "
          f"({storm_obs['time'].min()} to {storm_obs['time'].max()})")

    raw_dir = Path(args.raw_dir)
    h5_files = sorted(raw_dir.glob("*.h5"))
    if args.limit:
        h5_files = h5_files[: args.limit]
    print(f"Found {len(h5_files)} .h5 files to convert" + (f" (limited to first {args.limit})" if args.limit else ""))

    out_dir = Path(args.out_dir)
    ok, failed = 0, []
    for i, h5_path in enumerate(h5_files, 1):
        try:
            out_path = convert_one_file(h5_path, storm_obs, storm_id, out_dir, verbose=True)
            ok += 1
            if i <= 3 or i % 200 == 0:
                print(f"  [{i}/{len(h5_files)}] {h5_path.name} -> {out_path}")
        except Exception as e:
            failed.append((h5_path.name, str(e)))
            print(f"  [{i}/{len(h5_files)}] FAILED: {h5_path.name}: {e}")

    print(f"\nDone. {ok} converted, {len(failed)} failed.")
    if failed:
        print("Failed files:", failed[:10], "..." if len(failed) > 10 else "")


if __name__ == "__main__":
    main()
