"""
Batch version of convert_insat_to_npy.py -- converts a whole folder of
.h5 files spanning MULTIPLE storms in one pass, auto-detecting which
storm each file belongs to by matching its acquisition timestamp against
every storm's [start_time, end_time] window in imd_besttrack_storms.csv.
No need to run this once per storm or keep storms in separate folders.

Same verified Mercator projection + interpolation logic as the original
single-storm converter -- this only changes HOW a file's storm_id gets
decided, not the crop/projection math itself.

Usage:
    python convert_insat_batch.py --raw-dir "D:\\Hackathon Cyclovision AI\\cyclovision-ai\\MOSDAC_DATA"

Safety: if a file's timestamp falls inside TWO storms' windows at once
(rare, but real -- e.g. Gulab and Shaheen were concurrent in 2021), it is
reported as AMBIGUOUS and skipped rather than silently guessed, so you
can resolve it manually (e.g. by checking the file's lat/lon center
against each storm's basin).
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

CROP_SIZE = 128
PAD_HOURS = 6  # small tolerance around a storm's official start/end, since
                # a "trimmed" download range or a slightly early/late frame
                # shouldn't be discarded just for missing the exact boundary


def latlon_to_mercator_xy(lat_deg, lon_deg, a, b, lat_std_deg, lon0_deg):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    lat_std, lon0 = np.radians(lat_std_deg), np.radians(lon0_deg)
    e2 = 1 - (b**2 / a**2)
    e = np.sqrt(e2)
    k0 = np.cos(lat_std) / np.sqrt(1 - e2 * np.sin(lat_std)**2)
    x = a * k0 * (lon - lon0)
    y = a * k0 * np.log(np.tan(np.pi/4 + lat/2) * ((1 - e*np.sin(lat)) / (1 + e*np.sin(lat)))**(e/2))
    return x, y


def interpolate_position(obs_storm: pd.DataFrame, target_time: pd.Timestamp):
    obs_storm = obs_storm.sort_values("time")
    if target_time <= obs_storm["time"].iloc[0]:
        row = obs_storm.iloc[0]; return row["lat"], row["lon"]
    if target_time >= obs_storm["time"].iloc[-1]:
        row = obs_storm.iloc[-1]; return row["lat"], row["lon"]
    idx = obs_storm["time"].searchsorted(target_time)
    t0, t1 = obs_storm["time"].iloc[idx-1], obs_storm["time"].iloc[idx]
    frac = (target_time - t0) / (t1 - t0)
    lat = obs_storm["lat"].iloc[idx-1] + frac * (obs_storm["lat"].iloc[idx] - obs_storm["lat"].iloc[idx-1])
    lon = obs_storm["lon"].iloc[idx-1] + frac * (obs_storm["lon"].iloc[idx] - obs_storm["lon"].iloc[idx-1])
    return lat, lon


def parse_acquisition_time(f: h5py.File) -> pd.Timestamp:
    date_str = f.attrs["Acquisition_Date"]
    time_str = f.attrs["Acquisition_Time_in_GMT"]
    if isinstance(date_str, bytes): date_str = date_str.decode()
    if isinstance(time_str, bytes): time_str = time_str.decode()
    return pd.to_datetime(f"{date_str} {time_str}", format="%d%b%Y %H%M")


def match_storm(ts: pd.Timestamp, storms: pd.DataFrame) -> tuple[str | None, str]:
    """Returns (storm_id or None, status). status is 'ok', 'ambiguous', or 'no_match'."""
    pad = pd.Timedelta(hours=PAD_HOURS)
    matches = storms[(storms["start_time"] - pad <= ts) & (ts <= storms["end_time"] + pad)]
    if len(matches) == 1:
        return matches.iloc[0]["storm_id"], "ok"
    elif len(matches) > 1:
        return None, "ambiguous: " + ", ".join(matches["storm_id"].tolist())
    else:
        return None, "no_match"


def convert_one_file(h5_path: Path, storm_obs: pd.DataFrame, storm_id: str, out_dir: Path,
                      crop_size: int = CROP_SIZE) -> str:
    with h5py.File(h5_path, "r") as f:
        ts = parse_acquisition_time(f)
        bt = f["TIR1_BT"][0]
        fill_value = f["TIR1_BT"].attrs.get("_FillValue", -999.0)
        bt = np.where(bt == fill_value, np.nan, bt)

        proj = f["Projection_Information"]
        a, b = proj.attrs["semi_major_axis"][0], proj.attrs["semi_minor_axis"][0]
        lat_std, lon0 = proj.attrs["standard_parallel"][0], proj.attrs["longitude_of_projection_origin"][0]
        X, Y = f["X"][:], f["Y"][:]

    lat, lon = interpolate_position(storm_obs, ts)
    storm_x, storm_y = latlon_to_mercator_xy(lat, lon, a, b, lat_std, lon0)
    col = int(np.argmin(np.abs(X - storm_x)))
    row = int(np.argmin(np.abs(Y - storm_y)))

    half = crop_size // 2
    pad_amt = half + 1
    padded = np.pad(bt, pad_amt, mode="edge")
    r, c = row + pad_amt, col + pad_amt
    crop = padded[r - half: r + half, c - half: c + half]

    out_subdir = out_dir / storm_id
    out_subdir.mkdir(parents=True, exist_ok=True)
    out_path = out_subdir / f"{ts:%Y%m%d%H%M}.npy"
    np.save(out_path, crop.astype(np.float32))
    return str(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, help="Folder containing .h5 files from ANY number of storms")
    ap.add_argument("--besttrack-csv", default="data/raw/ibtracs/imd_besttrack_observations.csv")
    ap.add_argument("--storms-csv", default="data/raw/ibtracs/imd_besttrack_storms.csv")
    ap.add_argument("--out-dir", default="data/raw/insat")
    args = ap.parse_args()

    storms = pd.read_csv(args.storms_csv, parse_dates=["start_time", "end_time"])
    obs = pd.read_csv(args.besttrack_csv, parse_dates=["time"])

    raw_dir = Path(args.raw_dir)
    h5_files = sorted(raw_dir.rglob("*.h5"))
    print(f"Found {len(h5_files)} .h5 files under {raw_dir}\n")

    out_dir = Path(args.out_dir)
    per_storm_count: dict[str, int] = {}
    failed, ambiguous, unmatched = [], [], []
    already_done = 0

    for i, h5_path in enumerate(h5_files, 1):
        try:
            with h5py.File(h5_path, "r") as f:
                ts = parse_acquisition_time(f)
            storm_id, status = match_storm(ts, storms)

            if status == "no_match":
                unmatched.append((h5_path.name, str(ts)))
                continue
            if status.startswith("ambiguous"):
                ambiguous.append((h5_path.name, str(ts), status))
                continue

            # Skip if already converted -- makes re-running this script safe
            # and fast every time you add a new storm folder later, instead
            # of reprocessing everything from scratch each time.
            expected_out = out_dir / storm_id / f"{ts:%Y%m%d%H%M}.npy"
            if expected_out.exists():
                already_done += 1
                continue

            storm_obs = obs[obs["storm_id"] == storm_id]
            if storm_obs.empty:
                unmatched.append((h5_path.name, f"matched {storm_id} but no best-track observations for it"))
                continue

            convert_one_file(h5_path, storm_obs, storm_id, out_dir)
            per_storm_count[storm_id] = per_storm_count.get(storm_id, 0) + 1

            if i % 500 == 0:
                print(f"  ...{i}/{len(h5_files)} processed")

        except Exception as e:
            failed.append((h5_path.name, str(e)))

    print(f"\n{'='*60}\nDONE\n{'='*60}")
    print(f"Converted per storm (this run):")
    for sid, n in sorted(per_storm_count.items()):
        name = storms.loc[storms["storm_id"] == sid, "name"].values
        name = name[0] if len(name) else "?"
        print(f"  {sid} ({name}): {n} files")
    print(f"\nNewly converted: {sum(per_storm_count.values())}  |  "
          f"Already done (skipped): {already_done}  |  "
          f"Failed: {len(failed)}  |  Ambiguous (skipped): {len(ambiguous)}  |  No matching storm: {len(unmatched)}")

    if ambiguous:
        print(f"\nAMBIGUOUS files (matched >1 storm -- resolve manually, none converted):")
        for name, ts, status in ambiguous[:10]:
            print(f"  {name} @ {ts}: {status}")
    if unmatched:
        print(f"\nUNMATCHED files (no storm covers this timestamp -- check for typos in dates, or a storm missing from storms.csv):")
        for name, reason in unmatched[:10]:
            print(f"  {name}: {reason}")
    if failed:
        print(f"\nFAILED files (error during conversion):")
        for name, err in failed[:10]:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
