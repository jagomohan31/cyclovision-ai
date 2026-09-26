"""
Convert ALL cyclone INSAT-3DR H5 files to .npy crops in one pass.

Scans every subfolder under --raw-dir (Mocha, Biparjoy, Tauktae, etc.),
auto-detects which storm each H5 file belongs to via timestamp matching
against imd_besttrack_storms.csv, crops a CROP_SIZE x CROP_SIZE patch
centred on the storm's interpolated position, and writes it to:

    data/raw/insat/<storm_id>/<YYYYMMDDHHMM>.npy

Features
--------
- Idempotent: skips .npy files that already exist (safe to re-run).
- Per-storm progress bars with ETA.
- Parallel workers (default = CPU count, capped at 8).
- Quarantine: ambiguous / unmatched / failed files go to
    data/raw/insat/quarantine/ with a JSON log.
- Summary table printed at the end.

Usage
-----
    # Convert everything in MOSDAC_DATA (all subfolders):
    python convert_all_cyclones.py

    # Explicit raw directory:
    python convert_all_cyclones.py --raw-dir "D:\\MOSDAC_DATA"

    # Dry-run (count files, show what would happen, write nothing):
    python convert_all_cyclones.py --dry-run

    # Limit workers:
    python convert_all_cyclones.py --workers 4

    # Force re-conversion even if .npy already exists:
    python convert_all_cyclones.py --overwrite
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
BESTTRACK_OBS_CSV  = PROJECT_ROOT / "data" / "raw" / "ibtracs" / "imd_besttrack_observations.csv"
BESTTRACK_STORMS_CSV = PROJECT_ROOT / "data" / "raw" / "ibtracs" / "imd_besttrack_storms.csv"
MOSDAC_DATA_DIR    = PROJECT_ROOT / "MOSDAC_DATA"
OUTPUT_DIR         = PROJECT_ROOT / "data" / "raw" / "insat"
QUARANTINE_DIR     = OUTPUT_DIR / "quarantine"
CROP_SIZE          = 128
PAD_HOURS          = 6   # tolerance around storm start/end window


# ---------------------------------------------------------------------------
# Projection helpers (Mercator, matching convert_insat_batch.py)
# ---------------------------------------------------------------------------
def _latlon_to_mercator_xy(lat_deg, lon_deg, a, b, lat_std_deg, lon0_deg):
    lat = np.radians(lat_deg);  lon  = np.radians(lon_deg)
    lat_std = np.radians(lat_std_deg); lon0 = np.radians(lon0_deg)
    e2 = 1 - (b**2 / a**2);    e  = np.sqrt(e2)
    k0 = np.cos(lat_std) / np.sqrt(1 - e2 * np.sin(lat_std)**2)
    x  = a * k0 * (lon - lon0)
    y  = a * k0 * np.log(
        np.tan(np.pi / 4 + lat / 2)
        * ((1 - e * np.sin(lat)) / (1 + e * np.sin(lat))) ** (e / 2)
    )
    return x, y


def _interpolate_position(obs_storm: pd.DataFrame, target_time: pd.Timestamp):
    obs_storm = obs_storm.sort_values("time")
    if target_time <= obs_storm["time"].iloc[0]:
        row = obs_storm.iloc[0]; return row["lat"], row["lon"]
    if target_time >= obs_storm["time"].iloc[-1]:
        row = obs_storm.iloc[-1]; return row["lat"], row["lon"]
    idx = int(obs_storm["time"].searchsorted(target_time))
    t0, t1 = obs_storm["time"].iloc[idx - 1], obs_storm["time"].iloc[idx]
    frac = (target_time - t0) / (t1 - t0)
    lat = obs_storm["lat"].iloc[idx - 1] + frac * (obs_storm["lat"].iloc[idx] - obs_storm["lat"].iloc[idx - 1])
    lon = obs_storm["lon"].iloc[idx - 1] + frac * (obs_storm["lon"].iloc[idx] - obs_storm["lon"].iloc[idx - 1])
    return lat, lon


def _parse_acquisition_time(f: h5py.File) -> pd.Timestamp:
    date_str = f.attrs["Acquisition_Date"]
    time_str = f.attrs["Acquisition_Time_in_GMT"]
    if isinstance(date_str, bytes): date_str = date_str.decode()
    if isinstance(time_str, bytes): time_str = time_str.decode()
    return pd.to_datetime(f"{date_str} {time_str}", format="%d%b%Y %H%M")


def _match_storm(ts: pd.Timestamp, storms: pd.DataFrame):
    """Returns (storm_id | None, status_str)."""
    pad = pd.Timedelta(hours=PAD_HOURS)
    m = storms[(storms["start_time"] - pad <= ts) & (ts <= storms["end_time"] + pad)]
    if len(m) == 1:
        return m.iloc[0]["storm_id"], "ok"
    elif len(m) > 1:
        return None, "ambiguous:" + ",".join(m["storm_id"].tolist())
    return None, "no_match"


# ---------------------------------------------------------------------------
# Worker task
# ---------------------------------------------------------------------------
@dataclass
class Task:
    h5_path: Path
    dry_run: bool
    overwrite: bool
    # passed by the parent process (read from CSVs once, shared via args)
    storms_records: list   # list of dicts from storms DataFrame
    obs_records: dict      # {storm_id: list of {time, lat, lon}} sorted by time


@dataclass
class Result:
    h5_name: str
    status: str   # "converted" | "skipped" | "dry_run" | "no_match" | "ambiguous" | "failed"
    storm_id: Optional[str] = None
    out_path: Optional[str] = None
    error: Optional[str] = None


def _run_task(task: Task) -> Result:
    h5_path = task.h5_path
    try:
        with h5py.File(h5_path, "r") as f:
            ts = _parse_acquisition_time(f)

            # Rebuild storms DataFrame from records (avoids pickle of full DF)
            storms = pd.DataFrame(task.storms_records)
            storms["start_time"] = pd.to_datetime(storms["start_time"])
            storms["end_time"]   = pd.to_datetime(storms["end_time"])

            storm_id, status = _match_storm(ts, storms)

            if status != "ok":
                return Result(h5_path.name, status, error=status)

            expected_out = OUTPUT_DIR / storm_id / f"{ts:%Y%m%d%H%M}.npy"
            if expected_out.exists() and not task.overwrite:
                return Result(h5_path.name, "skipped", storm_id=storm_id, out_path=str(expected_out))

            if task.dry_run:
                return Result(h5_path.name, "dry_run", storm_id=storm_id, out_path=str(expected_out))

            # Load obs for this storm
            obs_list = task.obs_records.get(storm_id, [])
            if not obs_list:
                return Result(h5_path.name, "failed", storm_id=storm_id, error="no best-track obs")

            obs_df = pd.DataFrame(obs_list)
            obs_df["time"] = pd.to_datetime(obs_df["time"])

            lat, lon = _interpolate_position(obs_df, ts)

            # Read BT array and projection
            bt = f["TIR1_BT"][0]
            fill = f["TIR1_BT"].attrs.get("_FillValue", -999.0)
            bt = np.where(bt == fill, np.nan, bt)

            proj = f["Projection_Information"]
            a    = proj.attrs["semi_major_axis"][0]
            b    = proj.attrs["semi_minor_axis"][0]
            lat_std = proj.attrs["standard_parallel"][0]
            lon0    = proj.attrs["longitude_of_projection_origin"][0]
            X, Y = f["X"][:], f["Y"][:]

        storm_x, storm_y = _latlon_to_mercator_xy(lat, lon, a, b, lat_std, lon0)
        col = int(np.argmin(np.abs(X - storm_x)))
        row = int(np.argmin(np.abs(Y - storm_y)))

        half = CROP_SIZE // 2
        pad_amt = half + 1
        padded = np.pad(bt, pad_amt, mode="edge")
        r, c = row + pad_amt, col + pad_amt
        crop = padded[r - half: r + half, c - half: c + half]

        out_dir = OUTPUT_DIR / storm_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{ts:%Y%m%d%H%M}.npy"
        np.save(out_path, crop.astype(np.float32))

        return Result(h5_path.name, "converted", storm_id=storm_id, out_path=str(out_path))

    except Exception as exc:
        return Result(h5_path.name, "failed", error=str(exc))


# ---------------------------------------------------------------------------
# Simple progress printer
# ---------------------------------------------------------------------------
class Progress:
    def __init__(self, total: int, label: str):
        self.total = total
        self.done  = 0
        self.start = time.time()
        self.label = label

    def tick(self):
        self.done += 1
        elapsed = time.time() - self.start
        rate = self.done / elapsed if elapsed > 0 else 0
        eta  = (self.total - self.done) / rate if rate > 0 else 0
        pct  = 100 * self.done / self.total if self.total else 0
        bar_len = 30
        filled  = int(bar_len * self.done / self.total) if self.total else 0
        bar = "█" * filled + "░" * (bar_len - filled)
        print(
            f"\r  [{bar}] {self.done}/{self.total} ({pct:.0f}%)  "
            f"ETA {eta:.0f}s  {self.label}        ",
            end="", flush=True,
        )

    def finish(self):
        elapsed = time.time() - self.start
        print(f"\r  Done {self.done}/{self.total} in {elapsed:.1f}s" + " " * 30)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Convert ALL cyclone H5 files to .npy crops.")
    ap.add_argument("--raw-dir",    default=str(MOSDAC_DATA_DIR),
                    help="Root folder containing storm subfolders (default: MOSDAC_DATA/)")
    ap.add_argument("--out-dir",    default=str(OUTPUT_DIR),
                    help="Output root (default: data/raw/insat/)")
    ap.add_argument("--obs-csv",    default=str(BESTTRACK_OBS_CSV))
    ap.add_argument("--storms-csv", default=str(BESTTRACK_STORMS_CSV))
    ap.add_argument("--workers",    type=int, default=min(multiprocessing.cpu_count(), 8),
                    help="Parallel worker count (default: min(cpu_count, 8))")
    ap.add_argument("--dry-run",    action="store_true",
                    help="Scan and report without writing any files.")
    ap.add_argument("--overwrite",  action="store_true",
                    help="Re-convert even if .npy already exists.")
    ap.add_argument("--storm",      default=None,
                    help="Restrict to one storm folder name, e.g. --storm Mocha")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        sys.exit(f"[ERROR] --raw-dir does not exist: {raw_dir}")

    # Load reference data once
    print("Loading best-track data...")
    storms_df = pd.read_csv(args.storms_csv, parse_dates=["start_time", "end_time"])
    obs_df    = pd.read_csv(args.obs_csv,    parse_dates=["time"])
    obs_df    = obs_df.dropna(subset=["lat", "lon"]).sort_values(["storm_id", "time"])

    # Pre-group obs by storm_id as list-of-dicts for pickling into workers
    obs_by_storm: dict[str, list] = {}
    for sid, grp in obs_df.groupby("storm_id"):
        obs_by_storm[sid] = grp[["time", "lat", "lon"]].to_dict("records")

    storms_records = storms_df[["storm_id", "start_time", "end_time", "name"]].to_dict("records")

    # Discover H5 files
    pattern = f"{args.storm}/**/*.h5" if args.storm else "**/*.h5"
    h5_files = sorted(raw_dir.glob(pattern))
    if not h5_files:
        sys.exit(f"[ERROR] No .h5 files found under {raw_dir}")

    print(f"Found {len(h5_files):,} H5 files under {raw_dir}")
    if args.dry_run:
        print("DRY-RUN mode — no files will be written.\n")

    # Build tasks
    tasks = [
        Task(
            h5_path=p,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            storms_records=storms_records,
            obs_records=obs_by_storm,
        )
        for p in h5_files
    ]

    # Run
    results: list[Result] = []
    progress = Progress(len(tasks), "converting")

    if args.workers <= 1:
        for t in tasks:
            results.append(_run_task(t))
            progress.tick()
    else:
        print(f"Using {args.workers} parallel workers...")
        with multiprocessing.Pool(processes=args.workers) as pool:
            for r in pool.imap_unordered(_run_task, tasks, chunksize=20):
                results.append(r)
                progress.tick()

    progress.finish()

    # Aggregate stats
    by_status: dict[str, list[Result]] = {}
    for r in results:
        by_status.setdefault(r.status, []).append(r)

    by_storm: dict[str, dict[str, int]] = {}
    for r in results:
        if r.storm_id:
            entry = by_storm.setdefault(r.storm_id, {"converted": 0, "skipped": 0, "dry_run": 0})
            if r.status in entry:
                entry[r.status] += 1

    # Summary table
    print("\n" + "=" * 70)
    print("CONVERSION SUMMARY")
    print("=" * 70)
    col = "Converted" if not args.dry_run else "Would convert"
    _key = 'converted' if not args.dry_run else 'dry_run'
    print(f"  {col}   : {len(by_status.get(_key, [])):>6,}")
    print(f"  Skipped (exists): {len(by_status.get('skipped', [])):>6,}")
    print(f"  No match        : {len(by_status.get('no_match', [])):>6,}")
    print(f"  Ambiguous       : {len(by_status.get('ambiguous', [])):>6,}")
    print(f"  Failed          : {len(by_status.get('failed',   [])):>6,}")
    print()
    print(f"  {'Storm ID':<14} {'Name':<16} {'Converted':>10} {'Skipped':>9}")
    print(f"  {'-'*14} {'-'*16} {'-'*10} {'-'*9}")

    # Build a storm name lookup
    storm_name_map = {r["storm_id"]: str(r.get("name", "")) for r in storms_records}
    key = "dry_run" if args.dry_run else "converted"
    for sid in sorted(by_storm):
        entry = by_storm[sid]
        name  = storm_name_map.get(sid, "")
        print(f"  {sid:<14} {name:<16} {entry.get(key, 0):>10,} {entry.get('skipped', 0):>9,}")

    # Write quarantine log
    quarantine_items = (
        by_status.get("no_match",  []) +
        by_status.get("ambiguous", []) +
        by_status.get("failed",    [])
    )
    if quarantine_items and not args.dry_run:
        QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
        log_path = QUARANTINE_DIR / f"conversion_log_{datetime.now():%Y%m%d_%H%M%S}.json"
        log_data = [
            {"file": r.h5_name, "status": r.status, "error": r.error}
            for r in quarantine_items
        ]
        log_path.write_text(json.dumps(log_data, indent=2))
        print(f"\n  Quarantine log → {log_path}")

    print("=" * 70)

    if not args.dry_run and by_status.get("converted"):
        total_conv = len(by_status["converted"])
        print(f"\n✅  {total_conv:,} new .npy files written to {OUTPUT_DIR}")
    elif args.dry_run:
        total = len(by_status.get("dry_run", []))
        total_skip = len(by_status.get("skipped", []))
        print(f"\nDry-run: {total:,} files would be converted, {total_skip:,} already exist.")


if __name__ == "__main__":
    main()
