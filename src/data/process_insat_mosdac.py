"""
Turnkey MOSDAC / INSAT-3D / INSAT-3DR Satellite Imagery Ingestion Tool.

Converts raw MOSDAC Level-1B (L1B) or Level-1C (L1C) HDF5/NetCDF files
into the normalized .npy arrays required by CycloVision AI:
    data/raw/insat/<storm_id>/<YYYYMMDDHHMM>.npy

Usage:
    # Single file conversion
    python -m src.data.process_insat_mosdac --input-file path/to/3DIMG_12JUN2023_0300_L1B.h5 --storm-id 2023-003

    # Batch directory conversion
    python -m src.data.process_insat_mosdac --input-dir path/to/mosdac_downloads/ --storm-id 2023-003

    # Generate sample test verification frame
    python -m src.data.process_insat_mosdac --create-sample --storm-id 2023-003 --timestamp 202306120300
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from datetime import datetime
import numpy as np

from src.config import INSAT_DIR, CROP_SIZE


def parse_timestamp_from_filename(filename: str) -> str | None:
    """
    Extract YYYYMMDDHHMM timestamp from standard MOSDAC INSAT filename conventions.
    Examples:
      - 3DIMG_12JUN2023_0300_L1B_STD.h5 -> 202306120300
      - 3RIMG_15MAY2020_0600_L1C.h5     -> 202005150600
      - INSAT3D_TIR1_20230612_0300.nc   -> 202306120300
    """
    # Pattern 1: 12JUN2023_0300
    m = re.search(r"(\d{2})([A-Za-z]{3})(\d{4})_(\d{4})", filename)
    if m:
        day, mon_str, year, time_str = m.groups()
        try:
            dt = datetime.strptime(f"{day}{mon_str.upper()}{year}_{time_str}", "%d%b%Y_%H%M")
            return dt.strftime("%Y%m%d%H%M")
        except ValueError:
            pass

    # Pattern 2: 20230612_0300 or 202306120300
    m = re.search(r"(20\d{2})(\d{2})(\d{2})_?(\d{4})", filename)
    if m:
        year, month, day, time_str = m.groups()
        return f"{year}{month}{day}{time_str}"

    return None


def extract_tir1_array(file_path: Path) -> np.ndarray:
    """
    Read MOSDAC INSAT-3D/3DR HDF5 or NetCDF file and extract TIR-1 (Thermal Infrared) band.
    Returns a 2D float32 array of Brightness Temperature in Kelvin (typically 180K - 320K).
    """
    import netCDF4 as nc

    ds = nc.Dataset(str(file_path), mode="r")
    tir_var_names = ["TIR1_BT", "IMG_TIR1", "TIR1", "IMG_TIR_1", "IMG_TIR1_TEMP", "TIR_TEMP", "temp_tir1"]
    
    selected_var = None
    for name in tir_var_names:
        if name in ds.variables:
            selected_var = name
            break

    if selected_var is None:
        avail = list(ds.variables.keys())
        ds.close()
        raise KeyError(f"Could not find Thermal Infrared variable in {file_path.name}. Available variables: {avail[:10]}")

    raw_data = ds.variables[selected_var][:]
    
    # Handle calibration / scale factor if stored as raw counts
    var_obj = ds.variables[selected_var]
    scale = getattr(var_obj, "scale_factor", 1.0)
    offset = getattr(var_obj, "add_offset", 0.0)

    ds.close()

    # Convert to 2D float32 numpy array
    arr = np.squeeze(np.asarray(raw_data, dtype=np.float32))
    if scale != 1.0 or offset != 0.0:
        arr = arr * scale + offset

    # Sanity guard: Brightness temperature should be in reasonable atmospheric range (Kelvin)
    # If values are in Celsius (-90 to +50), convert to Kelvin (+273.15)
    if np.nanmean(arr) < 100.0:
        arr = arr + 273.15

    return arr


def process_and_save_frame(
    file_path: Path,
    storm_id: str,
    explicit_timestamp: str | None = None,
) -> Path:
    """Extract IR data from MOSDAC file and write to data/raw/insat/<storm_id>/<YYYYMMDDHHMM>.npy"""
    ts = explicit_timestamp or parse_timestamp_from_filename(file_path.name)
    if not ts:
        raise ValueError(f"Could not parse timestamp from {file_path.name}. Specify --timestamp explicitly.")

    tir_arr = extract_tir1_array(file_path)

    out_dir = INSAT_DIR / str(storm_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{ts}.npy"

    np.save(out_file, tir_arr.astype(np.float32))
    print(f"[OK] Saved real INSAT-3D TIR-1 frame -> {out_file} (Shape: {tir_arr.shape}, Range: {np.nanmin(tir_arr):.1f}K - {np.nanmax(tir_arr):.1f}K)")
    return out_file


def create_sample_verification_frame(storm_id: str, timestamp: str = "202306120300") -> Path:
    """Generate a valid test frame into data/raw/insat/<storm_id>/ for verifying pipeline readiness."""
    out_dir = INSAT_DIR / str(storm_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{timestamp}.npy"

    # 256x256 simulated INSAT-3D Brightness Temperature field in Kelvin
    size = CROP_SIZE * 2
    yy, xx = np.mgrid[0:size, 0:size]
    cy, cx = size / 2.0, size / 2.0
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    
    # Cold cloud tops (~200K) at eye wall, warmer ocean background (~295K)
    sample_frame = (295.0 - 95.0 * np.exp(-r / 35.0) + np.random.normal(0, 2.0, size=(size, size))).astype(np.float32)
    np.save(out_file, sample_frame)
    print(f"[SAMPLE CREATED] {out_file} (Shape: {sample_frame.shape})")
    return out_file


def main():
    parser = argparse.ArgumentParser(description="Process MOSDAC INSAT-3D/3DR files for CycloVision AI.")
    parser.add_argument("--input-file", type=str, help="Path to single MOSDAC .h5 or .nc file")
    parser.add_argument("--input-dir", type=str, help="Directory containing downloaded MOSDAC files")
    parser.add_argument("--storm-id", type=str, default="2023-003", help="Target Storm ID (e.g. 2023-003 for Biparjoy)")
    parser.add_argument("--timestamp", type=str, help="Explicit timestamp YYYYMMDDHHMM (optional)")
    parser.add_argument("--create-sample", action="store_true", help="Create a test verification frame")

    args = parser.parse_args()

    if args.create_sample:
        ts = args.timestamp or "202306120300"
        create_sample_verification_frame(args.storm_id, ts)
        return

    if args.input_file:
        process_and_save_frame(Path(args.input_file), args.storm_id, args.timestamp)
    elif args.input_dir:
        p = Path(args.input_dir)
        files = list(p.glob("*.h5")) + list(p.glob("*.nc")) + list(p.glob("*.hdf5"))
        print(f"Found {len(files)} MOSDAC files in {p}")
        for f in files:
            try:
                process_and_save_frame(f, args.storm_id)
            except Exception as e:
                print(f"[WARN] Failed to process {f.name}: {e}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
