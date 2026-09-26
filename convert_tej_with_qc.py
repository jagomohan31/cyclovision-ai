import time
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from convert_insat_batch import parse_acquisition_time, latlon_to_mercator_xy, interpolate_position, CROP_SIZE
from src.data.load_besttrack import load_observations

obs = load_observations()
storm_obs = obs[obs['storm_id'] == '2023-006'].sort_values('time')
h5_files = sorted(Path('MOSDAC_DATA/Tej').glob('*.h5'))
out_dir = Path('data/raw/insat/2023-006')
out_dir.mkdir(parents=True, exist_ok=True)
quarantine_dir = Path('data/raw/insat/quarantine/2023-006')
quarantine_dir.mkdir(parents=True, exist_ok=True)

print(f"Starting conversion of {len(h5_files)} H5 files for Cyclone Tej (2023-006)...")
t0 = time.time()
converted, skipped_done, corrupt_count, out_of_window = 0, 0, 0, 0

bt_start = storm_obs['time'].iloc[0]
bt_end = storm_obs['time'].iloc[-1]

for i, h5_path in enumerate(h5_files):
    try:
        with h5py.File(h5_path, 'r') as f:
            ts = parse_acquisition_time(f)
            
            # Check best-track window (+- 6 hours tolerance)
            if ts < bt_start - pd.Timedelta(hours=6) or ts > bt_end + pd.Timedelta(hours=6):
                out_of_window += 1
                continue
                
            out_file = out_dir / f"{ts:%Y%m%d%H%M}.npy"
            if out_file.exists():
                skipped_done += 1
                continue
                
            bt = f['TIR1_BT'][0]
            fill_value = f['TIR1_BT'].attrs.get('_FillValue', -999.0)
            bt = np.where(bt == fill_value, np.nan, bt)

            proj = f['Projection_Information']
            a, b = proj.attrs['semi_major_axis'][0], proj.attrs['semi_minor_axis'][0]
            lat_std, lon0 = proj.attrs['standard_parallel'][0], proj.attrs['longitude_of_projection_origin'][0]
            X, Y = f['X'][:], f['Y'][:]

        lat, lon = interpolate_position(storm_obs, ts)
        storm_x, storm_y = latlon_to_mercator_xy(lat, lon, a, b, lat_std, lon0)
        col = int(np.argmin(np.abs(X - storm_x)))
        row = int(np.argmin(np.abs(Y - storm_y)))

        half = CROP_SIZE // 2
        pad_amt = half + 1
        padded = np.pad(bt, pad_amt, mode='edge')
        r, c = row + pad_amt, col + pad_amt
        crop = padded[r - half : r + half, c - half : c + half].astype(np.float32)
        crop = np.nan_to_num(crop, nan=270.0)

        # Quality Control: check if crop is corrupt / partial scan
        is_corrupt = (crop[:25, :].std() < 0.2) or (crop[-25:, :].std() < 0.2) or ((crop == crop[0, 0]).mean() > 0.25)
        if is_corrupt:
            q_file = quarantine_dir / f"{ts:%Y%m%d%H%M}.npy"
            np.save(q_file, crop)
            corrupt_count += 1
        else:
            np.save(out_file, crop)
            converted += 1

        if (i + 1) % 200 == 0:
            print(f"  ... {i+1}/{len(h5_files)} processed in {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"Error processing {h5_path.name}: {e}")

all_clean = list(out_dir.glob("*.npy"))
print(f"\nFINISHED in {time.time()-t0:.1f}s:")
print(f"  Newly converted clean frames: {converted}")
print(f"  Already existing clean frames: {skipped_done}")
print(f"  Corrupt / partial scan quarantined: {corrupt_count}")
print(f"  Out of best-track window: {out_of_window}")
print(f"  Total clean frames now in 2023-006: {len(all_clean)}")
