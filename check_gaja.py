import h5py
import pandas as pd
from pathlib import Path

storms = pd.read_csv(r"data\raw\ibtracs\imd_besttrack_storms.csv", parse_dates=["start_time","end_time"])
PAD = pd.Timedelta(hours=6)

gaja_folder = Path(r"MOSDAC_DATA\GAJA")
files = sorted(gaja_folder.glob("*.h5"))
print(f"Total GAJA H5 files: {len(files)}")

# Sample 5 files and check what storm they match
for f in files[:5]:
    with h5py.File(f, "r") as hf:
        date_str = hf.attrs["Acquisition_Date"]
        time_str = hf.attrs["Acquisition_Time_in_GMT"]
        if isinstance(date_str, bytes): date_str = date_str.decode()
        if isinstance(time_str, bytes): time_str = time_str.decode()
        ts = pd.to_datetime(f"{date_str} {time_str}", format="%d%b%Y %H%M")

    matches = storms[(storms["start_time"] - PAD <= ts) & (ts <= storms["end_time"] + PAD)]
    print(f"  {f.name[:30]} -> ts={ts}  matches={matches['storm_id'].tolist()}")
