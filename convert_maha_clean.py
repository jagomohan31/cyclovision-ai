import h5py, numpy as np, pandas as pd
from pathlib import Path

PROJECT    = Path(r"D:/Hackathon Cyclovision AI/cyclovision-ai")
MOSDAC     = PROJECT / "MOSDAC_DATA" / "MAHA"
OUT_DIR    = PROJECT / "data/raw/insat/2019-008"
OBS_CSV    = PROJECT / "data/raw/ibtracs/imd_besttrack_observations.csv"
STORMS_CSV = PROJECT / "data/raw/ibtracs/imd_besttrack_storms.csv"
CROP_SIZE  = 128
STORM_ID   = "2019-008"

storms = pd.read_csv(STORMS_CSV, parse_dates=["start_time","end_time"])
obs    = pd.read_csv(OBS_CSV, parse_dates=["time"])
obs    = obs[obs["storm_id"]==STORM_ID].dropna(subset=["lat","lon"]).sort_values("time")

row   = storms[storms["storm_id"]==STORM_ID].iloc[0]
pad   = pd.Timedelta(hours=6)
WIN_S = row["start_time"] - pad
WIN_E = row["end_time"]   + pd.Timedelta(hours=12)

OUT_DIR.mkdir(parents=True, exist_ok=True)

def latlon_to_xy(lat, lon, a, b, lat_std, lon0):
    lat_r=np.radians(lat); lon_r=np.radians(lon)
    ls=np.radians(lat_std); l0=np.radians(lon0)
    e2=1-(b**2/a**2); e=np.sqrt(e2)
    k0=np.cos(ls)/np.sqrt(1-e2*np.sin(ls)**2)
    x=a*k0*(lon_r-l0)
    y=a*k0*np.log(np.tan(np.pi/4+lat_r/2)*((1-e*np.sin(lat_r))/(1+e*np.sin(lat_r)))**(e/2))
    return x, y

def interp_pos(ts):
    if ts <= obs["time"].iloc[0]:  return obs.iloc[0]["lat"], obs.iloc[0]["lon"]
    if ts >= obs["time"].iloc[-1]: return obs.iloc[-1]["lat"], obs.iloc[-1]["lon"]
    idx  = int(obs["time"].searchsorted(ts))
    t0,t1 = obs["time"].iloc[idx-1], obs["time"].iloc[idx]
    frac  = (ts-t0)/(t1-t0)
    lat = obs["lat"].iloc[idx-1] + frac*(obs["lat"].iloc[idx]-obs["lat"].iloc[idx-1])
    lon = obs["lon"].iloc[idx-1] + frac*(obs["lon"].iloc[idx]-obs["lon"].iloc[idx-1])
    return lat, lon

files = sorted(MOSDAC.glob("*.h5"))
converted=0; skipped=0; outside=0
total = len(files)
print(f"Starting Maha conversion: {total} H5 files found")

for i, f in enumerate(files):
    with h5py.File(f, "r") as hf:
        d = hf.attrs["Acquisition_Date"]
        t = hf.attrs["Acquisition_Time_in_GMT"]
        if isinstance(d, bytes): d = d.decode()
        if isinstance(t, bytes): t = t.decode()
        ts = pd.to_datetime(f"{d} {t}", format="%d%b%Y %H%M")
        if not (WIN_S <= ts <= WIN_E):
            outside += 1
            continue
        out_path = OUT_DIR / f"{ts:%Y%m%d%H%M}.npy"
        if out_path.exists():
            skipped += 1
            continue
        bt   = hf["TIR1_BT"][0]
        fill = hf["TIR1_BT"].attrs.get("_FillValue", -999.0)
        bt   = np.where(bt==fill, np.nan, bt)
        proj = hf["Projection_Information"]
        a    = proj.attrs["semi_major_axis"][0]
        b    = proj.attrs["semi_minor_axis"][0]
        lat_std = proj.attrs["standard_parallel"][0]
        lon0    = proj.attrs["longitude_of_projection_origin"][0]
        X, Y    = hf["X"][:], hf["Y"][:]
    lat, lon = interp_pos(ts)
    sx, sy   = latlon_to_xy(lat, lon, a, b, lat_std, lon0)
    col   = int(np.argmin(np.abs(X-sx)))
    row_i = int(np.argmin(np.abs(Y-sy)))
    half  = CROP_SIZE//2; p = half+1
    padded = np.pad(bt, p, mode="edge")
    r, c   = row_i+p, col+p
    crop   = padded[r-half:r+half, c-half:c+half]
    np.save(out_path, crop.astype(np.float32))
    converted += 1
    if converted % 200 == 0:
        pct = 100*(i+1)/total
        print(f"  [{pct:.0f}%] {converted} converted, {outside} outside window...")

npy_count = len(list(OUT_DIR.glob("*.npy")))
print("")
print("MAHA DONE")
print(f"  Converted : {converted}")
print(f"  Skipped   : {skipped}")
print(f"  Outside   : {outside}")
print(f"  Total npy : {npy_count}")