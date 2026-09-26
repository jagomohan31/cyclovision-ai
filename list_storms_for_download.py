import pandas as pd
from pathlib import Path

storms = pd.read_csv(
    r"data\raw\ibtracs\imd_besttrack_storms.csv",
    parse_dates=["start_time", "end_time"],
)
obs = pd.read_csv(
    r"data\raw\ibtracs\imd_besttrack_observations.csv",
    parse_dates=["time"],
)

insat_dir = Path(r"data\raw\insat")
npy_counts = {}
for p in insat_dir.iterdir():
    if p.is_dir() and p.name != "quarantine":
        npy_counts[p.name] = len(list(p.glob("*.npy")))

obs_counts = obs.groupby("storm_id").size()
grade_dist = obs.groupby(["storm_id", "grade"]).size().unstack(fill_value=0)
grade_rank = {"D": 1, "DD": 2, "CS": 3, "SCS": 4, "VSCS": 5, "ESCS": 6, "SuCS": 7}

# INSAT-3D launched Oct 2013 -- no useful TIR data before 2014
recent = storms[storms["start_time"].dt.year >= 2014].sort_values("start_time")

print("Storms 2014+ grouped by download priority")
print("(MOSDAC has INSAT-3D/3DR data from Oct 2013 onwards)")
print()
print("ALREADY HAVE (>=200 npy frames):")
print("-" * 72)
for _, row in recent.iterrows():
    sid = row["storm_id"]
    name = str(row.get("name", "?"))
    start = row["start_time"].strftime("%Y-%m-%d")
    end = row["end_time"].strftime("%Y-%m-%d")
    n_npy = npy_counts.get(sid, 0)
    if n_npy < 200:
        continue
    if sid in grade_dist.index:
        g = grade_dist.loc[sid]
        present = [c for c in g.index if g[c] > 0]
        ranked = sorted(present, key=lambda x: grade_rank.get(x, 0), reverse=True)
        peak = ranked[0] if ranked else "?"
        grades = " ".join(present)
    else:
        peak = "?"
        grades = "-"
    print(f"  {sid}  {name:<16} {start} -> {end}  npy={n_npy:>4}  peak={peak}  grades=[{grades}]")

print()
print("MISSING / INCOMPLETE (<200 npy) -- RECOMMEND DOWNLOADING:")
print("-" * 72)
for _, row in recent.iterrows():
    sid = row["storm_id"]
    name = str(row.get("name", "?"))
    start = row["start_time"].strftime("%Y-%m-%d")
    end = row["end_time"].strftime("%Y-%m-%d")
    n_npy = npy_counts.get(sid, 0)
    if n_npy >= 200:
        continue
    n_obs = obs_counts.get(sid, 0)
    if n_obs < 5:  # skip tiny storms
        continue
    if sid in grade_dist.index:
        g = grade_dist.loc[sid]
        present = [c for c in g.index if g[c] > 0]
        ranked = sorted(present, key=lambda x: grade_rank.get(x, 0), reverse=True)
        peak = ranked[0] if ranked else "?"
        grades = " ".join(present)
    else:
        peak = "?"
        grades = "-"
    print(f"  {sid}  {name:<16} {start} -> {end}  npy={n_npy:>4}  obs={n_obs:>3}  peak={peak}  grades=[{grades}]")

print()
print("MOSDAC download URL: https://mosdac.gov.in/live/index.php")
print("Product: INSAT-3DR  |  Channel: TIR1 (10.8 um)  |  Format: HDF5")
print("Download 1 file every ~30 min over the storm window to match training data density.")
