import pandas as pd
from pathlib import Path

storms = pd.read_csv(r"data\raw\ibtracs\imd_besttrack_storms.csv", parse_dates=["start_time","end_time"])
obs = pd.read_csv(r"data\raw\ibtracs\imd_besttrack_observations.csv", parse_dates=["time"])

insat_dir = Path(r"data\raw\insat")
npy_counts = {}
for p in insat_dir.iterdir():
    if p.is_dir() and p.name != "quarantine":
        npy_counts[p.name] = len(list(p.glob("*.npy")))

grade_dist = obs.groupby(["storm_id","grade"]).size().unstack(fill_value=0)
grade_rank = {"D":1,"DD":2,"CS":3,"SCS":4,"VSCS":5,"ESCS":6,"SuCS":7}

mosdac_start = pd.Timestamp("2016-10-03")
mosdac_end   = pd.Timestamp("2026-09-24")

window = storms[
    (storms["start_time"] >= mosdac_start) &
    (storms["end_time"]   <= mosdac_end)
].sort_values("start_time")

print("Storms within MOSDAC window (Oct 2016 - Sep 2026) with NO npy yet:")
print()
rows = []
for _, row in window.iterrows():
    sid = row["storm_id"]
    n_npy = npy_counts.get(sid, 0)
    if n_npy >= 200:
        continue
    n_obs = int(obs[obs["storm_id"]==sid].shape[0])
    if n_obs < 10:
        continue
    if sid not in grade_dist.index:
        continue
    g = grade_dist.loc[sid]
    present = [c for c in g.index if g[c] > 0]
    ranked = sorted(present, key=lambda x: grade_rank.get(x,0), reverse=True)
    peak = ranked[0] if ranked else "?"
    peak_rank = grade_rank.get(peak, 0)
    duration = (row["end_time"] - row["start_time"]).days
    name = str(row.get("name","?"))
    start = row["start_time"].strftime("%Y-%m-%d")
    end   = row["end_time"].strftime("%Y-%m-%d")
    grades_str = " ".join(present)
    rows.append((peak_rank, duration, sid, name, start, end, peak, grades_str))

# Sort by peak intensity desc, then duration desc
rows.sort(key=lambda x: (x[0], x[1]), reverse=True)

for peak_rank, duration, sid, name, start, end, peak, grades_str in rows:
    print(f"{sid}  {name:<14} {start} -> {end}  {duration}d  peak={peak}  grades=[{grades_str}]")
