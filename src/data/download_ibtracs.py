"""
Refresh the best-track data, or pull the fuller global IBTrACS record if
you want storms outside the North Indian Ocean for pretraining/comparison.

This repo already ships real, current data (see data/raw/ibtracs/SOURCE.md
— 425 IMD-tracked storms, 1982-2026) so you do NOT need to run this before
you start; it's here for when you want to:
  (a) pull the latest storms IMD has added since this repo's snapshot, or
  (b) get the full global IBTrACS (all basins) for pretraining the image
      backbone on a larger, more varied set of cyclone imagery before
      fine-tuning on the North Indian Ocean.

Run on your own machine (needs internet access this sandbox doesn't have):
    python src/data/download_ibtracs.py --basin NI
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import IBTRACS_DIR

IBTRACS_BASE_URL = (
    "https://www.ncei.noaa.gov/data/"
    "international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r01/access/csv/"
)

# Basin codes: NA=North Atlantic, EP=East Pacific, WP=West Pacific,
# NI=North Indian (Arabian Sea + Bay of Bengal — the one this project cares
# about), SI=South Indian, SP=South Pacific, SA=South Atlantic, ALL=global.
VALID_BASINS = ["NA", "EP", "WP", "NI", "SI", "SP", "SA", "ALL"]


def download_basin(basin: str, out_dir: Path = IBTRACS_DIR) -> Path:
    import requests  # local import: this script's only purpose needs it

    assert basin in VALID_BASINS, f"basin must be one of {VALID_BASINS}"
    url = f"{IBTRACS_BASE_URL}ibtracs.{basin}.list.v04r01.csv"
    out_path = out_dir / f"ibtracs.{basin}.list.v04r01.csv"

    print(f"Downloading {url} ...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)
    print(f"Saved -> {out_path}  ({len(resp.content):,} bytes)")
    print("\nNote: IBTrACS's own NI-basin file merges multiple agencies' "
          "estimates and uses knots + slightly different category cutoffs "
          "than IMD's own record. For IMD-calibrated category labels "
          "(what this project trains against), prefer "
          "data/raw/ibtracs/imd_besttrack_observations.csv — use this "
          "global file for extra pretraining variety instead.")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--basin", default="NI", choices=VALID_BASINS,
                         help="NI = North Indian Ocean (Arabian Sea + Bay of Bengal). "
                              "Use ALL for the full global record.")
    args = parser.parse_args()
    download_basin(args.basin)
