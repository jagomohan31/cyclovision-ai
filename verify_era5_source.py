"""
Run this against your actual project to get hard evidence of which
weather-data source is really integrated -- settles the ECMWF-vs-NCMRWF
question with a grep, not a guess.

Usage: python verify_era5_source.py
"""
import re
from pathlib import Path

SEARCH_DIRS = ["src", "."]
ECMWF_SIGNALS = [r"\bcdsapi\b", r"cds\.climate\.copernicus\.eu", r"reanalysis-era5"]
NCMRWF_SIGNALS = [r"\bncmrwf\b", r"rds\.ncmrwf\.gov\.in", r"\bimdaa\b", r"\bncum\b"]

def scan():
    ecmwf_hits, ncmrwf_hits = [], []
    seen = set()
    self_path = Path(__file__).resolve()
    for base in SEARCH_DIRS:
        for path in Path(base).rglob("*.py"):
            if path.resolve() == self_path:
                continue  # don't let this script match its own search patterns
            if path in seen or "site-packages" in str(path):
                continue
            seen.add(path)
            try:
                text = path.read_text(errors="ignore")
            except Exception:
                continue
            for pat in ECMWF_SIGNALS:
                if re.search(pat, text, re.IGNORECASE):
                    ecmwf_hits.append((str(path), pat))
            for pat in NCMRWF_SIGNALS:
                if re.search(pat, text, re.IGNORECASE):
                    ncmrwf_hits.append((str(path), pat))
    return ecmwf_hits, ncmrwf_hits

if __name__ == "__main__":
    ecmwf_hits, ncmrwf_hits = scan()
    print("=== ECMWF/Copernicus evidence ===")
    for f, pat in ecmwf_hits:
        print(f"  {f}  (matched: {pat})")
    print(f"\n=== NCMRWF evidence ===")
    for f, pat in ncmrwf_hits:
        print(f"  {f}  (matched: {pat})")

    print(f"\n{'='*50}")
    if ecmwf_hits and not ncmrwf_hits:
        print("VERDICT: ECMWF/Copernicus ERA5 only. Fix the diagram label --")
        print("remove 'NCMRWF' from it, it isn't real in this codebase.")
    elif ncmrwf_hits:
        print("VERDICT: Found real NCMRWF references -- if these are genuine")
        print("working integrations (not just comments/docs mentioning the")
        print("name), the diagram label may be accurate after all. Check")
        print("each file listed above manually before trusting this.")
    else:
        print("VERDICT: Neither found -- check SEARCH_DIRS points at your")
        print("actual source folder.")
