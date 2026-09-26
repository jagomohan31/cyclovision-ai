"""
Inspect ONE INSAT-3DR L1C HDF5 file to see its real internal structure —
dataset names, shapes, and attributes — before writing the actual
HDF5 -> .npy converter. Different INSAT products/versions sometimes use
different internal naming, so this checks ground truth first instead of
guessing and burning a run across all 2,818 files on a wrong assumption.

Usage:
    pip install h5py    (if not already installed)
    python inspect_insat_file.py "path\\to\\one\\file.h5"
"""
import sys
import h5py


def describe(name, obj):
    indent = "  " * name.count("/")
    if isinstance(obj, h5py.Dataset):
        print(f"{indent}[Dataset] {name}   shape={obj.shape}  dtype={obj.dtype}")
    else:
        print(f"{indent}[Group]   {name}")
    for key, val in obj.attrs.items():
        val_str = str(val)
        if len(val_str) > 100:
            val_str = val_str[:100] + "..."
        print(f"{indent}    attr: {key} = {val_str}")


def main(path: str):
    with h5py.File(path, "r") as f:
        print("=== Top-level (file) attributes ===")
        for key, val in f.attrs.items():
            val_str = str(val)
            if len(val_str) > 150:
                val_str = val_str[:150] + "..."
            print(f"  {key} = {val_str}")

        print("\n=== Full structure (groups + datasets) ===")
        f.visititems(describe)

        print("\n=== Looking for likely candidates ===")
        candidates = []
        f.visititems(lambda name, obj: candidates.append(name) if isinstance(obj, h5py.Dataset) else None)
        tir_like = [c for c in candidates if "TIR" in c.upper()]
        geo_like = [c for c in candidates if any(k in c.upper() for k in ["LAT", "LON"])]
        print("TIR-like datasets (your infrared channel):", tir_like or "none found — check full structure above")
        print("Lat/Lon-like datasets (for geolocation):", geo_like or "none found — check full structure above")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_insat_file.py \"path\\to\\one\\file.h5\"")
        sys.exit(1)
    main(sys.argv[1])
