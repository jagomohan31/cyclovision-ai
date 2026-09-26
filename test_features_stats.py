import numpy as np
from PIL import Image
from pathlib import Path
import glob

# Collect 20 real satellite frames from different storms
sat_samples = []
for sdir in sorted(Path("data/raw/insat").iterdir()):
    if sdir.is_dir() and sdir.name != "quarantine":
        files = list(sdir.glob("*.npy"))
        if files:
            sat_samples.extend(files[:5])

print(f"Collected {len(sat_samples)} real satellite .npy samples")

# Also preview png
f_preview = r"C:\Users\Jagomohan Das\Downloads\202306060015_preview.png"
# Classroom
f_cls = r"C:\Users\Jagomohan Das\Downloads\WhatsApp Image 2026-04-23 at 20.58.00.jpeg"

# Let us analyze spatial gradients, edge statistics, and texture
import cv2

def extract_features(img_gray):
    # Standardize to 128x128
    img = cv2.resize(img_gray, (128, 128)).astype(np.float32)
    # Normalize to [0, 1]
    i_min, i_max = img.min(), img.max()
    if i_max - i_min > 1e-5:
        norm = (img - i_min) / (i_max - i_min)
    else:
        norm = np.zeros_like(img)

    # 1. Gradient statistics
    gy, gx = np.gradient(norm)
    mag = np.sqrt(gx**2 + gy**2)
    
    # 2. Laplacian (edge sharpness / high frequency detail)
    lap = cv2.Laplacian(norm, cv2.CV_32F)
    lap_var = float(lap.var())
    
    # 3. Local Binary Pattern / Texture contrast
    # Satellite clouds have smooth diffuse transitions; camera photos of rooms have sharp edges
    # Check ratio of sharp gradient pixels
    sharp_edge_ratio = float((mag > 0.15).mean())
    smooth_ratio = float((mag < 0.03).mean())
    
    # 4. Straight line energy (Hough lines)
    edges = cv2.Canny((norm * 255).astype(np.uint8), 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=25, maxLineGap=5)
    n_lines = len(lines) if lines is not None else 0

    return {
        "lap_var": lap_var,
        "grad_mean": float(mag.mean()),
        "sharp_ratio": sharp_edge_ratio,
        "smooth_ratio": smooth_ratio,
        "n_lines": n_lines,
    }

print("=== SATELLITE SAMPLES ===")
sat_feats = []
for p in sat_samples[:10]:
    arr = np.load(p)
    arr_norm = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-6) * 255).astype(np.uint8)
    f = extract_features(arr_norm)
    sat_feats.append(f)
    print(f"  {p.parent.name}/{p.stem}: {f}")

# Preview png (with border stripped)
img_p = np.array(Image.open(f_preview).convert("L"))
# strip border
mask = (img_p > 15) & (img_p < 240)
rmin, rmax = np.where(np.any(mask, axis=1))[0][[0, -1]]
cmin, cmax = np.where(np.any(mask, axis=0))[0][[0, -1]]
f_prev = extract_features(img_p[rmin:rmax+1, cmin:cmax+1])
print(f"  Preview PNG (stripped): {f_prev}")

print("=== NON-SATELLITE SAMPLES ===")
# Classroom
img_c = np.array(Image.open(f_cls).convert("L"))
f_c = extract_features(img_c)
print(f"  Classroom: {f_c}")

# QR code
rng = np.random.default_rng(42)
qr = rng.choice([0, 255], size=(128, 128), p=[0.5, 0.5]).astype(np.uint8)
f_qr = extract_features(qr)
print(f"  QR Code: {f_qr}")
