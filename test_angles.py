import numpy as np
from PIL import Image
from pathlib import Path

def compute_vortex_metrics(img_gray):
    # Standardize to 128x128
    img = np.array(Image.fromarray(img_gray).resize((128, 128), Image.Resampling.BILINEAR), dtype=np.float32)
    gy, gx = np.gradient(img)
    mag = np.sqrt(gx**2 + gy**2)
    # Ignore low-gradient pixels
    threshold = np.percentile(mag, 50)
    significant = mag > threshold
    angles = np.arctan2(gy[significant], gx[significant]) # [-pi, pi]

    # Histogram of edge orientations (in 8 bins)
    hist, _ = np.histogram(np.abs(angles), bins=8, range=(0, np.pi))
    p = hist / (hist.sum() + 1e-9)
    # Entropy of gradient angles: uniform circular distribution has high entropy (~3.0)
    angle_entropy = -np.sum(p * np.log2(p + 1e-9))

    # Orthogonality ratio: fraction of edges within 10 degrees of horizontal (0, pi) or vertical (pi/2)
    deg = np.abs(np.degrees(angles))
    is_hv = (deg < 12) | (deg > 168) | ((deg > 78) & (deg < 102))
    hv_ratio = float(is_hv.mean())

    return angle_entropy, hv_ratio

f_cls = r"C:\Users\Jagomohan Das\Downloads\WhatsApp Image 2026-04-23 at 20.58.00.jpeg"
f_cyc = r"C:\Users\Jagomohan Das\Downloads\202306060015_preview.png"

# Strip borders from cyclone preview
gray_cyc = np.array(Image.open(f_cyc).convert("L"))
mask = (gray_cyc > 15) & (gray_cyc < 240)
rmin, rmax = np.where(np.any(mask, axis=1))[0][[0, -1]]
cmin, cmax = np.where(np.any(mask, axis=0))[0][[0, -1]]
gray_cyc_content = gray_cyc[rmin:rmax+1, cmin:cmax+1]

gray_cls = np.array(Image.open(f_cls).convert("L"))

ent_cls, hv_cls = compute_vortex_metrics(gray_cls)
ent_cyc, hv_cyc = compute_vortex_metrics(gray_cyc_content)

print(f"Classroom (Grayscale): Angle Entropy = {ent_cls:.3f}, H/V Line Ratio = {hv_cls*100:.1f}%")
print(f"Cyclone Preview:       Angle Entropy = {ent_cyc:.3f}, H/V Line Ratio = {hv_cyc*100:.1f}%")

# Real INSAT files
for f in sorted(Path("data/raw/insat/2023-003").glob("*.npy"))[:5]:
    arr = np.load(f)
    norm = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-6) * 255).astype(np.uint8)
    ent, hv = compute_vortex_metrics(norm)
    print(f"INSAT {f.stem}:  Angle Entropy = {ent:.3f}, H/V Line Ratio = {hv*100:.1f}%")
