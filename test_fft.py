import numpy as np
from PIL import Image

def compute_fft_isotropy(img_gray):
    # Standardize to 128x128
    img = np.array(Image.fromarray(img_gray).resize((128, 128), Image.Resampling.BILINEAR), dtype=np.float32)
    # 2D FFT
    F = np.fft.fftshift(np.fft.fft2(img - img.mean()))
    power = np.abs(F)**2
    
    H, W = power.shape
    yy, xx = np.mgrid[-H//2:H//2, -W//2:W//2]
    r = np.sqrt(xx**2 + yy**2)
    angles = np.abs(np.arctan2(yy, xx)) # [0, pi]
    
    # Exclude DC component and extreme high freq noise
    mask = (r >= 4) & (r <= H // 2 - 2)
    
    # Power along cardinal axes (within 10 degrees of 0, pi/2, pi)
    deg = np.degrees(angles)
    is_cardinal = (deg < 10) | (deg > 170) | ((deg > 80) & (deg < 100))
    
    cardinal_power = power[mask & is_cardinal].sum()
    diagonal_power = power[mask & ~is_cardinal].sum()
    cardinal_area = (mask & is_cardinal).sum()
    diagonal_area = (mask & ~is_cardinal).sum()
    
    cardinal_density = cardinal_power / (cardinal_area + 1e-6)
    diagonal_density = diagonal_power / (diagonal_area + 1e-6)
    
    anisotropy_ratio = cardinal_density / (diagonal_density + 1e-6)
    return float(anisotropy_ratio)

f_cls = r"C:\Users\Jagomohan Das\Downloads\WhatsApp Image 2026-04-23 at 20.58.00.jpeg"
f_cyc = r"C:\Users\Jagomohan Das\Downloads\202306060015_preview.png"

gray_cls = np.array(Image.open(f_cls).convert("L"))
gray_cyc = np.array(Image.open(f_cyc).convert("L"))

# Strip border from cyclone
mask = (gray_cyc > 15) & (gray_cyc < 240)
rmin, rmax = np.where(np.any(mask, axis=1))[0][[0, -1]]
cmin, cmax = np.where(np.any(mask, axis=0))[0][[0, -1]]
gray_cyc = gray_cyc[rmin:rmax+1, cmin:cmax+1]

print("Classroom FFT Anisotropy Ratio:", compute_fft_isotropy(gray_cls))
print("Cyclone Preview FFT Anisotropy Ratio:", compute_fft_isotropy(gray_cyc))

from pathlib import Path
for f in sorted(Path("data/raw/insat/2023-003").glob("*.npy"))[:5]:
    arr = np.load(f)
    print(f"INSAT {f.stem}:", compute_fft_isotropy(arr))
