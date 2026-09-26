import numpy as np
from PIL import Image
from pathlib import Path

def inspect_image_features(path_or_arr, name):
    if isinstance(path_or_arr, (str, Path)):
        img = Image.open(path_or_arr)
        arr_rgb = np.array(img.convert("RGB"), dtype=np.float32)
    else:
        arr = path_or_arr
        if arr.ndim == 2:
            arr_rgb = np.stack([arr, arr, arr], axis=-1)
        else:
            arr_rgb = arr.astype(np.float32)

    # 1. Strip uniform borders (e.g. white or black margin)
    gray = arr_rgb.mean(axis=2)
    h, w = gray.shape
    mask = (gray > 15) & (gray < 240)
    if np.any(mask):
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        if (rmax - rmin > 40) and (cmax - cmin > 40):
            arr_rgb = arr_rgb[rmin:rmax+1, cmin:cmax+1]
            gray = gray[rmin:rmax+1, cmin:cmax+1]

    # Feature A: Chromatic dispersion (RGB channel spread)
    # For grayscale / single-channel IR: R == G == B, chromatic_disp == 0.0
    # For pseudo-colormap: channels follow a smooth 1D curve
    # For natural camera photo: R, G, B vary independently across scene
    r, g, b = arr_rgb[:,:,0], arr_rgb[:,:,1], arr_rgb[:,:,2]
    chroma_diff = np.sqrt((r-g)**2 + (g-b)**2 + (b-r)**2)
    mean_chroma = float(chroma_diff.mean())
    std_chroma = float(chroma_diff.std())

    # Feature B: Pure Binary Ratio (QR codes / text)
    # Exactly pure black or pure white
    is_binary = ((gray < 5) | (gray > 250)).mean()

    # Feature C: Discrete levels
    n_unique = len(np.unique(np.round(gray)))

    # Feature D: Atmospheric gradient smoothness
    gy, gx = np.gradient(gray / 255.0)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_mean = float(grad_mag.mean())
    grad_max = float(grad_mag.max())

    print(f"{name:30s} | chroma={mean_chroma:5.1f} (std={std_chroma:4.1f}) | binary={is_binary*100:4.1f}% | unique={n_unique:3d} | grad_mean={grad_mean:5.3f}")

print("Testing various images:")
f_cls = r"C:\Users\Jagomohan Das\Downloads\WhatsApp Image 2026-04-23 at 20.58.00.jpeg"
f_cyc = r"C:\Users\Jagomohan Das\Downloads\202306060015_preview.png"

inspect_image_features(f_cls, "Classroom Photo (WhatsApp)")
inspect_image_features(f_cyc, "Cyclone Preview (Real)")

# QR Code
rng = np.random.default_rng(42)
qr = rng.choice([0, 255], size=(200, 200), p=[0.5, 0.5]).astype(np.uint8)
inspect_image_features(qr, "Synthetic QR Code")

# Real INSAT files
for f in sorted(Path("data/raw/insat/2023-003").glob("*.npy"))[:5]:
    arr = np.load(f)
    norm = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-6) * 255).astype(np.uint8)
    inspect_image_features(norm, f"INSAT {f.stem}")
