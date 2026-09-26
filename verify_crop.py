"""Quick visual sanity check: plot one converted crop so you can eyeball
whether it actually looks like cyclone-relevant cloud structure, not noise
or an empty/misregistered patch."""
import sys
import numpy as np
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/insat/2023-003/202306060015.npy"
arr = np.load(path)

fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(arr, cmap='gray_r')
plt.colorbar(im, label='Brightness Temperature (K)')
ax.set_title(f"{path}\nrange: {np.nanmin(arr):.1f}K - {np.nanmax(arr):.1f}K")
out_png = path.replace('.npy', '_preview.png')
plt.savefig(out_png, dpi=110)
print(f"Saved preview -> {out_png}")
print(f"Min/max temp: {np.nanmin(arr):.1f}K / {np.nanmax(arr):.1f}K  (NaN count: {np.isnan(arr).sum()})")