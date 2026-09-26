import numpy as np
from PIL import Image
import matplotlib.cm as cm

def check_colormap_vs_photo(arr_rgb):
    # arr_rgb is (H, W, 3)
    if arr_rgb.ndim < 3 or arr_rgb.shape[2] < 3:
        return True, "Monochrome / Single-channel IR"

    r, g, b = arr_rgb[:,:,0].astype(float), arr_rgb[:,:,1].astype(float), arr_rgb[:,:,2].astype(float)
    color_disp = float(np.mean((np.abs(r - g) + np.abs(g - b) + np.abs(b - r)) / 3.0))
    if color_disp < 1.0:
        return True, f"Grayscale / Infrared (dispersion={color_disp:.1f})"

    # For colored images: check if colors lie on a smooth 1D curve (colormap)
    # Subsample pixels
    pixels = arr_rgb[::4, ::4, :].reshape(-1, 3).astype(float)
    
    # Sort pixels by intensity (R + G + B)
    intensity = pixels.sum(axis=1)
    sort_idx = np.argsort(intensity)
    sorted_pixels = pixels[sort_idx]
    
    # Measure path smoothness in 3D color space
    diffs = np.diff(sorted_pixels, axis=0)
    step_lens = np.sqrt((diffs**2).sum(axis=1))
    
    # Total path length vs net distance
    net_dist = np.sqrt(((sorted_pixels[-1] - sorted_pixels[0])**2).sum())
    total_path = step_lens.sum()
    tortuosity = total_path / (net_dist + 1e-6)
    
    # In a 1D colormap, sorted by intensity, the path is relatively clean and low-dimensional.
    # In a natural camera photo, pixels with the same intensity have wildly different colors (e.g. blue jeans vs red shirt vs green board).
    # Group by intensity quantiles and measure color variance within intensity slices:
    n_slices = 10
    slices = np.array_split(sorted_pixels, n_slices)
    intra_slice_var = [float(np.std(s, axis=0).mean()) for s in slices if len(s) > 10]
    mean_intra_var = float(np.mean(intra_slice_var))

    is_photo = (mean_intra_var > 15.0) and (color_disp > 4.0)
    return not is_photo, f"dispersion={color_disp:.1f}, intra_slice_var={mean_intra_var:.1f}"

f_cls = r"C:\Users\Jagomohan Das\Downloads\WhatsApp Image 2026-04-23 at 20.58.00.jpeg"
f_cyc = r"C:\Users\Jagomohan Das\Downloads\202306060015_preview.png"

print("Classroom photo:", check_colormap_vs_photo(np.array(Image.open(f_cls).convert("RGB"))))
print("Cyclone preview:", check_colormap_vs_photo(np.array(Image.open(f_cyc).convert("RGB"))))

# Test on synthetic colormaps on real INSAT
arr_bip = np.load("data/raw/insat/2023-003/202306110002.npy")
norm_bip = (arr_bip - arr_bip.min()) / (arr_bip.max() - arr_bip.min())
for cname in ["magma", "inferno", "viridis", "jet", "coolwarm"]:
    cmap_img = (getattr(cm, cname)(norm_bip)[:, :, :3] * 255).astype(np.uint8)
    print(f"INSAT {cname:8s}:", check_colormap_vs_photo(cmap_img))
