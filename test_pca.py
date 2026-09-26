import numpy as np
from PIL import Image
import matplotlib.cm as cm

def test_color_intrinsic_dim(img_rgb):
    # Flatten pixels to (N, 3)
    pixels = img_rgb.reshape(-1, 3).astype(np.float32)
    # Center
    pixels_c = pixels - pixels.mean(axis=0)
    # Covariance matrix (3x3)
    cov = np.cov(pixels_c, rowvar=False)
    # Eigenvalues
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    eigvals = np.maximum(eigvals, 0)
    total_var = eigvals.sum() + 1e-9
    explained = eigvals / total_var
    # PC2 + PC3 ratio
    non_1d_ratio = float((eigvals[1] + eigvals[2]) / total_var)
    return explained, non_1d_ratio

f_cls = r"C:\Users\Jagomohan Das\Downloads\WhatsApp Image 2026-04-23 at 20.58.00.jpeg"
f_cyc = r"C:\Users\Jagomohan Das\Downloads\202306060015_preview.png"

img_cls = np.array(Image.open(f_cls).convert("RGB"))
img_cyc = np.array(Image.open(f_cyc).convert("RGB"))

exp_cls, non1d_cls = test_color_intrinsic_dim(img_cls)
exp_cyc, non1d_cyc = test_color_intrinsic_dim(img_cyc)

print(f"Classroom Photo: Explained Var = {exp_cls.round(3)} | Non-1D Color Ratio = {non1d_cls*100:.2f}%")
print(f"Cyclone Preview: Explained Var = {exp_cyc.round(3)} | Non-1D Color Ratio = {non1d_cyc*100:.2f}%")

# Test synthetic colormaps on real INSAT
arr_bip = np.load("data/raw/insat/2023-003/202306110002.npy")
norm_bip = (arr_bip - arr_bip.min()) / (arr_bip.max() - arr_bip.min())
for cname in ["magma", "inferno", "viridis", "jet"]:
    cmap_img = (getattr(cm, cname)(norm_bip)[:, :, :3] * 255).astype(np.uint8)
    exp, non1d = test_color_intrinsic_dim(cmap_img)
    print(f"INSAT {cname:7s}: Explained Var = {exp.round(3)} | Non-1D Color Ratio = {non1d*100:.2f}%")
