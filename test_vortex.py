import numpy as np
from PIL import Image

def compute_radial_vortex_score(img_gray, center_r, center_c, radius_inner=10, radius_outer=50):
    """
    Measures the radial/rotational gradient alignment around (center_r, center_c).
    For a tropical cyclone vortex, cloud bands wrap around the core, so the gradient
    vectors exhibit strong radial / tangential coherence.
    In a non-cyclone scene (e.g. classroom, room, document), gradients are dominated
    by linear horizontal/vertical features and have near-zero vortex coherence.
    """
    img = img_gray.astype(np.float32)
    gy, gx = np.gradient(img)
    mag = np.sqrt(gx**2 + gy**2)
    
    H, W = img.shape
    yy, xx = np.mgrid[0:H, 0:W]
    dy = yy - center_r
    dx = xx - center_c
    dist = np.sqrt(dx**2 + dy**2)
    
    # Ring mask around center
    ring = (dist >= radius_inner) & (dist <= radius_outer) & (mag > np.percentile(mag, 30))
    if not np.any(ring):
        return 0.0, 0.0
        
    rx = dx[ring] / dist[ring]
    ry = dy[ring] / dist[ring]
    
    g_norm_x = gx[ring] / (mag[ring] + 1e-6)
    g_norm_y = gy[ring] / (mag[ring] + 1e-6)
    
    # Radial component: dot product with radial unit vector
    radial_proj = np.abs(rx * g_norm_x + ry * g_norm_y)
    # Tangential component: dot product with tangential unit vector
    tangential_proj = np.abs(-ry * g_norm_x + rx * g_norm_y)
    
    # Circular coherence: how well gradients align with either radial or tangential contours
    coherence = np.mean(np.maximum(radial_proj, tangential_proj))
    
    # Angular uniformity: do gradients point in all quadrants around the eye?
    angles = np.arctan2(dy[ring], dx[ring])
    quadrants, _ = np.histogram(angles, bins=8, range=(-np.pi, np.pi))
    quad_uniformity = np.min(quadrants) / (np.max(quadrants) + 1e-6)
    
    return float(coherence), float(quad_uniformity)

f_cls = r"C:\Users\Jagomohan Das\Downloads\WhatsApp Image 2026-04-23 at 20.58.00.jpeg"
f_cyc = r"C:\Users\Jagomohan Das\Downloads\202306060015_preview.png"

# Load & resize to 128x128
gray_cls = np.array(Image.open(f_cls).convert("L").resize((128, 128), Image.Resampling.BILINEAR))
gray_cyc = np.array(Image.open(f_cyc).convert("L").resize((128, 128), Image.Resampling.BILINEAR))

print("Classroom at (64, 64):", compute_radial_vortex_score(gray_cls, 64, 64))
print("Cyclone preview at (64, 64):", compute_radial_vortex_score(gray_cyc, 64, 64))

# Real INSAT frames
from pathlib import Path
for f in sorted(Path("data/raw/insat/2023-003").glob("*.npy"))[:5]:
    arr = np.load(f)
    norm = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-6) * 255).astype(np.uint8)
    coh, unif = compute_radial_vortex_score(norm, 64, 64)
    print(f"INSAT {f.stem}: coherence={coh:.3f}, uniformity={unif:.3f}")
