import torch
import numpy as np
from PIL import Image
from src.models.detection import CycloneUNet
from src.config import CROP_SIZE

dev = torch.device("cpu")
det = CycloneUNet(in_channels=1, base_channels=16).to(dev)
det.load_state_dict(torch.load("detector_checkpoint.pt", map_location=dev, weights_only=True))
det.eval()

f_cls = r"C:\Users\Jagomohan Das\Downloads\WhatsApp Image 2026-04-23 at 20.58.00.jpeg"
f_cyc = r"C:\Users\Jagomohan Das\Downloads\202306060015_preview.png"

for name, path in [("Cyclone Preview", f_cyc), ("Classroom", f_cls)]:
    img = Image.open(path).convert("L")
    # strip white border if any
    arr = np.array(img)
    mask = (arr > 15) & (arr < 240)
    if np.any(mask):
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        if rmax - rmin > 50 and cmax - cmin > 50:
            arr = arr[rmin:rmax+1, cmin:cmax+1]
    
    pil_crop = Image.fromarray(arr).resize((CROP_SIZE, CROP_SIZE), Image.Resampling.BILINEAR)
    norm = np.array(pil_crop, dtype=np.float32) / 255.0
    t = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0).float()
    
    with torch.no_grad():
        prob = torch.sigmoid(det(t))[0, 0].numpy()
        
    print(f"=== {name} ===")
    print(f"Prob min={prob.min():.3f}, max={prob.max():.3f}, mean={prob.mean():.3f}, std={prob.std():.3f}")
    # High confidence mask area
    p80 = (prob > 0.80).sum()
    p50 = (prob > 0.50).sum()
    print(f"Pixels > 0.80: {p80} ({p80/16384*100:.2f}%)")
    print(f"Pixels > 0.50: {p50} ({p50/16384*100:.2f}%)")
    # Peak location
    peak_y, peak_x = np.unravel_index(np.argmax(prob), prob.shape)
    print(f"Peak prob={prob[peak_y, peak_x]:.3f} at ({peak_y}, {peak_x})")
