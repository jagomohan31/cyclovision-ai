import torch
import numpy as np
from PIL import Image
from src.models.detection import CycloneUNet
from src.models.classification import CycloneClassifier
from src.config import NUM_CATEGORIES, ERA5_FEATURES, CROP_SIZE

dev = torch.device("cpu")
det = CycloneUNet(in_channels=1, base_channels=16).to(dev)
det.load_state_dict(torch.load("detector_checkpoint.pt", map_location=dev, weights_only=True))
det.eval()

clf = CycloneClassifier(num_categories=NUM_CATEGORIES, num_era5_features=len(ERA5_FEATURES)).to(dev)
clf.load_state_dict(torch.load("classifier_checkpoint.pt", map_location=dev, weights_only=True))
clf.eval()

f_cls = r"C:\Users\Jagomohan Das\Downloads\WhatsApp Image 2026-04-23 at 20.58.00.jpeg"
f_cyc = r"C:\Users\Jagomohan Das\Downloads\202306060015_preview.png"

for name, path in [("Classroom", f_cls), ("Cyclone Preview", f_cyc)]:
    img = Image.open(path).convert("L")
    c_pil = img.resize((CROP_SIZE, CROP_SIZE), Image.Resampling.BILINEAR)
    c_norm = np.array(c_pil, dtype=np.float32) / 255.0
    t = torch.from_numpy(c_norm).unsqueeze(0).unsqueeze(0).float()
    
    with torch.no_grad():
        logits_det = det(t)
        prob_det = torch.sigmoid(logits_det)[0, 0].numpy()
        
        era5_dummy = torch.zeros(1, len(ERA5_FEATURES))
        logits_clf = clf(t, era5_dummy)
        probs_clf = torch.softmax(logits_clf, dim=1).numpy().flatten()
        
        print(f"=== {name} ===")
        print(f"  Det prob min={prob_det.min():.3f}, max={prob_det.max():.3f}, mean={prob_det.mean():.3f}")
        print(f"  Clf top cat={probs_clf.argmax()} ({probs_clf.max()*100:.1f}%)")
        print(f"  Clf entropy = {-np.sum(probs_clf * np.log(probs_clf + 1e-9)):.3f}")
