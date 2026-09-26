import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image
from pathlib import Path
import numpy as np

weights = MobileNet_V3_Small_Weights.DEFAULT
model = mobilenet_v3_small(weights=weights).eval()
preprocess = weights.transforms()

# List of terrestrial man-made categories in ImageNet (rooms, indoor, objects, vehicles, furniture, etc.)
# If the top predictions of an image are dominated by these, it is definitely a camera photo of a terrestrial scene!
def is_terrestrial_photo(img_pil):
    batch = preprocess(img_pil.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        preds = model(batch).squeeze(0).softmax(0)
        top5_prob, top5_idx = torch.topk(preds, 5)
        
    top5_cats = [weights.meta["categories"][idx.item()] for idx in top5_idx]
    top5_probs = [p.item() for p in top5_prob]
    return top5_cats, top5_probs

f_cls = r"C:\Users\Jagomohan Das\Downloads\WhatsApp Image 2026-04-23 at 20.58.00.jpeg"
f_cyc = r"C:\Users\Jagomohan Das\Downloads\202306060015_preview.png"

# Test classroom
cats_cls, p_cls = is_terrestrial_photo(Image.open(f_cls))
print("Classroom top 5:", list(zip(cats_cls, [round(p*100, 1) for p in p_cls])))

# Test cyclone preview (border stripped)
img_p = Image.open(f_cyc).convert("L")
arr_p = np.array(img_p)
mask = (arr_p > 15) & (arr_p < 240)
rmin, rmax = np.where(np.any(mask, axis=1))[0][[0, -1]]
cmin, cmax = np.where(np.any(mask, axis=0))[0][[0, -1]]
img_p_content = Image.fromarray(arr_p[rmin:rmax+1, cmin:cmax+1])
cats_cyc, p_cyc = is_terrestrial_photo(img_p_content)
print("Cyclone preview top 5:", list(zip(cats_cyc, [round(p*100, 1) for p in p_cyc])))

# Test 5 real INSAT frames
for f in sorted(Path("data/raw/insat/2023-003").glob("*.npy"))[:5]:
    arr = np.load(f)
    norm = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-6) * 255).astype(np.uint8)
    cats, probs = is_terrestrial_photo(Image.fromarray(norm))
    print(f"INSAT {f.stem}: {cats[0]} ({probs[0]*100:.1f}%), {cats[1]} ({probs[1]*100:.1f}%)")
