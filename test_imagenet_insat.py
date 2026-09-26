import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image
from pathlib import Path
import numpy as np

weights = MobileNet_V3_Small_Weights.DEFAULT
model = mobilenet_v3_small(weights=weights).eval()
preprocess = weights.transforms()

# Test on 10 real INSAT frames
insat_files = list(Path("data/raw/insat/2023-003").glob("*.npy"))[:10]
for f in insat_files:
    arr = np.load(f)
    norm = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-6) * 255).astype(np.uint8)
    img = Image.fromarray(norm).convert("RGB")
    batch = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        preds = model(batch).squeeze(0).softmax(0)
        top_prob, top_idx = torch.topk(preds, 3)
        cat_names = [f"{weights.meta['categories'][top_idx[i]]} ({top_prob[i]*100:.1f}%)" for i in range(3)]
        print(f"INSAT {f.stem}: {', '.join(cat_names)}")
