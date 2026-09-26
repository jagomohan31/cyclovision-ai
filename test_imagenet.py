import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image

weights = MobileNet_V3_Small_Weights.DEFAULT
model = mobilenet_v3_small(weights=weights).eval()
preprocess = weights.transforms()

f_cls = r"C:\Users\Jagomohan Das\Downloads\WhatsApp Image 2026-04-23 at 20.58.00.jpeg"
f_cyc = r"C:\Users\Jagomohan Das\Downloads\202306060015_preview.png"

for name, path in [("Classroom", f_cls), ("Cyclone", f_cyc)]:
    img = Image.open(path).convert("RGB")
    batch = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        prediction = model(batch).squeeze(0).softmax(0)
        top5_prob, top5_catid = torch.topk(prediction, 5)
        print(f"=== {name} ===")
        for i in range(5):
            print(f"   {i+1}: {weights.meta['categories'][top5_catid[i]]} ({top5_prob[i]*100:.1f}%)")
