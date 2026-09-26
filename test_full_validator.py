import numpy as np
from PIL import Image
import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# Load lightweight MobileNetV3 for terrestrial scene verification
_mobilenet_weights = MobileNet_V3_Small_Weights.DEFAULT
_mobilenet = mobilenet_v3_small(weights=_mobilenet_weights).eval()
_mobilenet_transforms = _mobilenet_weights.transforms()

# Terrestrial/indoor/man-made ImageNet categories that NEVER appear in satellite views
TERRESTRIAL_INDOOR_KEYWORDS = {
    "projector", "cinema", "shoe shop", "barbell", "desk", "dining table", "restaurant",
    "bookshop", "library", "classroom", "auditorium", "theater", "prison", "bakery",
    "butcher shop", "grocery store", "barbershop", "living room", "bedroom", "kitchen",
    "wardrobe", "refrigerator", "microwave", "dishwasher", "toaster", "television",
    "computer keyboard", "laptop", "monitor", "mouse", "cellular telephone", "suit",
    "jean", "jersey", "trench coat", "wig", "sunglasses", "park bench", "traffic light",
    "street sign", "fire engine", "garbage truck", "minivan", "convertible", "sports car"
}

def auto_strip_canvas_borders(img_pil: Image.Image) -> Image.Image:
    """Auto-crops solid white or black canvas borders (e.g. from matplotlib/saved previews)."""
    arr = np.array(img_pil.convert("L"))
    h, w = arr.shape
    # Check if border rows/cols are uniform white (>= 245) or uniform black (<= 10)
    mask = (arr > 15) & (arr < 240)
    if np.any(mask):
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        # Only crop if there is a substantial border (> 10 pixels on any side)
        if (rmin > 10 or (h - 1 - rmax) > 10 or cmin > 10 or (w - 1 - cmax) > 10):
            if (rmax - rmin >= 40) and (cmax - cmin >= 40):
                return img_pil.crop((cmin, rmin, cmax + 1, rmax + 1))
    return img_pil

def validate_custom_image(img_pil: Image.Image) -> tuple[bool, str, Image.Image]:
    """
    Validates whether an uploaded image is a plausible meteorological satellite frame
    or an out-of-distribution non-satellite input (e.g. QR code, classroom photo, document).
    Returns (is_valid, reason, preprocessed_pil).
    """
    # 1. Auto-strip canvas margins (fixes false rejection of exported previews)
    stripped = auto_strip_canvas_borders(img_pil)
    
    gray_arr = np.array(stripped.convert("L"), dtype=np.float32)
    h, w = gray_arr.shape
    if h < 32 or w < 32:
        return False, "Image dimensions are too small for satellite analysis.", stripped

    # 2. QR code / Synthetic Graphic Check:
    # A synthetic QR code / barcode / text document has almost NO pixels in the mid-tones
    norm_01 = (gray_arr - gray_arr.min()) / (gray_arr.max() - gray_arr.min() + 1e-6)
    mid_tone_ratio = float(((norm_01 >= 0.15) & (norm_01 <= 0.85)).mean())
    unique_levels = len(np.unique(np.round(gray_arr)))

    if mid_tone_ratio < 0.03 and unique_levels < 15:
        return False, f"Detected high-contrast binary/synthetic pattern (only {unique_levels} tonal levels, {mid_tone_ratio*100:.1f}% mid-tones, e.g. QR code or graphic).", stripped

    # 3. Terrestrial Scene Verification via MobileNetV3:
    # Checks if ImageNet detects terrestrial man-made indoor objects/rooms (e.g. projector, cinema, desk)
    try:
        rgb_tensor = _mobilenet_transforms(stripped.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            preds = _mobilenet(rgb_tensor).squeeze(0).softmax(0)
            top3_prob, top3_idx = torch.topk(preds, 3)
            
        top3_cats = [_mobilenet_weights.meta["categories"][idx.item()] for idx in top3_idx]
        top3_probs = [p.item() for p in top3_prob]
        
        # Check if top prediction is a terrestrial indoor/man-made scene with substantial confidence
        terrestrial_matches = [cat for cat in top3_cats if cat in TERRESTRIAL_INDOOR_KEYWORDS]
        if terrestrial_matches:
            top_cat = top3_cats[0]
            top_p = top3_probs[0]
            if top_cat in TERRESTRIAL_INDOOR_KEYWORDS and top_p > 0.05:
                return False, f"Terrestrial camera photograph detected (identified scene features: '{top_cat}', not an atmospheric satellite view).", stripped
            if len(terrestrial_matches) >= 2 and sum(top3_probs) > 0.12:
                return False, f"Terrestrial indoor/camera scene detected (identified features: {', '.join(terrestrial_matches)}).", stripped
    except Exception as e:
        pass

    return True, "Valid meteorological satellite frame.", stripped

# Test all 4 cases
f_cyc = r"C:\Users\Jagomohan Das\Downloads\202306060015_preview.png"
f_cls = r"C:\Users\Jagomohan Das\Downloads\WhatsApp Image 2026-04-23 at 20.58.00.jpeg"

print("--- Test 1: Real Cyclone Preview ---")
val, msg, _ = validate_custom_image(Image.open(f_cyc))
print(f"Result: valid={val} | {msg}")

print("\n--- Test 2: Classroom Photo ---")
val, msg, _ = validate_custom_image(Image.open(f_cls))
print(f"Result: valid={val} | {msg}")

print("\n--- Test 3: Synthetic QR Code ---")
rng = np.random.default_rng(42)
qr = Image.fromarray(rng.choice([0, 255], size=(200, 200), p=[0.5, 0.5]).astype(np.uint8))
val, msg, _ = validate_custom_image(qr)
print(f"Result: valid={val} | {msg}")

print("\n--- Test 4: Real INSAT Frames ---")
from pathlib import Path
for f in sorted(Path("data/raw/insat/2023-003").glob("*.npy"))[:3]:
    arr = np.load(f)
    norm = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-6) * 255).astype(np.uint8)
    val, msg, _ = validate_custom_image(Image.fromarray(norm))
    print(f"INSAT {f.stem}: valid={val} | {msg}")
