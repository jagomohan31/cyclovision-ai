import numpy as np
from PIL import Image
import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# Verify get_ood_classifier works smoothly
_mobilenet_weights = MobileNet_V3_Small_Weights.DEFAULT
_mobilenet = mobilenet_v3_small(weights=_mobilenet_weights).eval()
print("OOD Classifier ready!")
