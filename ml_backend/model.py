import os
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

NUM_CLASSES = 13

CLASS_NAMES = [
    "Potato_Bacterial Soft Rot",
    "Potato_Healthy Leaf Disease Dataset",
    "Potato_Late Blight",
    "Potato_PVX- Potato Virus X",
    "Potato_PVY- Potato Virus Y",
    "Potato_Potato Leaf Roll Virus",
    "Tomato_Bacterial Canker",
    "Tomato_Cercospora Leaf Spot",
    "Tomato_Healthy Leaf Disease Dataset",
    "Tomato_Late Blight",
    "Tomato_Tomato Leaf Curl Virus",
    "Tomato_Tomato Mosaic Virus",
    "Tomato_Tomato Spotted Wilt Virus",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = efficientnet_b0(weights=None)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "best_finetuned_model.pth")
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()
