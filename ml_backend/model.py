import os
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

POTATO_CLASSES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
]

TOMATO_MODEL_ID = "oshriagronov/tomadoc-mythos"
TOMATO_CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# Local high-accuracy potato model
potato_model = efficientnet_b0(weights=None)
potato_model.classifier[1] = nn.Linear(potato_model.classifier[1].in_features, 3)
potato_path = os.path.join(os.path.dirname(__file__), "..", "model", "potato_95plus.pth")
potato_checkpoint = torch.load(potato_path, map_location=DEVICE, weights_only=False)
potato_model.load_state_dict(potato_checkpoint["model_state_dict"])
potato_model = potato_model.to(DEVICE).eval()

# Tomato model is downloaded lazily from Hugging Face on first tomato request.
tomato_model = None
tomato_processor = None

def get_tomato_model():
    global tomato_model, tomato_processor
    if tomato_model is None:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        tomato_processor = AutoImageProcessor.from_pretrained(TOMATO_MODEL_ID)
        tomato_model = AutoModelForImageClassification.from_pretrained(TOMATO_MODEL_ID).to(DEVICE).eval()
    return tomato_model, tomato_processor
