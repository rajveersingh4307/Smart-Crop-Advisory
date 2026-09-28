from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
from torchvision import transforms

from model import (
    potato_model, get_tomato_model, POTATO_CLASSES, TOMATO_CLASSES, DEVICE,
)

app = FastAPI(title="AgroCare Unified Plant Disease API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

potato_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

TREATMENTS = {
    "healthy": {"immediate": ["No disease pattern was detected in this image."], "preventive": ["Continue regular monitoring and maintain good irrigation, nutrition, and field hygiene."]},
    "early": {"immediate": ["Remove badly affected leaves and avoid spreading infected plant material."], "preventive": ["Maintain field sanitation, airflow, and follow locally recommended disease-management practices."]},
    "late": {"immediate": ["Remove severely affected plant material and reduce prolonged leaf wetness."], "preventive": ["Use healthy planting material and follow locally recommended fungicide and crop-management practices."]},
    "bacterial": {"immediate": ["Remove severely infected plant parts and avoid handling wet foliage."], "preventive": ["Use clean planting material, sanitize tools, and avoid overhead irrigation."]},
    "virus": {"immediate": ["Isolate/remove strongly symptomatic plants where practical and avoid spreading sap between plants."], "preventive": ["Use healthy planting material and manage insect vectors using locally approved IPM methods."]},
    "leaf_mold": {"immediate": ["Remove heavily affected leaves and improve airflow around the crop."], "preventive": ["Reduce humidity and leaf wetness and maintain good field sanitation."]},
    "septoria": {"immediate": ["Remove heavily infected leaves and keep crop debris away from healthy plants."], "preventive": ["Improve airflow and avoid overhead irrigation where possible."]},
    "spider": {"immediate": ["Inspect the underside of leaves and remove heavily infested foliage where practical."], "preventive": ["Monitor mite populations regularly and use locally approved integrated pest-management methods."]},
    "target": {"immediate": ["Remove heavily affected leaves and improve airflow."], "preventive": ["Keep foliage dry when possible and remove infected crop debris."]},
}

def treatment_for(disease: str):
    d = disease.lower()
    if "healthy" in d: return TREATMENTS["healthy"]
    if "bacterial" in d: return TREATMENTS["bacterial"]
    if "spider" in d: return TREATMENTS["spider"]
    if "septoria" in d: return TREATMENTS["septoria"]
    if "leaf_mold" in d: return TREATMENTS["leaf_mold"]
    if "target" in d: return TREATMENTS["target"]
    if "yellow_leaf_curl" in d or "mosaic" in d: return TREATMENTS["virus"]
    if "late_blight" in d: return TREATMENTS["late"]
    if "early_blight" in d: return TREATMENTS["early"]
    return TREATMENTS["healthy"]

def display_name(name: str) -> str:
    replacements = {
        "Potato___Early_blight": "Potato Early Blight",
        "Potato___Late_blight": "Potato Late Blight",
        "Potato___healthy": "Potato Healthy",
        "Tomato___Bacterial_spot": "Tomato Bacterial Spot",
        "Tomato___Early_blight": "Tomato Early Blight",
        "Tomato___Late_blight": "Tomato Late Blight",
        "Tomato___Leaf_Mold": "Tomato Leaf Mold",
        "Tomato___Septoria_leaf_spot": "Tomato Septoria Leaf Spot",
        "Tomato___Spider_mites Two-spotted_spider_mite": "Tomato Spider Mites",
        "Tomato___Target_Spot": "Tomato Target Spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato Yellow Leaf Curl Virus",
        "Tomato___Tomato_mosaic_virus": "Tomato Mosaic Virus",
        "Tomato___healthy": "Tomato Healthy",
    }
    return replacements.get(name, name)

@app.get("/")
def home():
    return {"message": "AgroCare unified disease API is running", "potato_classes": 3, "tomato_classes": 10, "device": str(DEVICE)}

@app.post("/predict")
async def predict(file: UploadFile = File(...), crop: str = Form("potato")):
    if not file.content_type or not file.content_type.startswith("image/"):
        return {"success": False, "message": "Please upload a valid image file."}
    crop = crop.strip().lower()
    if crop not in {"potato", "tomato"}:
        return {"success": False, "message": "Crop must be potato or tomato."}

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    with torch.no_grad():
        if crop == "potato":
            tensor = potato_transform(image).unsqueeze(0).to(DEVICE)
            logits = potato_model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            idx = int(probs.argmax().item())
            confidence = float(probs[idx].item())
            disease = POTATO_CLASSES[idx]
        else:
            tomato, processor = get_tomato_model()
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            logits = tomato(**inputs).logits
            probs = torch.softmax(logits, dim=1)[0]
            idx = int(probs.argmax().item())
            confidence = float(probs[idx].item())
            disease = tomato.config.id2label.get(idx, TOMATO_CLASSES[idx])

    return {
        "success": True,
        "result": {
            "crop": crop,
            "diagnosis": display_name(disease),
            "confidenceScore": f"{confidence * 100:.2f}%",
            "classId": idx,
            "treatment": treatment_for(disease),
        },
    }
