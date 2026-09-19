from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
from torchvision import transforms

from model import model, CLASS_NAMES, DEVICE

app = FastAPI(title="AgroCare ML Disease Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

TREATMENTS = {
    "Potato_Bacterial Soft Rot": {
        "immediate": [
            "Remove and safely discard severely affected leaves or plants.",
            "Avoid over-irrigation and waterlogging around potato plants.",
        ],
        "preventive": [
            "Use healthy, disease-free seed tubers.",
            "Improve field drainage and avoid crop injury during cultivation.",
        ],
    },
    "Potato_Healthy Leaf Disease Dataset": {
        "immediate": ["No disease pattern was detected in this image."],
        "preventive": [
            "Continue regular field monitoring.",
            "Maintain balanced irrigation, nutrition, and good field sanitation.",
        ],
    },
    "Potato_Late Blight": {
        "immediate": [
            "Remove badly affected plant material and avoid spreading infected debris.",
            "Improve air circulation and avoid prolonged leaf wetness.",
        ],
        "preventive": [
            "Use disease-free planting material and resistant varieties where available.",
            "Follow locally recommended fungicide and crop-rotation practices.",
        ],
    },
    "Potato_PVX- Potato Virus X": {
        "immediate": [
            "Remove and isolate symptomatic plants where practical.",
            "Disinfect tools before working on healthy plants.",
        ],
        "preventive": [
            "Use certified disease-free seed tubers.",
            "Control volunteer plants and avoid unnecessary plant-to-plant contact.",
        ],
    },
    "Potato_PVY- Potato Virus Y": {
        "immediate": [
            "Remove strongly symptomatic plants where practical.",
            "Control aphids and other vectors using locally approved methods.",
        ],
        "preventive": [
            "Use certified virus-free seed material.",
            "Monitor vector populations and remove volunteer potato plants.",
        ],
    },
    "Potato_Potato Leaf Roll Virus": {
        "immediate": [
            "Remove severely symptomatic plants where practical.",
            "Manage aphid vectors using locally approved integrated pest-management practices.",
        ],
        "preventive": [
            "Use certified virus-free seed tubers.",
            "Monitor aphids regularly and maintain field sanitation.",
        ],
    },
    "Tomato_Bacterial Canker": {
        "immediate": [
            "Remove severely infected plant parts and dispose of them away from the field.",
            "Avoid overhead irrigation and handling plants when foliage is wet.",
        ],
        "preventive": [
            "Use clean seed and sanitized tools.",
            "Rotate crops and maintain good field sanitation.",
        ],
    },
    "Tomato_Cercospora Leaf Spot": {
        "immediate": [
            "Remove heavily infected leaves and improve air circulation.",
            "Avoid overhead watering where possible.",
        ],
        "preventive": [
            "Remove crop debris after harvest.",
            "Use locally recommended fungicide practices when necessary.",
        ],
    },
    "Tomato_Healthy Leaf Disease Dataset": {
        "immediate": ["No disease pattern was detected in this image."],
        "preventive": [
            "Continue regular monitoring for early symptoms.",
            "Maintain balanced irrigation, nutrition, and field hygiene.",
        ],
    },
    "Tomato_Late Blight": {
        "immediate": [
            "Remove severely affected leaves or plants and keep infected debris away from healthy plants.",
            "Reduce leaf wetness and improve airflow.",
        ],
        "preventive": [
            "Use healthy planting material and monitor weather conditions favorable to blight.",
            "Follow locally recommended fungicide and crop-management practices.",
        ],
    },
    "Tomato_Tomato Leaf Curl Virus": {
        "immediate": [
            "Remove severely symptomatic plants where practical.",
            "Manage whitefly vectors using locally approved integrated pest-management methods.",
        ],
        "preventive": [
            "Use healthy seedlings and suitable resistant varieties where available.",
            "Monitor and manage whiteflies and remove volunteer host plants.",
        ],
    },
    "Tomato_Tomato Mosaic Virus": {
        "immediate": [
            "Remove severely symptomatic plants and avoid spreading sap between plants.",
            "Disinfect hands and tools after handling infected plants.",
        ],
        "preventive": [
            "Use certified clean seed and healthy transplants.",
            "Maintain strict sanitation and avoid tobacco-product contamination around plants.",
        ],
    },
    "Tomato_Tomato Spotted Wilt Virus": {
        "immediate": [
            "Remove severely symptomatic plants where practical.",
            "Manage thrips vectors using locally approved integrated pest-management methods.",
        ],
        "preventive": [
            "Use healthy transplants and monitor thrips regularly.",
            "Remove weeds and volunteer hosts that can support virus vectors.",
        ],
    },
}


def display_name(name: str) -> str:
    name = name.replace("Potato_Healthy Leaf Disease Dataset", "Potato Healthy")
    name = name.replace("Tomato_Healthy Leaf Disease Dataset", "Tomato Healthy")
    name = name.replace("Potato_PVX- Potato Virus X", "Potato Virus X (PVX)")
    name = name.replace("Potato_PVY- Potato Virus Y", "Potato Virus Y (PVY)")
    name = name.replace("Potato_Potato Leaf Roll Virus", "Potato Leaf Roll Virus")
    name = name.replace("Tomato_Tomato Leaf Curl Virus", "Tomato Leaf Curl Virus")
    name = name.replace("Tomato_Tomato Mosaic Virus", "Tomato Mosaic Virus")
    name = name.replace("Tomato_Tomato Spotted Wilt Virus", "Tomato Spotted Wilt Virus")
    name = name.replace("Potato_", "Potato ")
    name = name.replace("Tomato_", "Tomato ")
    return name


@app.get("/")
def home():
    return {
        "message": "AgroCare ML Disease Detection API is running",
        "model": "EfficientNet-B0",
        "classes": len(CLASS_NAMES),
        "device": str(DEVICE),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        return {"success": False, "message": "Please upload a valid image file."}

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)

    class_id = predicted_class.item()
    confidence_value = confidence.item()
    disease = CLASS_NAMES[class_id]

    return {
        "success": True,
        "result": {
            "diagnosis": display_name(disease),
            "confidenceScore": f"{confidence_value * 100:.2f}%",
            "classId": class_id,
            "treatment": TREATMENTS[disease],
        },
    }
