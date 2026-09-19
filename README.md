# 🌾 AgroCare Major Project

AgroCare is a smart farming assistant with an AI leaf-disease detection module for **13 potato and tomato classes**.

## AI Disease Detection

The diagnosis page now uses the trained **EfficientNet-B0** model stored at:

`model/best_finetuned_model.pth`

The model is served by a small FastAPI service in `ml_backend/`.

### Supported classes

- Potato Bacterial Soft Rot
- Potato Healthy
- Potato Late Blight
- Potato Virus X (PVX)
- Potato Virus Y (PVY)
- Potato Leaf Roll Virus
- Tomato Bacterial Canker
- Tomato Cercospora Leaf Spot
- Tomato Healthy
- Tomato Late Blight
- Tomato Leaf Curl Virus
- Tomato Mosaic Virus
- Tomato Spotted Wilt Virus

## Project structure

```text
Agrocare/
├── model/
│   └── best_finetuned_model.pth
├── ml_backend/
│   ├── main.py
│   ├── model.py
│   └── requirements.txt
├── kisaan-backend/
│   └── Node/Express backend
├── src/
│   └── React/Vite frontend
├── package.json
└── vite.config.js
```

## Run the project locally

### 1. Frontend

From the project root:

```powershell
npm install
npm run dev
```

Frontend:

`http://localhost:5173`

### 2. ML disease-detection backend

Open a **new PowerShell window**:

```powershell
cd ml_backend
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

ML API:

`http://127.0.0.1:8000`

Health check:

`http://127.0.0.1:8000/`

### 3. Existing Node/Express backend

If you also need login/chat/database features, open another PowerShell window:

```powershell
cd kisaan-backend
npm install
npm start
```

Node backend:

`http://localhost:5000`

MongoDB is expected at:

`mongodb://localhost:27017/kisaan`

## Diagnosis flow

```text
React Diagnosis Page
        ↓
Upload leaf image
        ↓
FastAPI /predict
        ↓
EfficientNet-B0 trained checkpoint
        ↓
13-class prediction + confidence
        ↓
Treatment guidance
        ↓
DiagnosisResultCard
```

The diagnosis page does **not** send the leaf image to the Gemini/OpenRouter diagnosis endpoint. The trained local model is used for the crop-disease classification.

## Important

The model's confidence is a model prediction confidence, not a guarantee that the diagnosis is correct. Low-confidence or unusual field images should be confirmed by an agricultural expert.

## Existing backend

The Node/Express backend remains in `kisaan-backend/` for the project's existing authentication, chat, and other API functionality.
