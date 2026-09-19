# 🌱 AgroCare – AI-Powered Smart Crop Advisory System

AgroCare is an AI-powered smart agriculture platform designed to assist farmers in identifying crop diseases and making better crop-management decisions.

The platform uses a deep learning-based image classification model to analyze crop leaf images and identify diseases affecting potato and tomato plants. It provides the detected disease, model confidence, immediate actions, and preventive measures through an easy-to-use web interface.

how to run this project

Run karne ke liye

Terminal 1 — Frontend

npm install
npm run dev

Terminal 2 — ML model

cd ml_backend
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000

Terminal 3 — Existing Node backend
Agar login/chat/database use karna hai:

cd kisaan-backend
npm install
npm start

## ✨ Key Features

- 🌿 Potato & Tomato Disease Detection
- 🤖 EfficientNet-B0 Deep Learning Model
- 📸 Leaf Image-Based Diagnosis
- 🎯 Confidence Score
- 💊 Disease-Specific Recommendations
- 🌱 Preventive Measures
- ⚡ FastAPI ML Backend
- 💻 Modern Web Interface
- 🔄 Real-time communication between frontend and ML model

## 🛠️ Tech Stack

- React / JavaScript
- Python
- PyTorch
- EfficientNet-B0
- FastAPI
- Node.js / Express
- OpenRouter API
- HTML / CSS
- REST API

## 🧠 Machine Learning

The disease detection module is trained to classify **13 potato and tomato leaf conditions**. The trained EfficientNet-B0 model processes an uploaded leaf image and returns the predicted disease class and confidence score.

## 🎯 Project Goal

The goal of AgroCare is to make AI-assisted crop diagnosis more accessible and provide farmers with quick, understandable information that can support early disease identification and crop management.
