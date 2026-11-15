
import io
import os
from contextlib import asynccontextmanager
from typing import Annotated

import mlflow.pyfunc
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

import mlflow

MODEL_URI = "models:/food11@champion"

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000"
)

CLASSES = [
    "Bread",
    "Dairy product",
    "Dessert",
    "Egg",
    "Fried food",
    "Meat",
    "Noodles-Pasta",
    "Rice",
    "Seafood",
    "Soup",
    "Vegetable-Fruit",
]

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    if os.getenv("SKIP_MODEL"):
        model = None
        print("SKIP_MODEL enabled: skipping MLflow model loading.")
    else:
        mlflow.set_tracking_uri(TRACKING_URI)

        print(f"Connecting to MLflow: {TRACKING_URI}")
        print(f"Loading model: {MODEL_URI}")

        model = mlflow.pyfunc.load_model(MODEL_URI)

        print("Model loaded successfully!")

    yield


app = FastAPI(
    title="Food-11 Prediction API",
    description="Food classification using ResNet18 and MLflow",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: Annotated[UploadFile, File()]):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Invalid image file",
        )

    tensor = transform(image).unsqueeze(0)

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded",
        )

    predictions = model.predict(
        tensor.numpy()
    )

    logits = np.asarray(predictions).reshape(-1)

    if len(logits) != len(CLASSES):
        raise HTTPException(
            status_code=500,
            detail="Unexpected model output",
        )

    probabilities = torch.softmax(
        torch.tensor(logits, dtype=torch.float32),
        dim=0,
    )

    predicted_index = int(torch.argmax(probabilities))

    return {
        "category": CLASSES[predicted_index],
        "confidence": round(
            float(probabilities[predicted_index]), 4
        ),
    }
