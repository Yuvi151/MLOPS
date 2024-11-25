import os
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
import uvicorn
from pydantic import BaseModel
from typing import List
import json

app = FastAPI(title="Plant Disease Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionResponse(BaseModel):
    disease: str
    confidence: float
    possible_diseases: List[dict]

def load_model_alternative(model_path):
    """
    Alternative model loading method with multiple fallback strategies
    """
    # Strategy 1: Use tf.saved_model.load()
    try:
        model = tf.saved_model.load(model_path)
        return model
    except Exception as e:
        print(f"tf.saved_model.load failed: {e}")
    
    # Strategy 2: Attempt loading with Keras
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        print(f"Keras model loading failed: {e}")
    
    # Strategy 3: Check file permissions and path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    if not os.access(model_path, os.R_OK):
        # Attempt to change file permissions
        try:
            os.chmod(model_path, 0o644)  # Read permissions for owner, group, and others
            print(f"Modified permissions for {model_path}")
        except Exception as perm_error:
            print(f"Could not modify permissions: {perm_error}")
    
    raise ValueError("Could not load model using any available method")

# Flexible model path detection
POSSIBLE_MODEL_PATHS = [
    os.path.join(os.getcwd(), 'plant_disease'),
    os.path.join(os.path.dirname(__file__), 'plant_disease'),
    '/app/plant_disease',  # Docker container path
    '~/plant_disease',    # User home directory
]

# Find first valid model path
MODEL_PATH = next((path for path in POSSIBLE_MODEL_PATHS if os.path.exists(path)), None)

# Load model with error handling
try:
    model = load_model_alternative(MODEL_PATH) if MODEL_PATH else None
except Exception as e:
    print(f"Model loading error: {e}")
    model = None

# Load class indices from file
class_indices_path = os.path.join(os.getcwd(), "class_indices.json")
try:
    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)
    DISEASE_CLASSES = list(class_indices.keys())  # Extract class names
except Exception as e:
    DISEASE_CLASSES = []
    print(f"Error loading class indices: {e}")

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array.astype(np.float32)

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        processed_image = preprocess_image(image)

        # Predict
        predictions = model(tf.convert_to_tensor(processed_image)).numpy().flatten()

        if len(predictions) != len(DISEASE_CLASSES):
            raise HTTPException(
                status_code=500,
                detail=f"Mismatch: Model predictions ({len(predictions)}) vs classes ({len(DISEASE_CLASSES)})"
            )

        top_pred_idx = np.argmax(predictions)
        confidence = float(predictions[top_pred_idx])

        # Top 3 predictions
        top_3_idx = np.argsort(predictions)[-3:][::-1]
        possible_diseases = [
            {"disease": DISEASE_CLASSES[idx], "confidence": float(predictions[idx])}
            for idx in top_3_idx
        ]

        return PredictionResponse(
            disease=DISEASE_CLASSES[top_pred_idx],
            confidence=confidence,
            possible_diseases=possible_diseases,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
