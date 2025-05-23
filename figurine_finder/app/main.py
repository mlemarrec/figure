import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io

# Constants
IMG_WIDTH, IMG_HEIGHT = 150, 150  # Should match train_model.py
MODEL_PATH = "../model/figurine_classifier_model.h5" # Relative to this main.py file

# Global variable to hold the model
model = None

def load_model_on_startup():
    global model
    try:
        # Check if the model file exists before attempting to load
        if not os.path.exists(MODEL_PATH):
            print(f"Warning: Model file not found at {MODEL_PATH}. Prediction endpoint will be disabled.")
            model = None
            return

        print(f"Loading model from {MODEL_PATH}...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

def preprocess_image(image_bytes: bytes):
    """
    Preprocesses the image bytes to the format expected by MobileNetV2.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize((IMG_WIDTH, IMG_HEIGHT))
        image_array = np.array(image)
        image_array = image_array / 255.0  # Rescale to [0, 1] as done in training
        image_array = np.expand_dims(image_array, axis=0)  # Create batch dimension
        return image_array
    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image file or format: {e}")


app = FastAPI(title="Figurine Classifier API", version="0.1.0")

@app.on_event("startup")
async def startup_event():
    print("Application startup...")
    load_model_on_startup()

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Figurine Classifier API!"}

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    global model
    if model is None:
        raise HTTPException(status_code=503,
                            detail="Model not loaded. Please ensure the model is trained and available.")

    try:
        contents = await file.read()
        processed_image = preprocess_image(contents)
        
        prediction = model.predict(processed_image)
        
        is_figure = bool(prediction[0][0] > 0.5)
        confidence = float(prediction[0][0])
        
        return {"is_figure": is_figure, "confidence": confidence, "filename": file.filename}
    except HTTPException as e: # Re-raise HTTPExceptions from preprocess_image
        raise e
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

if __name__ == "__main__":
    import uvicorn
    print("Starting Uvicorn server for Figurine Classifier API...")
    # Note: Model is loaded on startup by FastAPI.
    # If MODEL_PATH doesn't exist, the /predict/ endpoint will return an error.
    uvicorn.run(app, host="0.0.0.0", port=8000)
