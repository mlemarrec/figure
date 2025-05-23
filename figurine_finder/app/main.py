import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
import json

# Constants
IMG_WIDTH, IMG_HEIGHT = 150, 150  # Should match train_model.py
MODEL_PATH = "../model/figurine_classifier_model.h5" # Relative to this main.py file
CLASS_INDICES_PATH = "../model/class_indices.json" # Relative to this main.py file

# Global variables to hold the model and class labels
model = None
class_labels = None # Will store {index: class_name} mapping

def load_dependencies_on_startup():
    global model, class_labels
    
    # Load Model
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"Warning: Model file not found at {MODEL_PATH}. Prediction endpoint will be disabled.")
            model = None
        else:
            print(f"Loading model from {MODEL_PATH}...")
            model = tf.keras.models.load_model(MODEL_PATH)
            print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

    # Load Class Indices
    try:
        if not os.path.exists(CLASS_INDICES_PATH):
            print(f"Warning: Class indices file not found at {CLASS_INDICES_PATH}. Predictions will not have class names.")
            class_labels = None
        else:
            print(f"Loading class indices from {CLASS_INDICES_PATH}...")
            with open(CLASS_INDICES_PATH, 'r') as f:
                class_indices_from_json = json.load(f)
            # Invert the dictionary to map from index (int) to class name (str)
            class_labels = {int(v): k for k, v in class_indices_from_json.items()}
            print(f"Class labels loaded successfully: {class_labels}")
    except Exception as e:
        print(f"Error loading or processing class indices: {e}")
        class_labels = None

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


app = FastAPI(title="Figurine Classifier API", version="0.2.0") # Updated version

@app.on_event("startup")
async def startup_event():
    print("Application startup...")
    load_dependencies_on_startup()

@app.get("/")
async def read_root():
    global model, class_labels
    status = {
        "message": "Welcome to the Figurine Classifier API!",
        "model_loaded": model is not None,
        "class_labels_loaded": class_labels is not None
    }
    if class_labels:
        status["available_classes"] = list(class_labels.values())
    return status

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    global model, class_labels
    if model is None:
        raise HTTPException(status_code=503,
                            detail="Model not loaded. Please ensure the model is trained and available.")
    if class_labels is None:
        raise HTTPException(status_code=503,
                            detail="Class labels not loaded. Please ensure class_indices.json is available from training.")

    try:
        contents = await file.read()
        processed_image = preprocess_image(contents)
        
        predictions_array = model.predict(processed_image) # This is an array of probabilities
        
        # For multi-class, predictions_array is [[prob_class0, prob_class1, ...]]
        # For binary (sigmoid), it's [[prob_class1]] or similar, check model output shape
        
        if predictions_array.shape[1] > 1: # Softmax output (multi-class)
            predicted_index = np.argmax(predictions_array[0])
            confidence_score = float(predictions_array[0][predicted_index])
            predicted_class_name = class_labels.get(predicted_index, "Unknown_Class")
        else: # Sigmoid output (binary case, assuming class_labels might be {0: 'non_figure', 1: 'figure'} or similar)
            confidence_score = float(predictions_array[0][0])
            # Determine class based on threshold 0.5 for binary
            # This part needs careful handling based on how binary class_labels are stored
            # Assuming class_labels for binary is {0: 'negative_class', 1: 'positive_class'}
            # and Keras output 1 means 'positive_class'
            if confidence_score > 0.5:
                predicted_class_name = class_labels.get(1, "Positive_Class") # Or the actual positive class name
            else:
                predicted_class_name = class_labels.get(0, "Negative_Class") # Or the actual negative class name
                confidence_score = 1.0 - confidence_score # Confidence for the predicted negative class

        return {
            "figure_id": predicted_class_name,
            "confidence": confidence_score,
            "filename": file.filename
        }
    except HTTPException as e: # Re-raise HTTPExceptions from preprocess_image
        raise e
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

if __name__ == "__main__":
    import uvicorn
    print("Starting Uvicorn server for Figurine Classifier API...")
    # Note: Model and class_indices are loaded on startup by FastAPI.
    # If files don't exist, the /predict/ endpoint will return an error.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) # Added reload for dev
