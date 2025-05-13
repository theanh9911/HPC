# API for Pneumonia Detection
# api/main.py

import os
import json
import uvicorn
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Pneumonia Detection API",
    description="API for detecting pneumonia from chest X-ray images using deep learning",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global variables for model and config
model = None
model_config = None

@app.on_event("startup")
async def startup_event():
    """Load model and configuration on startup."""
    global model, model_config
    
    try:
        logger.info("Loading model and configuration...")
        
        # Load model
        model_path = os.environ.get("MODEL_PATH", "model/model.h5")
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        model = load_model(model_path)
        logger.info("Model loaded successfully!")
        
        # Load configuration
        config_path = os.environ.get("CONFIG_PATH", "model/model_config.json")
        if not os.path.exists(config_path):
            logger.error(f"Config file not found at {config_path}")
            raise FileNotFoundError(f"Config file not found at {config_path}")
        
        with open(config_path, 'r') as f:
            model_config = json.load(f)
        logger.info("Configuration loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        # We'll let the app start, but endpoints will return errors if model is None

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Pneumonia Detection API is running! Use /predict to classify chest X-ray images."}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": "Model not loaded"}
        )
    return {"status": "healthy", "message": "Service is running correctly"}

def preprocess_image(image_data, target_size=(224, 224)):
    """
    Preprocess the image for model inference.
    """
    try:
        # Open image
        img = Image.open(io.BytesIO(image_data))
        
        # Convert grayscale to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize image
        img = img.resize(target_size)
        
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise HTTPException(status_code=422, detail=f"Error processing image: {str(e)}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict pneumonia from a chest X-ray image.
    """
    # Check if model is loaded
    if model is None:
        logger.error("Model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded. Service unavailable.")
    
    try:
        # Read image file
        image_data = await file.read()
        if not image_data:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Preprocess the image
        processed_image = preprocess_image(
            image_data,
            target_size=tuple(model_config['input_shape'][:2])
        )
        
        # Make prediction
        prediction = model.predict(processed_image)[0][0]
        
        # Determine class
        class_idx = 1 if prediction > 0.5 else 0
        class_name = model_config['class_names'][class_idx]
        confidence = float(prediction if class_idx == 1 else 1 - prediction)
        
        # Log prediction
        logger.info(f"Prediction: {class_name} with confidence {confidence:.4f}")
        
        # Return result
        return {
            "prediction": class_name,
            "confidence": confidence,
            "pneumonia_probability": float(prediction),
            "status": "success"
        }
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

@app.get("/model-info")
async def model_info():
    """Get information about the loaded model."""
    if model is None or model_config is None:
        raise HTTPException(status_code=503, detail="Model or configuration not loaded")
    
    return {
        "model_architecture": model_config.get("model_architecture", "Unknown"),
        "input_shape": model_config.get("input_shape", "Unknown"),
        "classes": model_config.get("class_names", ["Unknown"]),
        "status": "success"
    }

if __name__ == "__main__":
    # Run the FastAPI app with uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api_main:app", host="localhost", port=port, reload=False)
