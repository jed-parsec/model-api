import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from io import BytesIO
from PIL import Image

app = FastAPI()

# Load the trained model
model = load_model("model.keras")

# Pest classes
pest_names = ['Brown Planthopper', 'Green Leaf Hopper', 'Rice Black Bug', 'Rice Bug', 'White Yellow Stemborer']

# Pest Information
pest_info = {
    'Green Leaf Hopper': {
        'Details': "Most common leafhoppers in rice fields. They spread the viral disease tungro. Both nymphs and adults feed by extracting plant sap.",
        'Host plant': "Rice, sugarcane, and gramineous weeds.",
        'Life Cycle': "Egg – 6-9 days, Nymph – 16-18 days, Adult – 2-3 weeks.",
        'Damage': "Yellowing of leaves, stunted growth, drying up of plant.",
        'Identification': "Yellow dwarf, yellow-orange leaf.",
        'Management': "Cultural: Synchronous planting, sanitation. Biological: Lady Beetle, Ground Beetle, Metarhizium. Chemical: Last resort."
    }
}

# Image Preprocessing Function
def preprocess_image(image: Image.Image):
    """Preprocess the image for model prediction."""
    image = image.resize((180, 180))  # Resize image
    image_array = img_to_array(image)  # Convert to array
    image_expanded = np.expand_dims(image_array, axis=0)  # Expand dimensions
    return image_expanded

@app.get("/")
def home():
    return {"message": "Welcome to the Pest Recognition API!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Handles image upload and returns pest classification."""
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes))

    processed_image = preprocess_image(image)

    # Get predictions
    predictions = model.predict(processed_image)
    result = tf.nn.softmax(predictions[0])

    predicted_class = pest_names[np.argmax(result)]
    confidence_score = float(np.max(result) * 100)

    info = pest_info.get(predicted_class, {})

    return {
        "filename": file.filename,
        "predicted_class": predicted_class,
        "confidence": confidence_score,
        "details": info.get("Details", "No details available."),
        "host_plant": info.get("Host plant", "Unknown"),
        "life_cycle": info.get("Life Cycle", "Unknown"),
        "damage": info.get("Damage", "Unknown"),
        "identification": info.get("Identification", "Unknown"),
        "management": info.get("Management", "Unknown")
    }

# Vercel handler
def handler(event, context):
    return app(event, context)
