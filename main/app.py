import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from io import BytesIO
from PIL import Image

# Initialize FastAPI app
app = FastAPI()

# Load the trained model
model = load_model("model.keras")

# Define class labels
pest_names = ['Brown Planthopper', 'Green Leaf Hopper', 'Rice Black Bug', 'Rice Bug', 'White Yellow Stemborer']

# Pest information
pest_info = {
    'Green Leaf Hopper': {
        'Details': "Most common leafhoppers in rice fields. They spread the viral disease tungro. Both nymphs and adults feed by extracting plant sap.",
        'Host plant': "Rice, sugarcane, and gramineous weeds.",
        'Life Cycle': "Egg – 6-9 days, Nymph – 16-18 days, Adult – 2-3 weeks.",
        'Damage': "Yellowing of leaves, stunted growth, drying up of plant.",
        'Identification': "Yellow dwarf, yellow-orange leaf.",
        'Management': "Cultural: Synchronous planting, sanitation. Biological: Lady Beetle, Ground Beetle, Metarhizium. Chemical: Last resort."
    },
    'Brown Planthopper': {
        'Details': "Occurs only in rice fields, sucks the sap at the base of tillers, can cause Ragged Stunt virus and Grassy Stunt.",
        'Host plant': "Rice only.",
        'Life Cycle': "Egg – 5-8 days, Nymph – 13-15 days, Adult – 12-15 days.",
        'Damage': "Plants turn yellow and dry rapidly, heavy infestation creates sooty molds and hopper burn.",
        'Identification': "Crescent-shaped white eggs, white to brown nymphs, browning and drying of plants.",
        'Management': "Cultural: Synchronous planting, sanitation. Biological: Lady Beetle, Ground Beetle, Metarhizium. Chemical: Last resort."
    },
    'Rice Black Bug': {
        'Details': "Commonly found in rainfed and irrigated wetland environments, prefers poorly drained fields.",
        'Host plant': "Rice crop and weeds.",
        'Life Cycle': "Egg – 4-7 days, Nymph – 29-35 days, Adult – 3-4 weeks.",
        'Damage': "Browning of leaves, deadheart, bugburn, reduced tillering.",
        'Identification': "Check leaves for discoloration, decreased tillering, deadhearts.",
        'Management': "Cultural: Synchronous planting, sanitation. Biological: Light trap, Metarhizium. Chemical: Last resort."
    },
    'Rice Bug': {
        'Details': "Rice bug populations increase near woodlands, weedy areas, and staggered rice planting.",
        'Host plant': "Wild grasses.",
        'Life Cycle': "Egg – 4-9 days, Nymph – 17-27 days, Adult – 30-50 days.",
        'Damage': "Unfilled grains, discoloration, deformed grains.",
        'Identification': "Small or shriveled grains, erect panicles.",
        'Management': "Cultural: Synchronous planting, sanitation. Biological: Metarhizium, Beauveria. Chemical: Last resort."
    },
    'White Yellow Stemborer': {
        'Details': "A major insect pest that infests rice at all stages of growth.",
        'Host plant': "Rice, maize, millet, sorghum, sugarcane, wheat, grasses.",
        'Life Cycle': "Egg – 5-9 days, Larva – 20-36 days, Pupa – 6-11 days, Adult – 2-5 days.",
        'Damage': "Deadheart, drying of central tiller, whiteheads.",
        'Identification': "Deadhearts, tiny holes on stems, frass or fecal matter.",
        'Management': "Cultural: Synchronous planting, sanitation. Biological: Trichogramma, Lady Beetle, Spiders, Metarhizium. Chemical: Last resort."
    }
}

# Function to preprocess image
def preprocess_image(image: Image.Image):
    image = image.resize((180, 180))  # Resize to model input size
    image_array = img_to_array(image)  # Convert to NumPy array
    image_expanded = np.expand_dims(image_array, axis=0)  # Expand dimensions for batch
    return image_expanded

# API endpoint to classify an uploaded image
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read image file
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes))

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make prediction
    predictions = model.predict(processed_image)
    result = tf.nn.softmax(predictions[0])
    
    # Get predicted class and confidence
    predicted_class = pest_names[np.argmax(result)]
    confidence_score = float(np.max(result) * 100)

    # Get additional pest information
    info = pest_info.get(predicted_class, {})

    # Return response
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
