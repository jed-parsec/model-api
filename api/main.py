import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io
import json

# Load the trained model
model = load_model("pest_recognition_model.keras")

# Define class labels
pest_names = ['Brown Planthopper', 'Green Leaf Hopper', 'Rice Black Bug', 'Rice Bug', 'White Yellow Stemborer']

# Pest information dictionary
pest_info = {
    'Green Leaf Hopper': {
        'Details': "Most common leafhoppers in rice fields. They spread the viral disease tungro. Both nymphs and adults feed by extracting plant sap.",
        'Damage': "Yellowing of leaves, stunted growth, drying up of plant.",
    },
    'Brown Planthopper': {
        'Details': "Occurs only in rice fields, sucks the sap at the base of tillers, can cause Ragged Stunt virus and Grassy Stunt.",
        'Damage': "Plants turn yellow and dry rapidly, heavy infestation creates sooty molds and hopper burn.",
    },
    'Rice Black Bug': {
        'Details': "Commonly found in rainfed and irrigated wetland environments, prefers poorly drained fields.",
        'Damage': "Browning of leaves, deadheart, bugburn, reduced tillering.",
    },
    'Rice Bug': {
        'Details': "Rice bug populations increase near woodlands, weedy areas, and staggered rice planting.",
        'Damage': "Unfilled grains, discoloration, deformed grains.",
    },
    'White Yellow Stemborer': {
        'Details': "A major insect pest that infests rice at all stages of growth.",
        'Damage': "Deadheart, drying of central tiller, whiteheads.",
    }
}

# Function to preprocess the image
def preprocess_image(image: Image.Image):
    image = image.resize((180, 180))  # Resize image to match model input
    image_array = img_to_array(image) / 255.0  # Normalize
    image_expanded = np.expand_dims(image_array, axis=0)  # Expand dims for model input
    return image_expanded

# Streamlit API
st.title("Pest Recognition API")

# Handle file upload
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make a prediction
    predictions = model.predict(processed_image)
    result = tf.nn.softmax(predictions[0])

    # Get the predicted class and confidence score
    predicted_class = pest_names[np.argmax(result)]
    confidence_score = float(np.max(result) * 100)

    # Get additional pest information
    info = pest_info.get(predicted_class, {})

    # JSON response
    response = {
        "filename": uploaded_file.name,
        "predicted_class": predicted_class,
        "confidence": confidence_score,
        "details": info.get("Details", "No details available."),
        "damage": info.get("Damage", "Unknown")
    }

    # Show the JSON response in the Streamlit UI
    st.json(response)

    # Serve JSON response when accessed via a query parameter
    query_params = st.experimental_get_query_params()
    if "api" in query_params:
        st.write(json.dumps(response))

