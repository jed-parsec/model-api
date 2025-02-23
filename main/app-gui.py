import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io

# Load the trained model
model = load_model("model.keras")

# Define pest names and information
pest_names = ['Brown Planthopper', 'Green Leaf Hopper', 'Rice Black Bug', 'Rice Bug', 'White Yellow Stemborer']

pest_info = {
    'Green Leaf Hopper': {
        'Details': "Most common leafhoppers in rice fields. They spread the viral disease tungro.",
        'Damage': "Yellowing of leaves, stunted growth, drying up of plant.",
        'Management': "Cultural: Synchronous planting, sanitation. Biological: Lady Beetle, Ground Beetle."
    },
    'Brown Planthopper': {
        'Details': "Occurs only in rice fields, sucks the sap at the base of tillers.",
        'Damage': "Plants turn yellow and dry rapidly, can cause sooty molds and hopper burn.",
        'Management': "Cultural: Synchronous planting, sanitation. Biological: Metarhizium."
    },
    'Rice Black Bug': {
        'Details': "Common in rainfed and irrigated wetland environments.",
        'Damage': "Browning of leaves, deadheart, reduced tillering.",
        'Management': "Cultural: Synchronous planting, sanitation. Biological: Light trap, Metarhizium."
    },
    'Rice Bug': {
        'Details': "Rice bug populations increase near woodlands and staggered planting.",
        'Damage': "Unfilled grains, discoloration, deformed grains.",
        'Management': "Cultural: Synchronous planting, sanitation. Biological: Metarhizium."
    },
    'White Yellow Stemborer': {
        'Details': "Major pest infesting rice at all growth stages.",
        'Damage': "Deadheart, drying of central tiller, whiteheads.",
        'Management': "Cultural: Synchronous planting, sanitation. Biological: Trichogramma, Lady Beetle."
    }
}

# Image preprocessing function
def preprocess_image(image):
    image = image.resize((180, 180))
    image_array = img_to_array(image)
    image_expanded = np.expand_dims(image_array, axis=0)
    return image_expanded

# Streamlit UI
st.title("Pest Identification App")
st.write("Upload an image of a rice pest to identify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess and predict
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    result = tf.nn.softmax(predictions[0])
    predicted_class = pest_names[np.argmax(result)]
    confidence_score = float(np.max(result) * 100)

    # Display results
    st.subheader(f'🦟 Identified Pest: {predicted_class}')
    st.write(f'**Confidence Score:** {confidence_score:.2f}%')

    info = pest_info.get(predicted_class, {})
    st.write(f'**Details:** {info.get("Details", "No details available.")}')
    st.write(f'**Damage:** {info.get("Damage", "Unknown")}')
    st.write(f'**Management:** {info.get("Management", "Unknown")}')
