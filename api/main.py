import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

# Load the trained model
model = load_model("model.keras")

# Pest names and info
pest_names = ['Brown Planthopper', 'Green Leaf Hopper', 'Rice Black Bug', 'Rice Bug', 'White Yellow Stemborer']
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

# Preprocessing function
def preprocess_image(image: Image.Image):
    image = image.resize((180, 180))  # Resize to match model input
    image_array = img_to_array(image) / 255.0  # Normalize
    image_expanded = np.expand_dims(image_array, axis=0)  
    return image_expanded

# Streamlit App
st.title("Pest Recognition AI")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    processed_image = preprocess_image(image)

    # Prediction
    predictions = model.predict(processed_image)
    result = tf.nn.softmax(predictions[0])
    predicted_class = pest_names[np.argmax(result)]
    confidence_score = float(np.max(result) * 100)

    # Display results
    st.subheader(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence_score:.2f}%")
    st.write("### Pest Details:")
    st.write(f"**{pest_info[predicted_class]['Details']}**")
    st.write(f"**Damage:** {pest_info[predicted_class]['Damage']}")

