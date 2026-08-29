
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Semiconductor Wafer Defect Detection",
    page_icon="🔬",
    layout="centered"
)

# -----------------------------
# Class names
# -----------------------------
CLASS_NAMES = [
    "Center",
    "Donut",
    "Edge Local",
    "Edge Ring",
    "Local",
    "Scratch",
    "near full",
    "none",
    "random"
]

IMG_SIZE = (128, 128)

# -----------------------------
# Title
# -----------------------------
st.title("🔬 Semiconductor Wafer Defect Detection")

st.write(
    "Upload a wafer map image and the trained CNN model "
    "will predict the defect category."
)

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("wafer_defect_cnn.keras")

model = load_model()

st.success("CNN model loaded successfully!")

# -----------------------------
# Upload image
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a wafer map image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Wafer Map",
        use_container_width=True
    )

    # Resize
    resized_image = image.resize(IMG_SIZE)

    # Normalize
    image_array = np.array(resized_image).astype("float32") / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    predictions = model.predict(image_array, verbose=0)

    predicted_index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = predictions[0][predicted_index] * 100

    # Results
    st.success(f"Predicted Defect: {predicted_class}")

    st.info(f"Confidence: {confidence:.2f}%")

    # Probabilities
    st.subheader("Prediction Probabilities")

    probabilities = {
        CLASS_NAMES[i]: float(predictions[0][i] * 100)
        for i in range(len(CLASS_NAMES))
    }

    st.bar_chart(probabilities)
