import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

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
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "WM811k_Dataset"
MODEL_PATH = BASE_DIR / "wafer_defect_cnn.keras"

# -----------------------------
# Title
# -----------------------------
st.title("🔬 Semiconductor Wafer Defect Detection")

st.write(
    "Select a wafer map image from the dataset "
    "and the trained CNN model will predict the defect category."
)

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

st.success("CNN model loaded successfully!")

# -----------------------------
# Get dataset images
# -----------------------------
image_files = []

for class_name in CLASS_NAMES:
    class_dir = DATA_DIR / class_name

    if class_dir.exists():
        image_files.extend(
            list(class_dir.glob("*.jpg")) +
            list(class_dir.glob("*.jpeg")) +
            list(class_dir.glob("*.png"))
        )

# -----------------------------
# Image selection
# -----------------------------
if len(image_files) == 0:

    st.error("No dataset images found.")

else:

    selected_image = st.selectbox(
        "Select a wafer map image",
        image_files,
        format_func=lambda x: str(x.relative_to(DATA_DIR))
    )

    # -----------------------------
    # Load selected image
    # -----------------------------
    image = Image.open(selected_image).convert("RGB")

    st.image(
        image,
        caption=f"Selected Image: {selected_image.name}",
        use_container_width=True
    )

    # -----------------------------
    # Prediction button
    # -----------------------------
    if st.button("🔍 Predict Defect"):

        resized_image = image.resize(IMG_SIZE)

        image_array = np.array(resized_image).astype("float32") / 255.0

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        predictions = model.predict(
            image_array,
            verbose=0
        )

        predicted_index = int(np.argmax(predictions[0]))

        predicted_class = CLASS_NAMES[predicted_index]

        confidence = float(
            predictions[0][predicted_index] * 100
        )

        st.success(
            f"Predicted Defect: {predicted_class}"
        )

        st.info(
            f"Confidence: {confidence:.2f}%"
        )

        # -----------------------------
        # Prediction probabilities
        # -----------------------------
        st.subheader("Prediction Probabilities")

        probabilities = {
            CLASS_NAMES[i]: float(predictions[0][i] * 100)
            for i in range(len(CLASS_NAMES))
        }

        st.bar_chart(probabilities)
