import tensorflow as tf
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "WM811k_Dataset"
MODEL_PATH = BASE_DIR / "wafer_defect_cnn.keras"

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
BATCH_SIZE = 32

print("Loading dataset...")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_dataset = test_dataset.map(
    lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)
)

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")
print("Evaluating...\n")

loss, accuracy = model.evaluate(test_dataset, verbose=1)

print("\n==============================")
print(f"Test Loss:     {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print("==============================")

y_true = []
y_pred = []

for images, labels in test_dataset:
    predictions = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))

print("\nClassification Report:\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        zero_division=0
    )
)

print("\nConfusion Matrix:\n")

print(confusion_matrix(y_true, y_pred))
