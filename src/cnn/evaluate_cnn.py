import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

# Load model + data
model = tf.keras.models.load_model("models/dual_branch_cnn.h5")
X_img = np.load("processed_data/new_processed/official_images.npy")
X_noise = np.load("processed_data/new_processed/flatfield_noise.npy")
y = np.load("processed_data/new_processed/labels.npy")

print(f"Shapes before processing: X_img={X_img.shape}, X_noise={X_noise.shape}")

# --- START: ADD THIS MISSING CODE ---

# Convert flatfield noise to grayscale if it's not already (matching train_cnn.py)
if X_noise.shape[-1] == 3:
    X_noise = np.mean(X_noise, axis=-1, keepdims=True)

# Upsample X_noise to match the number of official images
if len(X_noise) < len(X_img):
    reps = len(X_img) // len(X_noise) + 1
    X_noise = np.tile(X_noise, (reps, 1, 1, 1))[:len(X_img)]

# --- END: ADD THIS MISSING CODE ---

print(f"Shapes after processing: X_img={X_img.shape}, X_noise={X_noise.shape}")

# Predict
y_pred = model.predict({"official_input": X_img, "noise_input": X_noise})
y_pred_classes = np.argmax(y_pred, axis=1)

print("Classification Report:")
print(classification_report(y, y_pred_classes))

print("Confusion Matrix:")
print(confusion_matrix(y, y_pred_classes))