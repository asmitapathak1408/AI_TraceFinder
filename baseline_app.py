# app.py
import streamlit as st
from src.baseline_model.predict_baseline import predict_scanner
from PIL import Image
import numpy as np

st.title("Scanner Prediction App")
st.write("Upload an image, and the model will predict which scanner it belongs to.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["tif", "png", "jpg", "jpeg"])

# Choose model
model_choice = st.selectbox("Select Model", ["Random Forest", "SVM"])
model_key = "rf" if model_choice == "Random Forest" else "svm"

if uploaded_file is not None:
    # Convert uploaded file to an OpenCV-compatible array
    image = Image.open(uploaded_file).convert("L")  # convert to grayscale
    img_array = np.array(image)

    # Save temporarily
    temp_path = "temp_image.tif"
    image.save(temp_path)

    # Predict
    try:
        pred, prob = predict_scanner(temp_path, model_choice=model_key)
        st.image(img_array, caption='Uploaded Image', use_column_width=True)
        st.success(f"Predicted Scanner: {pred}")
        st.write("Class Probabilities:")
        for idx, p in enumerate(prob):
            st.write(f"Class {idx}: {p:.4f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
