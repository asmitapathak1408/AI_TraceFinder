import streamlit as st
import pandas as pd
import numpy as np
import os
import cv2
from PIL import Image, ExifTags
from scipy.stats import skew, kurtosis, entropy

# 🔹 New theme and title
st.set_page_config(page_title="📊 Dataset Feature Extractor", layout="wide")
st.title("📊 Image Dataset Feature Extractor")

# --- Feature extractor ---
def extract_features(image_path, class_label):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"file_name": os.path.basename(image_path), "class": class_label, "error": "Unreadable file"}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Dimensions & file info
        height, width = gray.shape
        file_size = os.path.getsize(image_path) / 1024  # KB
        aspect_ratio = round(width / height, 3)

        # Intensity stats
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        skewness = skew(gray.flatten())
        kurt = kurtosis(gray.flatten())

        # Entropy
        hist = np.histogram(gray, bins=256, range=(0, 255))[0]
        shannon_entropy = entropy(hist + 1e-9)

        # Edge density
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.mean(edges > 0)

        # Texture (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Color stats (RGB means)
        mean_colors = cv2.mean(img)[:3]  # BGR
        mean_r, mean_g, mean_b = mean_colors[2], mean_colors[1], mean_colors[0]


        return {
            "file_name": os.path.basename(image_path),
            "class": class_label,
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            "file_size_kb": round(file_size, 2),
            "mean_intensity": round(mean_intensity, 3),
            "std_intensity": round(std_intensity, 3),
            "skewness": round(skewness, 3),
            "kurtosis": round(kurt, 3),
            "entropy": round(shannon_entropy, 3),
            "edge_density": round(edge_density, 3),
            "laplacian_var": round(laplacian_var, 3),
            "mean_r": round(mean_r, 2),
            "mean_g": round(mean_g, 2),
            "mean_b": round(mean_b, 2)
        }
    except Exception as e:
        return {"file_name": image_path, "class": class_label, "error": str(e)}

# --- UI for dataset path ---
dataset_root = st.text_input("📂 Enter dataset root path:", "")

if dataset_root and os.path.isdir(dataset_root):
    st.info("🔎 Scanning dataset...")
    records = []

    classes = [f for f in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, f))]
    st.success(f" Detected {len(classes)} classes: {classes}")

    for class_dir in classes:
        class_path = os.path.join(dataset_root, class_dir)
        files = [f for f in os.listdir(class_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        st.write(f" Class '{class_dir}' → {len(files)} images")
        for fname in files:
            path = os.path.join(class_path, fname)
            rec = extract_features(path, class_dir)
            records.append(rec)

    df = pd.DataFrame(records)
    st.subheader("📑 Features Extracted (Preview)")
    st.dataframe(df.head(20))

    save_path = os.path.join(dataset_root, "metadata_features.csv")
    df.to_csv(save_path, index=False)
    st.success(f"✅ Features saved to {save_path}")

    if "class" in df.columns:
        st.subheader("📊 Class Distribution")
        st.bar_chart(df["class"].value_counts())



elif dataset_root:
    st.error("❌ Invalid dataset path. Please enter a valid folder.")
