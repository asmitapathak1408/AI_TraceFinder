import streamlit as st
import os
import cv2
import numpy as np
import pickle
import tensorflow as tf
import pywt
from skimage.feature import local_binary_pattern
from PIL import Image
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import joblib # Import joblib for loading baseline models
from scipy.stats import skew, kurtosis, entropy # Needed for baseline features
from sklearn.exceptions import NotFittedError # For checking scaler

# ---------------------------
# 1. Global Parameters & Paths
# ---------------------------
# --- Hybrid CNN Paths ---
HYBRID_ART_DIR = "Residuals_Paths"
HYBRID_CKPT_PATH = os.path.join(HYBRID_ART_DIR, "scanner_hybrid_final.keras")
HYBRID_LE_PATH = os.path.join(HYBRID_ART_DIR, "hybrid_label_encoder.pkl") # Assuming same LE
HYBRID_SCALER_PATH = os.path.join(HYBRID_ART_DIR, "hybrid_feat_scaler.pkl")

# --- Baseline Model Paths ---
BASELINE_MODEL_DIR = "models"
BASELINE_RF_PATH = os.path.join(BASELINE_MODEL_DIR, "random_forest.pkl")
BASELINE_SVM_PATH = os.path.join(BASELINE_MODEL_DIR, "svm.pkl")
BASELINE_SCALER_PATH = os.path.join(BASELINE_MODEL_DIR, "scaler.pkl")
BASELINE_LE_PATH = HYBRID_LE_PATH # ASSUMING same label encoder

# --- Shared & EDA Paths ---
FP_PATH = "scanner_fingerprints.pkl" # Root folder (Used ONLY by Hybrid and EDA viz)
ORDER_NPY = "fp_keys.npy"           # Root folder (Used ONLY by Hybrid and EDA viz)
EDA_DATA_PATH = os.path.join(HYBRID_ART_DIR, "features.pkl") # Assumes 'labels' key for EDA count

# --- General Config ---
IMG_SIZE = (256, 256)
NUM_BASELINE_FEATURES = 10 # From feature_extractor.py

# ---------------------------
# 2. Load Models & Assets
# ---------------------------
@st.cache_resource
def load_all_assets():
    """Loads all models, scalers, encoders, fingerprints (for hybrid), and EDA labels."""
    assets = {
        "hybrid_model": None, "hybrid_le": None, "hybrid_scaler": None,
        "baseline_rf": None, "baseline_svm": None, "baseline_scaler": None, "baseline_le": None,
        "scanner_fps": None, "fp_keys": None, # Specific to Hybrid
        "labels_eda": None
    }
    error_occurred = False # Flag for critical errors

    # --- Load Hybrid Assets ---
    try: assets["hybrid_model"] = tf.keras.models.load_model(HYBRID_CKPT_PATH, compile=False)
    except Exception as e: st.error(f"Hybrid Model Error ({HYBRID_CKPT_PATH}): {e}"); error_occurred = True
    try:
        with open(HYBRID_LE_PATH, "rb") as f: assets["hybrid_le"] = pickle.load(f)
        assets["baseline_le"] = assets["hybrid_le"] # Assign to baseline LE too
    except Exception as e: st.error(f"Label Encoder Error ({HYBRID_LE_PATH}): {e}"); error_occurred = True
    try:
        with open(HYBRID_SCALER_PATH, "rb") as f: assets["hybrid_scaler"] = pickle.load(f)
    except Exception as e: st.error(f"Hybrid Scaler Error ({HYBRID_SCALER_PATH}): {e}"); error_occurred = True

    # --- Load Baseline Assets (using joblib) ---
    try: assets["baseline_rf"] = joblib.load(BASELINE_RF_PATH)
    except Exception as e: st.warning(f"Baseline RF Load Error/Not Found ({BASELINE_RF_PATH}): {e}.")
    try: assets["baseline_svm"] = joblib.load(BASELINE_SVM_PATH)
    except Exception as e: st.warning(f"Baseline SVM Load Error/Not Found ({BASELINE_SVM_PATH}): {e}.")
    try: assets["baseline_scaler"] = joblib.load(BASELINE_SCALER_PATH)
    except Exception as e: st.warning(f"Baseline Scaler Load Error/Not Found ({BASELINE_SCALER_PATH}): {e}. Baseline models disabled.")

    # Disable baseline models if scaler failed
    if assets["baseline_scaler"] is None:
        assets["baseline_rf"] = None
        assets["baseline_svm"] = None
    else:
        # Check if baseline scaler is fitted (basic check)
        try:
             # A more robust check might involve checking 'n_features_in_' if available
             check_scaler = assets["baseline_scaler"]
             if hasattr(check_scaler, 'mean_') and check_scaler.mean_ is None:
                  st.warning("Baseline Scaler might not be fitted (mean_ is None). Baseline predictions may fail.")
                  assets["baseline_scaler"] = None # Disable
             elif hasattr(check_scaler, 'n_features_in_') and check_scaler.n_features_in_ != NUM_BASELINE_FEATURES:
                  st.warning(f"Baseline Scaler expected {check_scaler.n_features_in_} features, but code expects {NUM_BASELINE_FEATURES}.")
                  assets["baseline_scaler"] = None # Disable
             elif not hasattr(check_scaler, 'transform'): # Simple check if it looks like a scaler
                  st.warning("Loaded baseline scaler might be invalid type.")
                  assets["baseline_scaler"] = None # Disable if invalid
        except NotFittedError:
             st.warning("Baseline Scaler is not fitted. Baseline predictions will fail.")
             assets["baseline_scaler"] = None # Disable if not fitted
        except AttributeError: # Handle cases where attributes don't exist
             st.warning("Could not fully validate baseline scaler type/fit status.")
        except Exception as e:
            st.warning(f"Could not validate baseline scaler: {e}")


    # --- Load Shared Assets (Fingerprints - ONLY for Hybrid and Viz) ---
    try:
        with open(FP_PATH, "rb") as f: assets["scanner_fps"] = pickle.load(f)
    except Exception as e: st.warning(f"Fingerprints Error ({FP_PATH}): {e}. Hybrid model/EDA viz may fail.")
    try: assets["fp_keys"] = np.load(ORDER_NPY, allow_pickle=True).tolist()
    except Exception as e: st.warning(f"FP Keys Error ({ORDER_NPY}): {e}. Hybrid model/EDA viz may fail.")

    # --- Load EDA Labels ---
    try:
        with open(EDA_DATA_PATH, "rb") as f:
            data = pickle.load(f); assets["labels_eda"] = data.get('labels')
            if not assets["labels_eda"]: st.warning(f"EDA Warning: 'labels' key missing/empty in {EDA_DATA_PATH}.")
    except Exception as e: st.warning(f"EDA Warning: Error loading data from {EDA_DATA_PATH}: {e}")

    if error_occurred: # Stop only if essential assets failed
        st.error("Critical error loading essential model assets. App cannot continue."); st.stop()

    assets["baseline_available"] = (assets["baseline_rf"] is not None or assets["baseline_svm"] is not None) and assets["baseline_scaler"] is not None
    return assets

st.set_page_config(layout="wide")
st.title("Trace Finder: Scanner Brand Identification")

# --- Sidebar ---
st.sidebar.title("App Information & Controls")
assets = load_all_assets()

st.sidebar.success("Assets loading complete.")
if assets["hybrid_le"]:
    st.sidebar.write(f"Models expect **{len(assets['hybrid_le'].classes_)}** classes:")
    st.sidebar.code('\n'.join(assets['hybrid_le'].classes_))
else: st.sidebar.error("Label Encoder failed to load!")

if assets["labels_eda"] is not None: st.sidebar.info(f"EDA label data loaded: {len(assets['labels_eda'])} samples.")
else: st.sidebar.warning("EDA label data could not be loaded.")

# --- Model Selection ---
st.sidebar.header("Model Selection")
model_options = []
if assets["hybrid_model"]: model_options.append("Hybrid CNN")
if assets["baseline_available"]: model_options.append("Baseline")

if not model_options: st.sidebar.error("No models loaded successfully!"); st.stop()
elif len(model_options) == 1: model_type = model_options[0]; st.sidebar.info(f"Only {model_type} model is available.")
else: model_type = st.sidebar.radio("Choose Model Type:", model_options, key="model_type_radio")

baseline_model_choice = None
if model_type == "Baseline":
    baseline_options = []
    if assets["baseline_rf"]: baseline_options.append("Random Forest")
    if assets["baseline_svm"]: baseline_options.append("SVM")
    if baseline_options and assets["baseline_available"]:
        baseline_model_choice = st.sidebar.radio("Choose Baseline Model:", baseline_options, key="baseline_choice_radio")
    else:
        st.sidebar.error("Error: Baseline selected but required assets are missing.")
        if assets["hybrid_model"]:
             st.sidebar.warning("Switching to Hybrid CNN."); model_type = "Hybrid CNN"
        else: st.stop()

# --- Main App Tabs ---
tab1, tab2 = st.tabs(["Scanner Prediction", "Data Overview (EDA)"])

# ---------------------------
# 3. Helper & Preprocessing Functions
# (Functions remain the same)
# ---------------------------
def corr2d(a, b): # Used by Hybrid
    a = a.astype(np.float32).ravel(); b = b.astype(np.float32).ravel()
    a -= a.mean(); b -= b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a @ b) / d) if d != 0 else 0.0

def fft_radial_energy(img, K=6): # Used by Hybrid
    f = np.fft.fftshift(np.fft.fft2(img)); mag = np.abs(f)
    h, w = mag.shape; cy, cx = h//2, w//2
    yy, xx = np.ogrid[:h, :w]; r = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    rmax = r.max() + 1e-6; bins = np.linspace(0, rmax, K+1); feats = []
    for i in range(K):
        m = (r >= bins[i]) & (r < bins[i+1])
        feats.append(float(mag[m].mean() if m.any() else 0.0))
    return feats

def lbp_hist_safe(img, P=8, R=1.0): # Used by Hybrid
    rng = float(np.ptp(img)); n_bins = P + 2
    if rng < 1e-12: return [0.0] * n_bins
    else: g = (img - float(np.min(img))) / (rng + 1e-8)
    g8 = (g * 255.0).astype(np.uint8)
    codes = local_binary_pattern(g8, P=P, R=R, method="uniform")
    hist, _ = np.histogram(codes, bins=np.arange(n_bins+1), range=(0, n_bins), density=True)
    return hist.astype(np.float32).tolist()

def normalize_image_display(img):
    norm_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return norm_img.astype(np.uint8)

def preprocess_image_common(img_array):
    if img_array is None: raise ValueError("Input image array is None")
    if img_array.ndim == 3: gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    elif img_array.ndim == 2: gray = img_array
    else: raise ValueError(f"Unsupported image dimensions: {img_array.ndim}")
    gray_resized = cv2.resize(gray, IMG_SIZE, interpolation=cv2.INTER_AREA)
    return gray_resized

def calculate_residual(gray_resized):
    img_norm = gray_resized.astype(np.float32) / 255.0
    cA, (cH, cV, cD) = pywt.dwt2(img_norm, 'haar')
    cH.fill(0); cV.fill(0); cD.fill(0)
    den = pywt.idwt2((cA, (cH, cV, cD)), 'haar')
    if den.shape != img_norm.shape: den = cv2.resize(den, img_norm.shape[::-1], interpolation=cv2.INTER_LINEAR)
    return (img_norm - den).astype(np.float32)

def extract_baseline_features(gray_resized):
    try:
        height, width = gray_resized.shape
        aspect_ratio = round(width / height, 3) if height != 0 else 0
        mean_intensity = np.mean(gray_resized)
        std_intensity = np.std(gray_resized)
        skewness = skew(gray_resized.flatten())
        kurt = kurtosis(gray_resized.flatten())
        hist = np.histogram(gray_resized, bins=256, range=(0, 255))[0]
        shannon_entropy = entropy(hist + 1e-9) # Add epsilon
        edges = cv2.Canny(gray_resized, 100, 200)
        edge_density = np.mean(edges > 0)
        features_with_placeholder = [
            float(width), float(height), float(aspect_ratio),
            0.0, # Placeholder for file_size_kb
            round(mean_intensity, 3), round(std_intensity, 3), round(skewness, 3),
            round(kurt, 3), round(shannon_entropy, 3), round(edge_density, 3)
        ]
        baseline_features = np.array(features_with_placeholder, dtype=np.float32)
        if baseline_features.shape[0] != NUM_BASELINE_FEATURES: raise ValueError(f"Baseline features shape: {baseline_features.shape[0]}, expected {NUM_BASELINE_FEATURES}.")
        return baseline_features.reshape(1, -1)
    except Exception as e:
        st.error(f"Error during baseline feature extraction: {e}")
        return np.zeros((1, NUM_BASELINE_FEATURES), dtype=np.float32)

def extract_hybrid_features(res, loaded_scanner_fps, loaded_fp_keys):
     if not loaded_scanner_fps or not loaded_fp_keys: raise ValueError("Fingerprints/keys not loaded for hybrid.")
     num_zncc = len(loaded_fp_keys)
     v_corr = [corr2d(res, loaded_scanner_fps[k]) for k in loaded_fp_keys]
     v_fft  = fft_radial_energy(res, K=6)
     v_lbp  = lbp_hist_safe(res, P=8, R=1.0)
     if len(v_lbp) != 10: v_lbp.extend([0.0] * (10 - len(v_lbp)))
     v = np.array(v_corr + v_fft + v_lbp, dtype=np.float32)
     expected_len = num_zncc + 6 + 10
     if v.shape[0] != expected_len: raise ValueError(f"Hybrid feature length mismatch: expected {expected_len}, got {v.shape[0]}.")
     return v.reshape(1, -1)

# ---------------------------
# 4. EDA Section (Simplified)
# ---------------------------
with tab2:
    st.header("Data Overview")
    eda_labels_ready = assets["labels_eda"] is not None
    eda_fps_ready = assets["scanner_fps"] is not None

    if eda_labels_ready:
        st.subheader("1. Overall Distribution of Scanner Brands (Training Labels)")
        try:
            # Assume labels might be detailed (e.g., HP_150), extract base brand
            def get_base_brand(label_str):
                # Basic split, adjust if format is different (e.g., uses '-')
                return label_str.split('_')[0] if isinstance(label_str, str) else label_str

            base_labels = [get_base_brand(lbl) for lbl in assets["labels_eda"]]
            label_df = pd.DataFrame(base_labels, columns=['Scanner Brand'])
            scanner_counts = label_df['Scanner Brand'].value_counts().reset_index()
            scanner_counts.columns = ['Scanner Brand', 'Number of Images']

            fig_bar = px.bar(scanner_counts, x='Scanner Brand', y='Number of Images',
                             title='Total Images per Scanner Brand (Based on loaded labels)',
                             text='Number of Images', color='Scanner Brand', template='plotly_white')
            fig_bar.update_traces(textposition='outside')
            fig_bar.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_bar, use_container_width=True)
            with st.expander("Show Raw Counts"): st.dataframe(scanner_counts, use_container_width=True)
        except Exception as e: st.error(f"Error generating class distribution plot: {e}")
    else: st.warning(f"Could not load labels from '{EDA_DATA_PATH}'. Cannot display class distribution.")

    if eda_fps_ready:
        st.subheader("2. Average Scanner Fingerprints (Noise Patterns)")
        st.write("Average residual noise patterns from the Flatfield dataset (used for Hybrid model correlation features).")
        try:
            cols = st.columns(4); col_idx = 0
            sorted_keys = sorted(assets["scanner_fps"].keys())
            for i, key in enumerate(sorted_keys):
                fp_image = assets["scanner_fps"][key]
                norm_fp = normalize_image_display(fp_image)
                with cols[col_idx]: st.image(norm_fp, caption=f"{key} Fingerprint", use_container_width=True)
                col_idx = (col_idx + 1) % 4
        except Exception as e: st.error(f"Error displaying fingerprints: {e}")
    else: st.warning(f"Could not load fingerprints from '{FP_PATH}'. Cannot display fingerprint visualizations.")

    st.info("Feature-based EDA (like correlation heatmaps) is disabled as baseline features are specific and separate from hybrid features.")

# ---------------------------
# 5. Prediction Section (Corrected baseline confidence logic)
# ---------------------------
with tab1:
    st.header("Predict Scanner from Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["tif", "tiff", "png", "jpg", "jpeg"], key="uploader")

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        img_np = None # Initialize img_np
        with col1: # Display Uploaded Image
            st.subheader("Uploaded Image")
            try:
                image = Image.open(uploaded_file)
                img_np = np.array(image)
                if img_np.ndim == 3 and img_np.shape[2] == 3: img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                elif img_np.ndim == 3 and img_np.shape[2] == 4: img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
                st.image(image, caption=f"Original: {uploaded_file.name}", use_container_width=True)
            except Exception as e: st.error(f"Error reading image file: {e}")

        if img_np is not None: # Process and Predict only if image loaded successfully
            with st.spinner(f"Processing using {model_type} model..."):
                label = "Error"
                display_label = "Error" # Label to show user
                conf = 0.0
                res_display = np.zeros((*IMG_SIZE, 1), dtype=np.uint8) # Placeholder

                try:
                    # 1. Common Preprocessing: Grayscale and Resize
                    gray_resized = preprocess_image_common(img_np) # uint8

                    le_to_use = assets.get("hybrid_le") # Assumed common LE
                    if le_to_use is None: raise ValueError("Label Encoder failed to load.")

                    # 2. Extract Features & Predict based on selected model
                    if model_type == "Hybrid CNN":
                        res = calculate_residual(gray_resized) # Needs float input
                        res_display = normalize_image_display(res)

                        if not all([assets["hybrid_model"], assets["hybrid_scaler"], assets["scanner_fps"], assets["fp_keys"]]):
                            raise ValueError("Hybrid CNN assets not loaded correctly.")

                        x_img = np.expand_dims(res, axis=(0,-1))
                        x_ft_hybrid = extract_hybrid_features(res, assets["scanner_fps"], assets["fp_keys"])
                        x_ft_scaled = assets["hybrid_scaler"].transform(x_ft_hybrid)

                        prob = assets["hybrid_model"].predict([x_img, x_ft_scaled], verbose=0)
                        idx = int(np.argmax(prob))
                        label = le_to_use.classes_[idx] # Hybrid predicts base label
                        display_label = label
                        conf = float(prob[0, idx] * 100.0)

                    elif model_type == "Baseline":
                        if not all([assets["baseline_scaler"], assets["baseline_rf"] or assets["baseline_svm"]]):
                             raise ValueError("Required baseline assets not loaded correctly.")

                        x_ft_baseline = extract_baseline_features(gray_resized) # uint8 input
                        x_ft_scaled = assets["baseline_scaler"].transform(x_ft_baseline)

                        model_to_use = None
                        if baseline_model_choice == "Random Forest": model_to_use = assets["baseline_rf"]
                        elif baseline_model_choice == "SVM": model_to_use = assets["baseline_svm"]
                        if model_to_use is None: raise ValueError(f"{baseline_model_choice} model not loaded.")

                        # Baseline predicts detailed label (e.g., HP_150)
                        predicted_label_str_array = model_to_use.predict(x_ft_scaled)
                        label = predicted_label_str_array[0] # The detailed label

                        # --- Extract base brand for display and confidence lookup ---
                        try:
                             # Simple split, adjust if needed (e.g., if brand name itself has '_')
                             base_brand_label = label.split('_')[0]
                        except Exception:
                             base_brand_label = label # Fallback if split fails

                        display_label = base_brand_label # Show the user the base brand

                        # Calculate confidence using the base_brand_label
                        if hasattr(model_to_use, "predict_proba"):
                            prob_array = model_to_use.predict_proba(x_ft_scaled)[0]
                            # Find the index of the base_brand_label in the shared LE
                            try:
                                known_classes = list(le_to_use.classes_)
                                # Find index corresponding to the *base brand*
                                base_brand_idx = known_classes.index(base_brand_label)

                                # Sum probabilities of all classes belonging to this base brand?
                                # OR just take the probability of the most likely class WITHIN that brand?
                                # For simplicity, let's find the probability associated with the *exact predicted detailed label*
                                # if the model's classes_ attribute matches the LE's classes_
                                # --> This is complex because model.classes_ might be different from le_to_use.classes_
                                # --> SAFER APPROACH: Find index of BASE BRAND in LE and take max prob matching it?
                                # --> EVEN SIMPLER: Use the probability of the detailed class if possible, otherwise skip.

                                # Try getting index of the *detailed* label from model's internal classes
                                if hasattr(model_to_use, 'classes_'):
                                    model_classes = list(model_to_use.classes_)
                                    try:
                                        detailed_label_idx_in_model = model_classes.index(label)
                                        conf = float(prob_array[detailed_label_idx_in_model] * 100.0)
                                    except ValueError:
                                        st.warning(f"Predicted detailed label '{label}' not in model's classes list. Confidence unreliable.")
                                        conf = 0.0 # Cannot find confidence for the exact prediction
                                else:
                                    # Fallback: find base brand index in LE and use that probability
                                    # This might overestimate confidence if other resolutions had higher prob
                                    try:
                                        base_brand_idx_in_le = known_classes.index(base_brand_label)
                                        conf = float(prob_array[base_brand_idx_in_le] * 100.0)
                                        st.warning(f"Confidence shown is for the base brand '{base_brand_label}', not the specific prediction '{label}'.")
                                    except ValueError:
                                         st.warning(f"Base brand '{base_brand_label}' not found in Label Encoder. Confidence calculation failed.")
                                         conf = 0.0


                            except ValueError: # Base brand not found in LE
                                st.warning(f"Base brand '{base_brand_label}' (from '{label}') not found in Label Encoder. Confidence calculation failed.")
                                conf = 0.0
                            except Exception as conf_e:
                                st.warning(f"Error calculating confidence: {conf_e}")
                                conf = 0.0

                        elif baseline_model_choice == "SVM": conf = 100.0; st.warning("SVM confidence unavailable.")
                        else: conf = 100.0 # Default if predict_proba not available

                        # Calculate residual just for display consistency
                        try: res = calculate_residual(gray_resized); res_display = normalize_image_display(res)
                        except Exception: pass


                    # 3. Display Results
                    with col2:
                        st.subheader("Prediction Result")
                        st.info(f"Using: **{model_type}{f' ({baseline_model_choice})' if baseline_model_choice else ''}**")
                        if display_label != "Error":
                            st.success(f"**Predicted Scanner:** {display_label}") # Show base brand
                            st.metric(label="Confidence", value=f"{conf:.2f}%")
                            # Optionally show the detailed prediction if different
                            if model_type == "Baseline" and label != display_label:
                                st.caption(f"(Model predicted specific label: {label})")
                        else: st.error("Prediction failed.")
                        st.subheader("Extracted Noise Residual")
                        st.image(res_display, caption="Scanner Noise Pattern (Residual)", use_container_width=True)

                except ValueError as ve:
                    with col2: st.error(f"Processing Error: {ve}")
                except Exception as e:
                    with col2:
                        st.error(f"An unexpected error occurred: {e}")
                        import traceback
                        st.error(traceback.format_exc())
        else:
             st.error("Cannot proceed with prediction as image failed to load.")

