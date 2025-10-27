import streamlit as st
import os
import cv2
import numpy as np
import pickle
import tensorflow as tf
import pywt
from skimage.feature import local_binary_pattern
from PIL import Image
import pandas as pd # Added for EDA
import plotly.express as px # Added for EDA
# Removed PCA import
import matplotlib.pyplot as plt # Added for EDA Fingerprint Viz and Heatmap
import seaborn as sns # Added for EDA Heatmap

# ---------------------------
# 1. Global Parameters & Paths
# ---------------------------
ART_DIR = "Residuals_Paths"
FP_PATH = "scanner_fingerprints.pkl" # In the root folder
ORDER_NPY = "fp_keys.npy"           # In the root folder
CKPT_PATH = os.path.join(ART_DIR, "scanner_hybrid.keras")
LE_PATH = os.path.join(ART_DIR, "hybrid_label_encoder.pkl")
SCALER_PATH = os.path.join(ART_DIR, "hybrid_feat_scaler.pkl")
# --- Path for EDA data (Features AND Labels) ---
FEATURES_LABELS_PATH = os.path.join(ART_DIR, "features.pkl") # Contains 'features' and 'labels'
IMG_SIZE = (256, 256)

# ---------------------------
# 2. Load Models & Assets
# ---------------------------
@st.cache_resource
def load_assets():
    """Loads model, encoder, scaler, fingerprints, keys, features, and labels."""
    model = None
    le_inf = None
    scaler_inf = None
    scanner_fps_inf = None
    fp_keys_inf = None
    features_for_eda = None
    labels_for_eda = None

    # Load Model, Encoder, Scaler, Fingerprints, Keys (with error handling)
    try: model = tf.keras.models.load_model(CKPT_PATH, compile=False)
    except Exception as e: st.error(f"Error loading Keras model: {e}"); st.stop()
    try:
        with open(LE_PATH, "rb") as f: le_inf = pickle.load(f)
    except Exception as e: st.error(f"Error loading label encoder: {e}"); st.stop()
    try:
        with open(SCALER_PATH, "rb") as f: scaler_inf = pickle.load(f)
    except Exception as e: st.error(f"Error loading scaler: {e}"); st.stop()
    try:
        with open(FP_PATH, "rb") as f: scanner_fps_inf = pickle.load(f)
    except Exception as e: st.error(f"Error loading fingerprints: {e}"); st.stop()
    try:
        fp_keys_inf = np.load(ORDER_NPY, allow_pickle=True).tolist()
    except Exception as e: st.error(f"Error loading fingerprint keys: {e}"); st.stop()

    # --- Load Features and Labels for EDA ---
    try:
        with open(FEATURES_LABELS_PATH, "rb") as f:
            data = pickle.load(f)
            features_for_eda = data.get('features') # Get the ZNCC features
            labels_for_eda = data.get('labels')
            if features_for_eda is None or labels_for_eda is None:
                 st.warning(f"Could not find 'features' or 'labels' key in {FEATURES_LABELS_PATH} for EDA.")
                 features_for_eda, labels_for_eda = None, None # Ensure both are None if one fails
    except FileNotFoundError:
        st.warning(f"Warning: Features/labels file for EDA not found at {FEATURES_LABELS_PATH}. EDA section may be limited.")
        features_for_eda, labels_for_eda = None, None
    except Exception as e:
        st.warning(f"Warning: Error loading features/labels for EDA: {e}")
        features_for_eda, labels_for_eda = None, None

    # Validate loaded EDA data
    if features_for_eda is not None and labels_for_eda is not None:
        if len(features_for_eda) != len(labels_for_eda):
            st.warning("Warning: Mismatch between number of features and labels loaded for EDA. Disabling EDA plots.")
            features_for_eda, labels_for_eda = None, None
        else:
             # Convert features to NumPy array for easier processing
            try:
                features_for_eda = np.array(features_for_eda, dtype=np.float32)
                # Check feature dimension (should match fp_keys length for ZNCC)
                if fp_keys_inf and features_for_eda.shape[1] != len(fp_keys_inf):
                     st.warning(f"Warning: EDA feature dimension ({features_for_eda.shape[1]}) doesn't match number of fingerprints ({len(fp_keys_inf)}). Assuming ZNCC features. Heatmap might be incorrect if using enhanced features.")
                     # Take only the first N features for heatmap if mismatch
                     features_for_eda = features_for_eda[:, :len(fp_keys_inf)]

            except Exception as e:
                st.warning(f"Could not convert EDA features to NumPy array: {e}")
                features_for_eda = None


    return model, le_inf, scaler_inf, scanner_fps_inf, fp_keys_inf, features_for_eda, labels_for_eda

st.set_page_config(layout="wide")
st.title("Trace Finder: Scanner Brand Identification")

# --- Sidebar ---
st.sidebar.title("App Information")

# Load assets safely
try:
    model, le, scaler, scanner_fps, fp_keys, train_features_eda, train_labels_eda = load_assets()
    st.sidebar.success("Model and assets loaded successfully.")
    if le: st.sidebar.write(f"Model trained on **{len(le.classes_)}** classes:"); st.sidebar.write(le.classes_)
    if train_labels_eda is not None: st.sidebar.info(f"EDA data loaded: {len(train_labels_eda)} samples.")
    else: st.sidebar.warning("EDA data could not be loaded or is invalid.")

except Exception as e:
    st.sidebar.error(f"Failed to load necessary assets: {e}")
    st.stop()

# --- Main App Sections ---
tab1, tab2 = st.tabs(["Scanner Prediction", "Data Overview (EDA)"])

# ---------------------------
# 3. Helper Functions (Unchanged)
# ---------------------------
# ... (corr2d, fft_radial_energy, lbp_hist_safe, make_feats_from_res, preprocess_residual_pywt_from_array functions remain exactly the same as the previous version) ...
def corr2d(a, b):
    """Computes the normalized cross-correlation between two 2D arrays."""
    a = a.astype(np.float32).ravel(); b = b.astype(np.float32).ravel()
    a -= a.mean(); b -= b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a @ b) / d) if d != 0 else 0.0

def fft_radial_energy(img, K=6):
    """Computes radial energy features from the FFT magnitude spectrum."""
    f = np.fft.fftshift(np.fft.fft2(img))
    mag = np.abs(f)
    h, w = mag.shape; cy, cx = h//2, w//2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    rmax = r.max() + 1e-6
    bins = np.linspace(0, rmax, K+1)
    feats = []
    for i in range(K):
        m = (r >= bins[i]) & (r < bins[i+1])
        feats.append(float(mag[m].mean() if m.any() else 0.0))
    return feats

def lbp_hist_safe(img, P=8, R=1.0):
    """Computes the LBP histogram safely, handling flat images."""
    rng = float(np.ptp(img))
    if rng < 1e-12: # Avoid division by zero for flat images
        return [0.0] * (P + 2)
    else:
        g = (img - float(np.min(img))) / (rng + 1e-8)
    g8 = (g * 255.0).astype(np.uint8)
    codes = local_binary_pattern(g8, P=P, R=R, method="uniform")
    n_bins = P + 2
    hist, _ = np.histogram(codes, bins=np.arange(n_bins+1), range=(0, n_bins), density=True)
    return hist.astype(np.float32).tolist()

def make_feats_from_res(res, loaded_scanner_fps, loaded_fp_keys, loaded_scaler):
    """Generates the handcrafted feature vector from a residual image."""
    v_corr = [corr2d(res, loaded_scanner_fps[k]) for k in loaded_fp_keys]
    v_fft  = fft_radial_energy(res, K=6)
    v_lbp  = lbp_hist_safe(res, P=8, R=1.0)
    if len(v_lbp) != 10:
         st.warning(f"LBP histogram length is {len(v_lbp)}, expected 10. Padding with zeros.")
         v_lbp.extend([0.0] * (10 - len(v_lbp)))
    v = np.array(v_corr + v_fft + v_lbp, dtype=np.float32)
    if v.shape[0] != 27:
         st.error(f"Feature vector has incorrect length: {v.shape[0]}, expected 27.")
         raise ValueError("Incorrect feature vector length.")
    v = v.reshape(1,-1)
    v = loaded_scaler.transform(v) # Use the scaler loaded for inference
    return v

def preprocess_residual_pywt_from_array(img_array):
    """Preprocesses an image array to extract the wavelet residual."""
    if img_array is None: raise ValueError("Input image array is None")
    if img_array.ndim == 3: img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    elif img_array.ndim == 2: img = img_array
    else: raise ValueError(f"Unsupported image dimensions: {img_array.ndim}")
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    cA, (cH, cV, cD) = pywt.dwt2(img, 'haar')
    cH.fill(0); cV.fill(0); cD.fill(0)
    den = pywt.idwt2((cA, (cH, cV, cD)), 'haar')
    if den.shape != img.shape: den = cv2.resize(den, img.shape[::-1], interpolation=cv2.INTER_LINEAR)
    return (img - den).astype(np.float32)

# --- Function to normalize fingerprint for display ---
def normalize_image_display(img):
    """Normalize image array to 0-255 uint8 for display."""
    norm_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return norm_img.astype(np.uint8)

# ---------------------------
# 4. EDA Section (Simplified)
# ---------------------------
with tab2:
    st.header("Training Data Overview")

    # Check if necessary data for EDA is loaded
    eda_ready = (train_labels_eda is not None and
                 train_features_eda is not None and
                 scanner_fps is not None and
                 fp_keys is not None)

    if eda_ready:
        try:
            # --- 1. Class Distribution (Bar Chart / Countplot) ---
            st.subheader("1. Distribution of Scanner Brands")
            label_df = pd.DataFrame(train_labels_eda, columns=['scanner_brand'])
            scanner_counts = label_df['scanner_brand'].value_counts().reset_index()
            scanner_counts.columns = ['Scanner Brand', 'Number of Images']
            fig_bar = px.bar(scanner_counts, x='Scanner Brand', y='Number of Images',
                             title='Images per Scanner Brand in Training Data',
                             text='Number of Images', color='Scanner Brand', template='plotly_white')
            fig_bar.update_traces(textposition='outside')
            fig_bar.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_bar, use_container_width=True) # <<< This plotly chart already uses the correct parameter
            with st.expander("Show Raw Counts"):
                st.dataframe(scanner_counts, use_container_width=True) # <<< This dataframe already uses the correct parameter

            # --- 2. Scanner Fingerprint Visualization ---
            st.subheader("2. Average Scanner Fingerprints (Noise Patterns)")
            st.write("Average residual noise patterns from the Flatfield dataset.")
            cols = st.columns(4) # Adjust number of columns as needed
            col_idx = 0
            sorted_keys = sorted(scanner_fps.keys())
            for i, key in enumerate(sorted_keys):
                fp_image = scanner_fps[key]
                norm_fp = normalize_image_display(fp_image)
                with cols[col_idx]:
                    # --- CHANGE HERE ---
                    st.image(norm_fp, caption=f"{key} Fingerprint", use_container_width=True) # Changed from use_column_width
                col_idx = (col_idx + 1) % 4

            # --- 3. Feature Correlation Heatmap (ZNCC Features) ---
            num_zncc_features = len(fp_keys)
            if train_features_eda.shape[1] >= num_zncc_features:
                st.subheader("3. ZNCC Feature Correlation Heatmap")
                st.write("Shows the correlation between the different fingerprint correlation (ZNCC) features.")

                # Create DataFrame with only ZNCC features
                zncc_feature_names = [f"Corr_{key}" for key in fp_keys]
                zncc_df = pd.DataFrame(train_features_eda[:, :num_zncc_features], columns=zncc_feature_names)

                # Calculate correlation matrix
                corr_matrix = zncc_df.corr()

                # Plot heatmap
                fig_heatmap, ax = plt.subplots(figsize=(10, 8)) # Use matplotlib for heatmap
                sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt=".2f", ax=ax, cbar=True)
                ax.set_title('Correlation Matrix of ZNCC Features')
                ax.tick_params(axis='x', rotation=90)
                ax.tick_params(axis='y', rotation=0)
                plt.tight_layout() # Adjust layout
                st.pyplot(fig_heatmap) # Display matplotlib plot in Streamlit

                with st.expander("Show Correlation Values"):
                    st.dataframe(corr_matrix)
            else:
                st.warning("Cannot generate heatmap: Feature dimension doesn't match fingerprint count.")


        except Exception as e:
            st.error(f"An error occurred while generating EDA plots: {e}")
            import traceback
            st.error(traceback.format_exc())

    else:
        st.warning(f"Could not load valid training features/labels from '{FEATURES_LABELS_PATH}'. Cannot display EDA visualizations.")

# ---------------------------
# 5. Prediction Section (Unchanged)
# ---------------------------
with tab1:
    st.header("Predict Scanner from Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["tif", "tiff", "png", "jpg", "jpeg"], key="uploader")

    if uploaded_file is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uploaded Image")
            try:
                image = Image.open(uploaded_file)
                img_np = np.array(image)
                if img_np.ndim == 3 and img_np.shape[2] == 3: img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                elif img_np.ndim == 3 and img_np.shape[2] == 4: img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
                st.image(image, caption=f"Original: {uploaded_file.name}", use_container_width=True)
            except Exception as e:
                st.error(f"Error reading image file: {e}")
                st.stop()

        if 'img_np' in locals():
            with st.spinner("Analyzing scanner fingerprint... Please wait."):
                try:
                    # 1. Preprocess and Feature Extraction
                    res = preprocess_residual_pywt_from_array(img_np)
                    x_img = np.expand_dims(res, axis=(0,-1))
                    # Pass loaded assets explicitly to make_feats_from_res
                    x_ft = make_feats_from_res(res, scanner_fps, fp_keys, scaler)

                    # 2. Predict
                    if model is None: st.error("Model not loaded."); st.stop()
                    prob = model.predict([x_img, x_ft], verbose=0)

                    # 3. Decode
                    idx = int(np.argmax(prob))
                    if le is None: st.error("Label encoder not loaded."); st.stop()
                    label = le.classes_[idx]
                    conf = float(prob[0, idx] * 100.0)

                    # --- MOVED DISPLAY CODE INSIDE TRY BLOCK ---
                    # 4. Display Results
                    with col2:
                        st.subheader("Prediction Result")
                        st.success(f"**Predicted Scanner:** {label}")
                        st.metric(label="Confidence", value=f"{conf:.2f}%")
                        st.subheader("Extracted Noise Residual")
                        res_display = normalize_image_display(res) # Use helper
                        st.image(res_display, caption="Scanner Noise Pattern (Residual)", use_container_width=True)
                    # --- END OF MOVED CODE ---

                except ValueError as ve:
                    # Display error in the second column if prediction fails
                    with col2:
                        st.error(f"Processing Error: {ve}")
                except Exception as e:
                    # Display error in the second column if prediction fails
                    with col2:
                        st.error(f"An unexpected error occurred during prediction: {e}")
                        import traceback
                        st.error(traceback.format_exc()) # Show detailed traceback for debugging



