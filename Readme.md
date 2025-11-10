# 📊 AI TraceFinder — Scanner Source Identification

**Short version:** determine which physical scanner produced a scanned image by detecting device-specific micro-artifacts. Useful for forensic validation, authentication, and tamper detection.

---

## What this does
AI TraceFinder spots the tiny, machine-specific signatures scanners leave in images — sensor noise, compression quirks, texture patterns — and uses ML/CNN models to attribute a scan to a scanner model. Output includes predicted scanner, confidence score, and visual explainability (heatmaps / feature highlights).

---

## 🎯 Why it matters
- Proves whether a scanned document came from an authorized device  
- Flags suspicious or forged scans in audits and legal workflows  
- Provides traceable, explainable evidence for investigations

---

## 🧩 Quick features
- Automatic preprocessing pipeline (resize, grayscale, normalize, optional denoise)  
- Hybrid feature set: PRNU, FFT features, LBP texture descriptors, edge statistics  
- Baseline ML: Random Forest, SVM, Logistic Regression  
- Deep model: CNN trained on raw and augmented images  
- Explainability: Grad-CAM heatmaps and SHAP feature importance where applicable  
- Lightweight Streamlit UI for uploading images and getting fast predictions  
- Exportable CSVs for features and evaluation reports

---

##🛠 Tech stack

| Category | Technology | Purpose |
|-----------|-------------|----------|
| **Backend & ML** | **Python** | Core programming language |
| | **Scikit-learn** | Random Forest & SVM (Baseline Models) |
| | **Pandas** | Data manipulation and CSV handling |
| | **OpenCV** | Image processing (loading, color conversion, etc.) |
| | **NumPy** | Numerical operations |
| | **TensorFlow / Keras** | For CNN Model |
| **Frontend & UI** | **Streamlit** | Creating the interactive web application |
| | **Matplotlib & Seaborn** | Data visualization (confusion matrix, plots) |
| | **Pillow (PIL)** | Displaying sample images in the UI |
| **Tooling** | **Git & GitHub** | Version control and source management |
| | **venv** | Python virtual environment management |

---

## 📂 Dataset
Primary dataset: [NIST OpenMFC](https://www.nist.gov/) (scans from multiple scanner models at DPI settings such as 150/300/600). Local dataset collection recommended to match target scanners and environmental scanning differences.

---

## 🛠 How it works — pipeline
1. **Ingest:** read images, store metadata (dpi, resolution, scanner label).  
2. **Preprocess:** resize to fixed shape, convert to grayscale, normalize pixel range; optional denoising to emphasize sensor artifacts.  
3. **Feature extraction:** compute PRNU/noise residuals, FFT bands, LBP histograms, edge-based stats.  
4. **Train:** baseline ML on extracted features; CNN on raw/augmented images.  
5. **Explain:** produce Grad-CAM maps for CNN predictions and SHAP summaries for ML models.  
6. **Deploy:** Streamlit app exposes upload → predict path with downloadable reports.

---

## System architecture 
Input image → Preprocessing → Feature extractor & CNN backbone → Classifier (ML / DL) → Evaluator → Streamlit UI (predict + explain)
![System Architecture](./images/Architecture.png)

---

## Performance snapshot (example)
- CNN accuracy: **~85%** (on reported test split)  
- Weighted precision / recall / F1 ≈ **0.85**  
- Test set size (example): **~500 images**  
- Average reported model confidence: **~94%**

Performance will vary with dataset size, scanner diversity, scanning DPI, and preprocessing choices.

---

## Getting started (local)
Clone, venv, install, run:

```bash
git clone https://github.com/<username>/ai-tracefinder.git
cd ai-tracefinder

# create venv
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt

# run web app
streamlit run app.py
```

---

## 📁 Suggested Project Structure

```
tracefinderPred/
├── Data/                         # Raw dataset
├── models/                       # Trained ML models
├── pre_process/                  # Data preprocessing scripts
├── processed_data/               # Cleaned and processed datasets
├── results/                      # Model evaluation results
├── scr/(Baseline and CNN)        # Source code modules
├── venv/                         # Virtual environment
├── app.py                        # Main Streamlit application
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore file
├── LICENSE                       # Project license
└── Readme.md                     # Project documentation
```

---

## Usage examples

- **Forensics team**: upload questioned scan → check predicted scanner + Grad-CAM → export report for chain-of-custody

- **Compliance auditor**: bulk-run feature extraction on intake scans → check distribution shifts vs known authorized devices

- **R&D**: use feature CSVs and notebooks to iterate on classifiers

  ---
  

## Tips & caveats

- Model generalization needs representative data per target scanner.

- Environmental factors (lighting, paper type, scanning settings) affect signatures. Collect diverse samples.

- PRNU extraction benefits from multiple samples per device to average sensor noise.

  ---

## 📋 Prerequisites

Before running this project, ensure you have the following installed:

- Python 3.8 or higher
- pip (Python package installer)
- Git
- Virtual environment tool (venv or virtualenv)

  ---

## 📧 Contact

**Asmita Pathak**

- **Email:** asmitapathak2004@gmail.com
- **LinkedIn:** [linkedin.com/in/asmitapathak](https://www.linkedin.com/in/asmita-pathak-278447313/)
- **GitHub:** [github.com/asmitapathak1408](https://github.com/asmitapathak1408)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.





