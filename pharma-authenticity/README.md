# 💊 Pharmaceutical Authenticity & Supply Chain Security System

> AI-powered detection of counterfeit medicines and anomalies in drug distribution using supervised learning, anomaly detection, and deep learning.

---

## 📌 Overview

Counterfeit medicines are a global health crisis — the WHO estimates **1 in 10 medical products** in low- and middle-income countries is substandard or falsified. This project builds a machine learning pipeline to:

- **Classify** pharmaceutical products as genuine or counterfeit using supply chain metadata
- **Detect anomalies** in drug distribution patterns that may indicate fraud or diversion
- **Authenticate packaging** through CNN-based visual pattern recognition

---

## 🧠 Models Used

| Model | Type | Task |
|---|---|---|
| **Random Forest** | Supervised Classification | Genuine vs. Counterfeit detection |
| **Isolation Forest** | Unsupervised Anomaly Detection | Suspicious distribution patterns |
| **CNN (ConvNet)** | Deep Learning | Packaging image authenticity |

---

## 📁 Project Structure

```
pharma-authenticity/
│
├── data/
│   └── generate_dataset.py          # Synthetic supply chain & anomaly data generator
│
├── preprocessing/
│   └── feature_engineering.py       # Cleaning, encoding, scaling, feature creation
│
├── models/
│   ├── random_forest.py             # RF classifier with cross-validation
│   ├── isolation_forest.py          # Isolation Forest anomaly detector
│   └── cnn_model.py                 # CNN for packaging image classification
│
├── evaluation/
│   └── cross_validation.py          # K-fold CV, metrics, comparison plots
│
├── images/
│   └── packaging_samples/           # Generated/real packaging images
│       ├── genuine/
│       └── counterfeit/
│
├── main.py                          # 🚀 Run full pipeline here
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/pharma-authenticity.git
cd pharma-authenticity
```

### 2. Create a Virtual Environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note:** TensorFlow (for CNN) is optional. If you don't need CNN, comment it out in `requirements.txt`.

---

## 🚀 How to Run

### Run Full Pipeline
```bash
python main.py
```

### Skip CNN (faster, no TensorFlow needed)
```bash
python main.py --no-cnn
```

### Run Individual Modules
```bash
# Generate data only
python data/generate_dataset.py

# Train Random Forest only
python models/random_forest.py

# Train Isolation Forest only
python models/isolation_forest.py

# Train CNN only
python models/cnn_model.py

# Evaluation & comparison
python evaluation/cross_validation.py
```

---

## 🔬 Feature Engineering

The preprocessing pipeline creates engineered features from raw supply chain data:

| Feature | Description |
|---|---|
| `trust_score` | Weighted composite of barcode, seal, expiry, distributor, lot number validity |
| `cold_chain_ok` | Binary flag: temperature 2–8°C AND humidity 30–60% |
| `price_suspicious` | Flag if price deviates more than 25% below market |
| `supplier_risk` | Flag for unknown/unregistered supplier IDs |
| `seal_barcode_combo` | Interaction: seal intact × barcode valid |
| `delivery_efficiency` | order_quantity / delivery_time (anomaly data) |
| `cost_per_unit` | invoice_amount / order_quantity (anomaly data) |
| `risk_score` | Weighted combo of return rate, complaints, route deviation |

---

## 📊 Model Details

### Random Forest Classifier
- **200 estimators**, max depth 12
- `class_weight='balanced'` to handle class imbalance (85% genuine / 15% counterfeit)
- 5-fold Stratified Cross-Validation
- Outputs: Confusion Matrix, ROC Curve, Feature Importances

### Isolation Forest
- **200 trees**, contamination = 6%
- Trained on distribution transaction features
- PCA 2D visualization of detected anomalies
- Manual stratified cross-validation for evaluation

### CNN Architecture
```
Input (64×64×3)
  → Conv2D(32) + BN + MaxPool + Dropout
  → Conv2D(64) + BN + MaxPool + Dropout
  → Conv2D(128) + BN + MaxPool + Dropout
  → GlobalAveragePooling2D
  → Dense(128) + Dropout
  → Dense(1, sigmoid)
```
- Data augmentation: rotation, shifts, horizontal flip
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

---

## 📈 Output Files

After running the pipeline, the `evaluation/` folder contains:

| File | Description |
|---|---|
| `random_forest_results.png` | Confusion matrix, ROC curve, feature importances |
| `isolation_forest_results.png` | PCA scatter plots, anomaly score distribution |
| `cnn_training_history.png` | Accuracy and loss curves over epochs |
| `model_comparison.png` | Side-by-side metric comparison bar chart |
| `evaluation_report.csv` | Numeric results table (mean ± std per metric) |

---

## 🔮 Potential Improvements

- **Real Datasets**: Replace synthetic data with actual pharmaceutical supply chain datasets (e.g., WHO, FDA databases)
- **SMOTE**: Apply oversampling for better handling of class imbalance
- **Transfer Learning**: Use pretrained CNNs (ResNet, EfficientNet) for packaging recognition
- **Explainability**: Add SHAP values for Random Forest predictions
- **Deployment**: Wrap models in a FastAPI REST endpoint
- **Barcode Integration**: Add real GS1 barcode parsing and validation

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **scikit-learn** — Random Forest, Isolation Forest, cross-validation
- **TensorFlow / Keras** — CNN model
- **pandas / numpy** — Data manipulation
- **matplotlib / seaborn** — Visualizations
- **joblib** — Model serialization

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 👤 Author

Built as part of an AI/ML research project on pharmaceutical supply chain integrity.
