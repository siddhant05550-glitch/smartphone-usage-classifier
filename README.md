# 📱 SmartSense ML — Smartphone Health Monitor

An end-to-end machine learning system that analyses smartphone sensor data to predict
**usage level** (Normal / Heavy / Overheating) and surface actionable recommendations.

---

## 🗂️ Project Structure

```
smartphone_ml_project/
│
├── main.py              # Full ML training pipeline
├── app.py               # Streamlit interactive web app
├── utils.py             # Shared helper functions
├── requirements.txt     # Python dependencies
├── README.md            # This file
│
├── models/              # Auto-created by main.py
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── shap_explainer.pkl
│   ├── feature_cols.pkl
│   ├── label_map.pkl
│   ├── model_comparison.csv
│   └── processed_data.csv
│
└── plots/               # Auto-created by main.py
    ├── 01_class_distribution.png
    ├── 02_feature_distributions.png
    ├── 03_correlation_heatmap.png
    ├── 04_boxplots.png
    ├── 05_pairplot.png
    ├── 06_temp_vs_battery.png
    ├── 07_pca_2d.png
    ├── 08_selectkbest.png
    ├── 09_pca_variance.png
    ├── 10_roc_curves.png
    ├── 11_model_comparison.png
    ├── 12_shap_summary.png
    └── cm_*.png          (per-model confusion matrices)
```

---

## 🚀 Quick Start

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Python 3.8+** recommended. For a clean environment:
> ```bash
> python -m venv venv
> source venv/bin/activate   # Windows: venv\Scripts\activate
> pip install -r requirements.txt
> ```

### Step 2 — Train models

```bash
python main.py
```

This will:
- Generate a 2,000-sample synthetic dataset
- Perform preprocessing and EDA
- Train 5 classifiers with 5-fold cross-validation
- Save the best model and all artefacts to `models/`
- Save 13+ visualisation plots to `plots/`

### Step 3 — Launch the web app

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 🔬 Features

| Component | Details |
|-----------|---------|
| **Data** | 2,000-sample synthetic dataset with 7 sensor features |
| **Preprocessing** | Median imputation, IQR outlier capping, StandardScaler |
| **EDA** | 9 visualisation plots including pair plots, correlation heatmap, PCA |
| **Feature Selection** | SelectKBest (ANOVA F), RFE (Random Forest), PCA variance analysis |
| **Models** | Logistic Regression, Decision Tree, Random Forest, SVM, MLP |
| **Evaluation** | Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix, ROC curves |
| **XAI** | SHAP TreeExplainer values per prediction |
| **Alerts** | Rule-based + model-based with actionable tips |
| **App** | Dark-themed Streamlit app with live sliders and charts |

---

## 📊 Sensor Features

| Feature | Unit | Description |
|---------|------|-------------|
| `screen_time` | min/hr | Active screen use per hour |
| `motion_activity` | 0–10 | Accelerometer / gyroscope score |
| `touch_frequency` | events/min | Touch event rate |
| `battery_consumption` | %/hr | Battery drain rate |
| `cpu_usage` | % | CPU utilisation |
| `temperature` | °C | Device surface temperature |
| `device_activity` | 0–10 | Background process score |

---

## 🏷️ Output Classes

| Label | Code | Typical Profile |
|-------|------|-----------------|
| ✅ Normal | 0 | CPU < 40 %, Temp < 35 °C |
| ⚠️ Heavy  | 1 | CPU 40–80 %, Temp 35–45 °C |
| 🔴 Overheating | 2 | CPU > 80 %, Temp > 45 °C |

---

## 🛠️ Tech Stack

- **ML**: scikit-learn, numpy, pandas
- **Visualisation**: matplotlib, seaborn
- **XAI**: SHAP
- **App**: Streamlit
- **Persistence**: joblib

---

## 📌 Notes

- SHAP requires `pip install shap` and is optional; the app falls back to built-in feature importances.
- To use your own dataset, replace the `generate_dataset()` call in `main.py` with a `pd.read_csv()` call and ensure it has the same 7 feature columns plus a `usage_level` column (0/1/2).
