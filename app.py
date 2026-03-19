"""
app.py — Streamlit Interactive Web Application
===============================================
Smartphone Sensor Usage Analyser & Overheating Predictor

Run:
    streamlit run app.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartSense ML — Smartphone Health Monitor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] { background: #1a1a2e; }
    section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    section[data-testid="stSidebar"] .stSlider label { color: #aab2d3 !important; }

    /* ── Header ── */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem 1.5rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .main-header h1 { font-size: 2rem; font-weight: 700; margin: 0 0 0.25rem; }
    .main-header p  { color: #a8b2d8; font-size: 0.95rem; margin: 0; }

    /* ── Alert boxes ── */
    .alert-green {
        background: linear-gradient(135deg, #0d4c32, #1a7a4e);
        border-left: 5px solid #2ecc71;
        padding: 1.2rem 1.5rem; border-radius: 10px; color: #d4f5e5;
    }
    .alert-orange {
        background: linear-gradient(135deg, #4a2c00, #7a4a00);
        border-left: 5px solid #f39c12;
        padding: 1.2rem 1.5rem; border-radius: 10px; color: #fde9c5;
    }
    .alert-red {
        background: linear-gradient(135deg, #4a0000, #7a0000);
        border-left: 5px solid #e74c3c;
        padding: 1.2rem 1.5rem; border-radius: 10px; color: #ffc5c5;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0%,100% { box-shadow: 0 0 0 0 rgba(231,76,60,0.4); }
        50%      { box-shadow: 0 0 0 8px rgba(231,76,60,0); }
    }

    /* ── Metric cards ── */
    .metric-card {
        background: #1e2a3a; border-radius: 12px;
        padding: 1rem 1.2rem; text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    .metric-card h3 { color: #a8b2d8; font-size: 0.8rem; margin-bottom: 0.3rem; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card p  { font-size: 1.6rem; font-weight: 700; margin: 0; }

    /* ── Tip items ── */
    .tip-item {
        background: #1e2a3a; border-radius: 8px;
        padding: 0.6rem 1rem; margin: 0.3rem 0;
        border-left: 3px solid #3498db; color: #d0d8f0;
        font-size: 0.9rem;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab"] { font-size: 0.9rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────
# LOAD MODELS & ARTEFACTS
# ─────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_artefacts():
    base = "models"
    artefacts = {}
    required = ["best_model.pkl", "scaler.pkl", "feature_cols.pkl", "label_map.pkl"]
    for f in required:
        path = os.path.join(base, f)
        if not os.path.exists(path):
            return None
        artefacts[f.replace(".pkl", "")] = joblib.load(path)

    # Optional SHAP explainer
    shap_path = os.path.join(base, "shap_explainer.pkl")
    artefacts["shap_explainer"] = joblib.load(shap_path) if os.path.exists(shap_path) else None

    # Optional model comparison table
    csv_path = os.path.join(base, "model_comparison.csv")
    artefacts["comparison_df"] = pd.read_csv(csv_path, index_col=0) if os.path.exists(csv_path) else None

    return artefacts


artefacts = load_artefacts()
MODELS_READY = artefacts is not None

# ─────────────────────────────────────────────────────────────────────────
# HELPER — run prediction
# ─────────────────────────────────────────────────────────────────────────
def predict(input_dict: dict):
    feature_cols = artefacts["feature_cols"]
    scaler       = artefacts["scaler"]
    model        = artefacts["best_model"]
    label_map    = artefacts["label_map"]

    X = np.array([[input_dict[c] for c in feature_cols]])
    X_sc = scaler.transform(X)

    pred  = model.predict(X_sc)[0]
    label = label_map[pred]

    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_sc)[0]

    return pred, label, proba, X_sc


def get_shap_values(X_sc):
    explainer = artefacts.get("shap_explainer")
    if explainer is None:
        return None
    try:
        shap_vals = explainer.shap_values(X_sc)
        return shap_vals
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>📱 SmartSense ML — Smartphone Health Monitor</h1>
  <p>Real-time prediction of smartphone usage patterns and overheating risk using multi-sensor data analysis</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# SIDEBAR — INPUT PANEL
# ─────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🎛️ Sensor Input Panel")
st.sidebar.markdown("Adjust the sliders to match your current device readings.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Activity Sensors")
screen_time      = st.sidebar.slider("🖥️ Screen Time (min/hr)",      0.0, 60.0, 20.0, 0.5)
motion_activity  = st.sidebar.slider("🏃 Motion Activity (0–10)",     0.0, 10.0,  3.0, 0.1)
touch_frequency  = st.sidebar.slider("👆 Touch Frequency (events/min)", 0.0, 30.0, 5.0, 0.5)

st.sidebar.markdown("### ⚡ Power & Performance")
battery_consumption = st.sidebar.slider("🔋 Battery Consumption (%/hr)", 0.0, 50.0, 8.0, 0.5)
cpu_usage           = st.sidebar.slider("💻 CPU Usage (%)",              0.0, 100.0, 25.0, 1.0)
device_activity     = st.sidebar.slider("⚙️ Device Activity (0–10)",    0.0, 10.0,   3.0, 0.1)

st.sidebar.markdown("### 🌡️ Thermal")
temperature = st.sidebar.slider("🌡️ Device Temperature (°C)",  20.0, 65.0, 30.0, 0.5)

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🔍  **Analyse Now**", use_container_width=True, type="primary")

# ─────────────────────────────────────────────────────────────────────────
# MAIN CONTENT TABS
# ─────────────────────────────────────────────────────────────────────────
tab_pred, tab_models, tab_eda, tab_about = st.tabs([
    "📊  Prediction & Alerts",
    "🏆  Model Performance",
    "📈  EDA Visualisations",
    "ℹ️  About"
])

# ════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ════════════════════════════════════════════════════════════════════════
with tab_pred:
    if not MODELS_READY:
        st.warning("⚠️ Models not found. Please run `python main.py` first.")
        st.code("python main.py", language="bash")
        st.stop()

    input_data = {
        "screen_time":          screen_time,
        "motion_activity":      motion_activity,
        "touch_frequency":      touch_frequency,
        "battery_consumption":  battery_consumption,
        "cpu_usage":            cpu_usage,
        "temperature":          temperature,
        "device_activity":      device_activity,
    }

    # Auto-predict on slider change OR button click
    pred_class, pred_label, proba, X_sc = predict(input_data)

    # ── Confidence gauge ─────────────────────────────────────────────
    st.markdown("### Current Sensor Readings")
    col1, col2, col3, col4 = st.columns(4)
    readings = [
        ("Screen Time", f"{screen_time:.0f} min/hr", "#3498db"),
        ("CPU Usage",   f"{cpu_usage:.0f} %",        "#e67e22"),
        ("Temperature", f"{temperature:.1f} °C",     "#e74c3c"),
        ("Battery",     f"{battery_consumption:.0f} %/hr", "#2ecc71"),
    ]
    for col, (title, val, color) in zip([col1, col2, col3, col4], readings):
        col.markdown(f"""
        <div class="metric-card">
            <h3>{title}</h3>
            <p style="color:{color}">{val}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Prediction result ─────────────────────────────────────────────
    st.markdown("### 🎯 Prediction Result")

    color_map = {"Normal": "#2ecc71", "Heavy": "#f39c12", "Overheating": "#e74c3c"}
    icon_map  = {"Normal": "✅", "Heavy": "⚠️", "Overheating": "🔴"}

    col_res, col_proba = st.columns([1, 1])

    with col_res:
        pcolor = color_map[pred_label]
        st.markdown(f"""
        <div style="background:#1e2a3a;border-radius:14px;padding:2rem;text-align:center;
                    border: 2px solid {pcolor}; box-shadow: 0 0 20px {pcolor}44;">
            <div style="font-size:3.5rem">{icon_map[pred_label]}</div>
            <div style="font-size:2rem;font-weight:700;color:{pcolor};margin-top:0.5rem">
                {pred_label}
            </div>
            <div style="color:#a8b2d8;font-size:0.9rem;margin-top:0.5rem">
                Usage Level Detected
            </div>
        </div>""", unsafe_allow_html=True)

    with col_proba:
        if proba is not None:
            labels = ["Normal", "Heavy", "Overheating"]
            colors = ["#2ecc71", "#f39c12", "#e74c3c"]
            fig, ax = plt.subplots(figsize=(5, 3))
            fig.patch.set_facecolor("#1e2a3a")
            ax.set_facecolor("#1e2a3a")
            bars = ax.barh(labels, proba, color=colors, alpha=0.85, height=0.5)
            for bar, p in zip(bars, proba):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        f"{p:.1%}", va="center", color="white", fontsize=10)
            ax.set_xlim(0, 1.18)
            ax.set_xlabel("Confidence", color="white")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#334")
            ax.set_title("Prediction Confidence", color="white", fontsize=11, fontweight="bold")
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Probability scores not available for this model.")

    st.markdown("---")

    # ── Alerts & Recommendations ──────────────────────────────────────
    st.markdown("### 🚨 Alerts & Recommendations")
    from utils import get_alerts_and_recommendations
    alert_info = get_alerts_and_recommendations(pred_class, input_data)

    alert_html = f"""
    <div class="alert-{alert_info['alert_level']}">
        <h3 style="margin:0 0 0.4rem">{alert_info['alert_title']}</h3>
        <p style="margin:0;opacity:0.9">{alert_info['alert_msg']}</p>
    </div>"""
    st.markdown(alert_html, unsafe_allow_html=True)

    st.markdown("#### 💡 Actionable Recommendations")
    for tip in alert_info["tips"]:
        st.markdown(f'<div class="tip-item">• {tip}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── SHAP Explanation ──────────────────────────────────────────────
    st.markdown("### 🔍 Explainable AI — Feature Influence")

    shap_vals = get_shap_values(X_sc)
    feature_cols = artefacts["feature_cols"]

    if shap_vals is not None:
        try:
            import shap
            sv = shap_vals[pred_class][0]          # shape: (n_features,)
            contrib = pd.Series(sv, index=feature_cols).sort_values()

            fig, ax = plt.subplots(figsize=(7, 4))
            fig.patch.set_facecolor("#1e2a3a")
            ax.set_facecolor("#1e2a3a")
            bar_colors = ["#e74c3c" if v > 0 else "#3498db" for v in contrib]
            contrib.plot(kind="barh", ax=ax, color=bar_colors, edgecolor="none")
            ax.axvline(0, color="white", linewidth=0.8)
            ax.set_title(f"SHAP Values — {pred_label} prediction",
                         color="white", fontsize=12, fontweight="bold")
            ax.tick_params(colors="white")
            ax.set_xlabel("SHAP Value (impact on output)", color="white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#334")
            st.pyplot(fig)
            plt.close()

            st.caption("🔴 Red bars push toward this prediction | 🔵 Blue bars push away from it")
        except Exception as e:
            st.info(f"SHAP visualisation unavailable: {e}")
    else:
        # Fallback: feature importance from RF if available
        model = artefacts["best_model"]
        if hasattr(model, "feature_importances_"):
            fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values()
            fig, ax = plt.subplots(figsize=(7, 4))
            fig.patch.set_facecolor("#1e2a3a")
            ax.set_facecolor("#1e2a3a")
            fi.plot(kind="barh", ax=ax, color="#9b59b6", edgecolor="none")
            ax.set_title("Feature Importances (Model-Based)",
                         color="white", fontsize=12, fontweight="bold")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#334")
            st.pyplot(fig)
            plt.close()
        else:
            st.info("SHAP explainer not found. Run `python main.py` with SHAP installed.")

    # ── Input summary table ───────────────────────────────────────────
    with st.expander("📋 Full Input Summary"):
        inp_df = pd.DataFrame([input_data]).T
        inp_df.columns = ["Value"]
        inp_df["Status"] = inp_df.apply(lambda row: (
            "⚠️ High"   if (row.name == "cpu_usage"    and row["Value"] > 80) or
                           (row.name == "temperature"  and row["Value"] > 45) or
                           (row.name == "screen_time"  and row["Value"] > 50) else
            "🟡 Medium" if (row.name == "cpu_usage"    and row["Value"] > 60) or
                           (row.name == "temperature"  and row["Value"] > 38) else
            "✅ Normal"
        ), axis=1)
        st.dataframe(inp_df, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════
with tab_models:
    st.markdown("### 🏆 Model Performance Summary")

    if MODELS_READY and artefacts["comparison_df"] is not None:
        comp_df = artefacts["comparison_df"]

        # Highlight best model
        st.dataframe(
            comp_df.style
                .background_gradient(cmap="Blues", subset=["Accuracy","F1-Score"])
                .format("{:.4f}"),
            use_container_width=True
        )

        # Bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#1e2a3a")
        ax.set_facecolor("#1e2a3a")
        x      = np.arange(len(comp_df))
        width  = 0.2
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
        clrs    = ["#3498db", "#2ecc71", "#e67e22", "#e74c3c"]
        for i, (m, c) in enumerate(zip(metrics, clrs)):
            ax.bar(x + i*width, comp_df[m], width, label=m, color=c, alpha=0.85)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(comp_df.index, rotation=20, ha="right",
                           color="white", fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.tick_params(colors="white")
        ax.set_title("Model Comparison", color="white", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, labelcolor="white", facecolor="#1e2a3a")
        for spine in ax.spines.values():
            spine.set_edgecolor("#334")
        st.pyplot(fig)
        plt.close()
    else:
        st.info("Run `python main.py` to generate model comparison data.")

    # Show saved EDA plots
    st.markdown("### 📊 Saved Evaluation Plots")
    plot_files = {
        "Confusion Matrices": [f for f in os.listdir("plots") if f.startswith("cm_")]
                               if os.path.exists("plots") else [],
        "ROC Curves":         ["10_roc_curves.png"],
        "Model Comparison":   ["11_model_comparison.png"],
        "SHAP":               ["12_shap_summary.png"],
    }
    for section, fnames in plot_files.items():
        for fname in fnames:
            path = os.path.join("plots", fname)
            if os.path.exists(path):
                st.markdown(f"**{section} — {fname}**")
                st.image(path, use_column_width=True)


# ════════════════════════════════════════════════════════════════════════
# TAB 3 — EDA VISUALISATIONS
# ════════════════════════════════════════════════════════════════════════
with tab_eda:
    st.markdown("### 📈 Exploratory Data Analysis")

    eda_plots = [
        ("01_class_distribution.png",    "Class Distribution"),
        ("02_feature_distributions.png", "Feature Distributions by Usage Level"),
        ("03_correlation_heatmap.png",   "Correlation Heatmap"),
        ("04_boxplots.png",              "Box Plots"),
        ("05_pairplot.png",              "Pair Plot"),
        ("06_temp_vs_battery.png",       "Temperature vs Battery Consumption"),
        ("07_pca_2d.png",                "PCA 2-D Projection"),
        ("08_selectkbest.png",           "SelectKBest ANOVA F-Scores"),
        ("09_pca_variance.png",          "PCA Explained Variance"),
    ]

    cols = st.columns(2)
    for i, (fname, title) in enumerate(eda_plots):
        path = os.path.join("plots", fname)
        with cols[i % 2]:
            if os.path.exists(path):
                st.markdown(f"**{title}**")
                st.image(path, use_column_width=True)
            else:
                st.info(f"Plot not found: {fname}. Run `python main.py` first.")


# ════════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
    ### 📱 About SmartSense ML

    **SmartSense ML** is an end-to-end machine learning system that analyses smartphone
    sensor data to predict device usage levels and potential overheating conditions.

    ---

    #### 🔬 How It Works
    1. **Sensors monitored**: Screen time, motion activity, touch frequency, battery consumption, CPU usage, device temperature, background activity
    2. **Preprocessing**: Missing value imputation, IQR-based outlier capping, StandardScaler normalisation
    3. **Feature selection**: SelectKBest (ANOVA), RFE, PCA
    4. **Models trained**: Logistic Regression, Decision Tree, Random Forest, SVM, Neural Network (MLP)
    5. **Best model** is selected by weighted F1-score and saved to `models/best_model.pkl`
    6. **Explainability**: SHAP values highlight which features drove each prediction
    7. **Alerts**: Rule-based + model-based alert system with actionable recommendations

    ---

    #### 📊 Output Classes
    | Class | Description | Typical CPU | Typical Temp |
    |-------|-------------|-------------|--------------|
    | ✅ Normal     | Safe usage zone          | < 40 %  | < 35 °C |
    | ⚠️ Heavy      | High load, watch out     | 40–80 % | 35–45 °C |
    | 🔴 Overheating | Critical — act now      | > 80 %  | > 45 °C |

    ---

    #### 🛠️ Tech Stack
    `Python` · `scikit-learn` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `SHAP` · `Streamlit` · `joblib`

    ---

    #### 🚀 Getting Started
    ```bash
    # 1. Install dependencies
    pip install -r requirements.txt

    # 2. Train models
    python main.py

    # 3. Launch app
    streamlit run app.py
    ```
    """)
