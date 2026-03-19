"""
utils.py — Helper Functions for Smartphone Sensor ML Project
============================================================
Contains utilities for data generation, preprocessing, evaluation,
and explainability used across main.py and app.py.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, accuracy_score, precision_score, recall_score, f1_score
)
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1.  SYNTHETIC DATA GENERATION
# ─────────────────────────────────────────────

def generate_dataset(n_samples: int = 2000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic smartphone-sensor dataset.

    Features
    --------
    screen_time          : minutes of active screen use per hour
    motion_activity      : accelerometer / gyroscope activity score (0–10)
    touch_frequency      : touch events per minute
    battery_consumption  : % battery drained per hour
    cpu_usage            : CPU utilisation percentage
    temperature          : device surface temperature (°C)
    device_activity      : background-process activity score (0–10)

    Target  (usage_level)
    ----------------------
    0 → Normal   1 → Heavy   2 → Overheating
    """
    np.random.seed(random_state)
    n = n_samples

    # ── Normal usage (≈50 %) ──────────────────────────────────
    n_normal = int(n * 0.50)
    normal = pd.DataFrame({
        "screen_time":         np.random.normal(20, 5,  n_normal).clip(0, 60),
        "motion_activity":     np.random.normal(3,  1,  n_normal).clip(0, 10),
        "touch_frequency":     np.random.normal(5,  2,  n_normal).clip(0, 30),
        "battery_consumption": np.random.normal(8,  2,  n_normal).clip(0, 50),
        "cpu_usage":           np.random.normal(25, 8,  n_normal).clip(0, 100),
        "temperature":         np.random.normal(30, 2,  n_normal).clip(20, 60),
        "device_activity":     np.random.normal(3,  1,  n_normal).clip(0, 10),
        "usage_level": 0
    })

    # ── Heavy usage (≈35 %) ───────────────────────────────────
    n_heavy = int(n * 0.35)
    heavy = pd.DataFrame({
        "screen_time":         np.random.normal(45, 8,  n_heavy).clip(0, 60),
        "motion_activity":     np.random.normal(6,  1.5, n_heavy).clip(0, 10),
        "touch_frequency":     np.random.normal(18, 4,  n_heavy).clip(0, 30),
        "battery_consumption": np.random.normal(22, 4,  n_heavy).clip(0, 50),
        "cpu_usage":           np.random.normal(65, 10, n_heavy).clip(0, 100),
        "temperature":         np.random.normal(40, 3,  n_heavy).clip(20, 60),
        "device_activity":     np.random.normal(6,  1.5, n_heavy).clip(0, 10),
        "usage_level": 1
    })

    # ── Overheating (≈15 %) ───────────────────────────────────
    n_over = n - n_normal - n_heavy
    over = pd.DataFrame({
        "screen_time":         np.random.normal(55, 5,  n_over).clip(0, 60),
        "motion_activity":     np.random.normal(8,  1,  n_over).clip(0, 10),
        "touch_frequency":     np.random.normal(25, 3,  n_over).clip(0, 30),
        "battery_consumption": np.random.normal(40, 5,  n_over).clip(0, 50),
        "cpu_usage":           np.random.normal(88, 7,  n_over).clip(0, 100),
        "temperature":         np.random.normal(52, 3,  n_over).clip(20, 60),
        "device_activity":     np.random.normal(9,  0.8, n_over).clip(0, 10),
        "usage_level": 2
    })

    df = pd.concat([normal, heavy, over], ignore_index=True)

    # ── Inject synthetic noise ─────────────────────────────────
    # Missing values (~2 %)
    for col in df.columns[:-1]:
        mask = np.random.rand(len(df)) < 0.02
        df.loc[mask, col] = np.nan

    # Outliers (~1 %)
    for col in ["cpu_usage", "temperature", "battery_consumption"]:
        idx = np.random.choice(df.index, size=int(n * 0.01), replace=False)
        df.loc[idx, col] = df[col].max() * np.random.uniform(1.3, 1.8, size=len(idx))

    return df.sample(frac=1, random_state=random_state).reset_index(drop=True)


# ─────────────────────────────────────────────
# 2.  PREPROCESSING
# ─────────────────────────────────────────────

FEATURE_COLS = [
    "screen_time", "motion_activity", "touch_frequency",
    "battery_consumption", "cpu_usage", "temperature", "device_activity"
]
TARGET_COL   = "usage_level"
LABEL_MAP    = {0: "Normal", 1: "Heavy", 2: "Overheating"}


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill numeric NaNs with column median."""
    df = df.copy()
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)
    return df


def remove_outliers_iqr(df: pd.DataFrame, cols: list = None, factor: float = 2.5) -> pd.DataFrame:
    """Cap outliers using IQR fencing (clips, does not drop rows)."""
    df = df.copy()
    cols = cols or FEATURE_COLS
    for col in cols:
        if col not in df.columns:
            continue
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - factor * IQR, Q3 + factor * IQR
        df[col] = df[col].clip(lower, upper)
    return df


def normalize_features(X_train: np.ndarray, X_test: np.ndarray):
    """
    Fit a StandardScaler on training data, transform both splits.
    Returns (X_train_scaled, X_test_scaled, scaler).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


# ─────────────────────────────────────────────
# 3.  EVALUATION UTILITIES
# ─────────────────────────────────────────────

def evaluate_model(name: str, model, X_test, y_test,
                   label_names=None) -> dict:
    """
    Compute and print a full classification report, then return
    a dict of scalar metrics for comparison tables.
    """
    label_names = label_names or list(LABEL_MAP.values())
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # ROC-AUC (one-vs-rest)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
        try:
            auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
        except Exception:
            auc = np.nan
    else:
        auc = np.nan

    print(f"\n{'═'*50}")
    print(f"  {name}")
    print(f"{'═'*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}" if not np.isnan(auc) else "  ROC-AUC   : N/A")
    print(classification_report(y_test, y_pred, target_names=label_names))

    return {"Model": name, "Accuracy": acc, "Precision": prec,
            "Recall": rec, "F1-Score": f1, "ROC-AUC": auc}


def plot_confusion_matrix(model, X_test, y_test,
                          title: str = "Confusion Matrix",
                          save_path: str = None):
    """Plot a styled confusion matrix heatmap."""
    labels = list(LABEL_MAP.values())
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual",    fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_roc_curves(models_dict: dict, X_test, y_test,
                   save_path: str = None):
    """
    Plot one-vs-rest ROC curves for every model that supports
    predict_proba.
    """
    from sklearn.preprocessing import label_binarize
    classes = sorted(np.unique(y_test))
    y_bin   = label_binarize(y_test, classes=classes)
    colors  = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]

    fig, axes = plt.subplots(1, len(classes),
                             figsize=(5 * len(classes), 4), sharey=True)
    label_names = list(LABEL_MAP.values())

    for cls_idx, cls in enumerate(classes):
        ax = axes[cls_idx]
        for model_idx, (name, model) in enumerate(models_dict.items()):
            if not hasattr(model, "predict_proba"):
                continue
            y_prob = model.predict_proba(X_test)[:, cls_idx]
            fpr, tpr, _ = roc_curve(y_bin[:, cls_idx], y_prob)
            auc = roc_auc_score(y_bin[:, cls_idx], y_prob)
            ax.plot(fpr, tpr,
                    color=colors[model_idx % len(colors)],
                    label=f"{name} (AUC={auc:.2f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.set_title(f"ROC — {label_names[cls_idx]}", fontsize=11)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=7)

    plt.suptitle("ROC Curves (One-vs-Rest)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
    plt.close()


# ─────────────────────────────────────────────
# 4.  ALERT / RECOMMENDATION ENGINE
# ─────────────────────────────────────────────

def get_alerts_and_recommendations(prediction: int,
                                   input_data: dict) -> dict:
    """
    Given a predicted class and raw input values, return:
    - alert_level  : 'green' | 'orange' | 'red'
    - alert_title  : short heading
    - alert_msg    : detailed message
    - tips         : list of actionable recommendations
    """
    label = LABEL_MAP[prediction]

    recommendations = {
        "Normal": {
            "alert_level": "green",
            "alert_title": "✅ Normal Usage Detected",
            "alert_msg": "Your device is operating within safe parameters.",
            "tips": [
                "Keep screen brightness at a comfortable level.",
                "Enable battery saver mode during idle periods.",
                "Schedule app updates for overnight charging."
            ]
        },
        "Heavy": {
            "alert_level": "orange",
            "alert_title": "⚠️ Heavy Usage Detected",
            "alert_msg": "Your device is under significant load. Consider reducing activity.",
            "tips": [
                "Close unused background applications.",
                "Reduce screen brightness by 20–30 %.",
                "Switch to low-power mode.",
                "Avoid resource-intensive apps simultaneously.",
                "Take a short break to let the device cool down."
            ]
        },
        "Overheating": {
            "alert_level": "red",
            "alert_title": "🔴 Overheating Risk Detected!",
            "alert_msg": "Critical: Your device shows signs of overheating. Immediate action needed.",
            "tips": [
                "⚡ IMMEDIATELY close all non-essential apps.",
                "Remove the phone case to improve heat dissipation.",
                "Place device on a hard, flat surface — never soft bedding.",
                "Turn off Wi-Fi, Bluetooth, and GPS if not needed.",
                "Reduce screen brightness to minimum.",
                "Do not charge while using demanding apps.",
                "If temperature persists above 50 °C, power off the device."
            ]
        }
    }

    result = recommendations[label].copy()

    # Dynamic tip injection based on thresholds
    extra_tips = []
    if input_data.get("cpu_usage", 0) > 85:
        extra_tips.append("CPU usage is critically high — restart device if unresponsive.")
    if input_data.get("battery_consumption", 0) > 35:
        extra_tips.append("Battery draining fast — connect charger soon.")
    if input_data.get("screen_time", 0) > 50:
        extra_tips.append("Extended screen time detected — rest your eyes and device.")

    result["tips"] = extra_tips + result["tips"]
    return result
