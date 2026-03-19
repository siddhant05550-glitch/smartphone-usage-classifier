"""
main.py — End-to-End Training Pipeline
=======================================
Runs the complete ML workflow:
  1. Data generation & preprocessing
  2. Exploratory Data Analysis (EDA)
  3. Feature Engineering & Selection (PCA, RFE, SelectKBest)
  4. Model Training (LR, DT, RF, SVM, MLP)
  5. Evaluation (metrics, confusion matrices, ROC curves)
  6. Best-model selection & persistence (model.pkl)
  7. SHAP explainability

Run:
    python main.py
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

import utils

warnings.filterwarnings("ignore")

# ── Output directories ──────────────────────────────────────────────────────
os.makedirs("plots",  exist_ok=True)
os.makedirs("models", exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — DATA GENERATION & PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 1 — DATA GENERATION & PREPROCESSING")
print("="*60)

df = utils.generate_dataset(n_samples=2000)
print(f"[INFO] Dataset generated: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"[INFO] Missing values before cleaning:\n{df.isnull().sum()}")
print(f"[INFO] Class distribution:\n{df['usage_level'].value_counts()}")

# Handle missing values
df = utils.handle_missing_values(df)
print(f"\n[INFO] Missing values after imputation:\n{df.isnull().sum()}")

# Outlier removal (IQR capping)
df = utils.remove_outliers_iqr(df)
print("[INFO] Outliers capped via IQR fencing.")

# Save processed dataset
df.to_csv("models/processed_data.csv", index=False)
print("[INFO] Processes dataset saved -> models/precessed_data.csv ")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — EXPLORATORY DATA ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 2 — EXPLORATORY DATA ANALYSIS")
print("="*60)

sns.set_theme(style="whitegrid", palette="muted")
label_names = list(utils.LABEL_MAP.values())
palette     = {"Normal": "#3498db", "Heavy": "#f39c12", "Overheating": "#e74c3c"}
df["label"] = df["usage_level"].map(utils.LABEL_MAP)

# ── 2a. Class distribution bar chart ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
counts = df["label"].value_counts()[label_names]
bars = ax.bar(counts.index, counts.values,
              color=[palette[l] for l in counts.index], edgecolor="white", linewidth=0.8)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
            str(val), ha="center", fontsize=10)
ax.set_title("Class Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("Usage Level"); ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig("plots/01_class_distribution.png", dpi=120)
plt.close()
print("[EDA] Saved -> plots/01_class_distribution.png")

# ── 2b. Feature distributions (KDE) ───────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()
for i, col in enumerate(utils.FEATURE_COLS):
    for lbl in label_names:
        subset = df[df["label"] == lbl][col]
        axes[i].hist(subset, bins=30, alpha=0.5,
                     label=lbl, color=palette[lbl], density=True)
    axes[i].set_title(col.replace("_", " ").title(), fontsize=11)
    axes[i].legend(fontsize=7)
axes[-1].axis("off")
fig.suptitle("Feature Distributions by Usage Level", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/02_feature_distributions.png", dpi=120)
plt.close()
print("[EDA] Saved -> plots/02_feature_distributions.png")

# ── 2c. Correlation heatmap ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
corr = df[utils.FEATURE_COLS + ["usage_level"]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", ax=ax, linewidths=0.5,
            annot_kws={"size": 9})
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/03_correlation_heatmap.png", dpi=120)
plt.close()
print("[EDA] Saved -> plots/03_correlation_heatmap.png")

# ── 2d. Box plots ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()
for i, col in enumerate(utils.FEATURE_COLS):
    data = [df[df["label"] == lbl][col].values for lbl in label_names]
    bp = axes[i].boxplot(data, patch_artist=True, labels=label_names)
    for patch, lbl in zip(bp["boxes"], label_names):
        patch.set_facecolor(palette[lbl])
        patch.set_alpha(0.7)
    axes[i].set_title(col.replace("_", " ").title(), fontsize=11)
    axes[i].tick_params(axis="x", labelsize=8)
axes[-1].axis("off")
fig.suptitle("Box Plots by Usage Level", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/04_boxplots.png", dpi=120)
plt.close()
print("[EDA] Saved -> plots/04_boxplots.png")

# ── 2e. Pair plot (subset) ────────────────────────────────────────────────
pair_cols = ["screen_time", "cpu_usage", "temperature", "battery_consumption", "label"]
pair_df   = df[pair_cols].sample(500, random_state=42)
pg = sns.pairplot(pair_df, hue="label", palette=palette,
                  plot_kws={"alpha": 0.4, "s": 20}, diag_kind="kde")
pg.fig.suptitle("Pair Plot — Key Features", y=1.01, fontsize=14, fontweight="bold")
pg.savefig("plots/05_pairplot.png", dpi=100)
plt.close()
print("[EDA] Saved -> plots/05_pairplot.png")

# ── 2f. Battery vs temperature scatter ───────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
for lbl in label_names:
    sub = df[df["label"] == lbl]
    ax.scatter(sub["temperature"], sub["battery_consumption"],
               alpha=0.35, s=18, label=lbl, color=palette[lbl])
ax.set_xlabel("Temperature (°C)", fontsize=12)
ax.set_ylabel("Battery Consumption (% / hr)", fontsize=12)
ax.set_title("Temperature vs Battery Consumption", fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("plots/06_temp_vs_battery.png", dpi=120)
plt.close()
print("[EDA] Saved -> plots/06_temp_vs_battery.png")

# ── 2g. Statistical summary ───────────────────────────────────────────────
print("\n[EDA] Descriptive statistics:\n")
print(df[utils.FEATURE_COLS].describe().round(2).to_string())

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — FEATURE ENGINEERING & SELECTION
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 3 — FEATURE ENGINEERING & SELECTION")
print("="*60)

X = df[utils.FEATURE_COLS].values
y = df[utils.TARGET_COL].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_sc, X_test_sc, scaler = utils.normalize_features(X_train, X_test)
joblib.dump(scaler, "models/scaler.pkl")
print("[INFO] Scaler saved -> models/scaler.pkl")

# ── 3a. PCA (visualisation) ───────────────────────────────────────────────
pca_2d = PCA(n_components=2, random_state=42)
X_pca  = pca_2d.fit_transform(X_train_sc)
fig, ax = plt.subplots(figsize=(7, 5))
colors  = [["#3498db", "#f39c12", "#e74c3c"][c] for c in y_train]
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.4, s=15)
from matplotlib.patches import Patch
legend_handles = [Patch(color=c, label=l)
                  for c, l in zip(["#3498db","#f39c12","#e74c3c"], label_names)]
ax.legend(handles=legend_handles)
ax.set_title(f"PCA 2-D Projection  (var explained: "
             f"{pca_2d.explained_variance_ratio_.sum():.1%})",
             fontsize=12, fontweight="bold")
ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
plt.tight_layout()
plt.savefig("plots/07_pca_2d.png", dpi=120)
plt.close()
print("[FS] Saved -> plots/07_pca_2d.png")

# ── 3b. SelectKBest (ANOVA F-score) ──────────────────────────────────────
selector = SelectKBest(f_classif, k="all")
selector.fit(X_train_sc, y_train)
f_scores = pd.Series(selector.scores_, index=utils.FEATURE_COLS).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(7, 4))
f_scores.plot(kind="bar", ax=ax, color="#3498db", edgecolor="white")
ax.set_title("SelectKBest — ANOVA F-Scores", fontsize=13, fontweight="bold")
ax.set_ylabel("F-Score"); ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
plt.tight_layout()
plt.savefig("plots/08_selectkbest.png", dpi=120)
plt.close()
top_k_features = f_scores.head(5).index.tolist()
print(f"[FS] Top-5 features (SelectKBest): {top_k_features}")

# ── 3c. RFE with Random Forest ────────────────────────────────────────────
rfe_rf  = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
rfe_sel = RFE(estimator=rfe_rf, n_features_to_select=5, step=1)
rfe_sel.fit(X_train_sc, y_train)
rfe_features = [f for f, s in zip(utils.FEATURE_COLS, rfe_sel.support_) if s]
print(f"[FS] Top-5 features (RFE):          {rfe_features}")

# ── 3d. Variance explained (PCA full) ────────────────────────────────────
pca_full = PCA(random_state=42)
pca_full.fit(X_train_sc)
cum_var  = np.cumsum(pca_full.explained_variance_ratio_)
n_95     = np.argmax(cum_var >= 0.95) + 1
print(f"[FS] PCA: {n_95} components explain >= 95% variance")

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(range(1, len(cum_var)+1), pca_full.explained_variance_ratio_,
       color="#9b59b6", label="Individual")
ax.step(range(1, len(cum_var)+1), cum_var, where="mid",
        color="#e74c3c", linewidth=2, label="Cumulative")
ax.axhline(0.95, ls="--", color="grey", linewidth=0.8)
ax.set_title("PCA — Explained Variance", fontsize=13, fontweight="bold")
ax.set_xlabel("Principal Components"); ax.set_ylabel("Variance Ratio")
ax.legend()
plt.tight_layout()
plt.savefig("plots/09_pca_variance.png", dpi=120)
plt.close()
print("[FS] Saved -> plots/09_pca_variance.png")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 4 — MODEL TRAINING")
print("="*60)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=8, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=150, random_state=42, n_jobs=-1),
    "SVM": SVC(
        kernel="rbf", probability=True, random_state=42, C=1.0),
    "Neural Network": MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu", max_iter=500, random_state=42,
        early_stopping=True, validation_fraction=0.1)
}

# 5-fold cross-validation summary before final evaluation
print("\n[INFO] 5-Fold Cross-Validation Accuracies:")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, model in models.items():
    scores = cross_val_score(model, X_train_sc, y_train,
                             cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"  {name:22s}: {scores.mean():.4f} ± {scores.std():.4f}")

# Fit all models on full training set
for name, model in models.items():
    model.fit(X_train_sc, y_train)
    print(f"  [TRAINED] {name}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 — EVALUATION
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 5 — EVALUATION")
print("="*60)

results = []
for name, model in models.items():
    metrics = utils.evaluate_model(name, model, X_test_sc, y_test, label_names)
    results.append(metrics)
    utils.plot_confusion_matrix(
        model, X_test_sc, y_test,
        title=f"Confusion Matrix — {name}",
        save_path=f"plots/cm_{name.replace(' ', '_').lower()}.png"
    )

# ROC curves (all models on one figure set)
utils.plot_roc_curves(models, X_test_sc, y_test,
                      save_path="plots/10_roc_curves.png")
print("[EVAL] Saved -> plots/10_roc_curves.png")

# Comparison table
results_df = pd.DataFrame(results).set_index("Model")
print("\n[EVAL] Model Comparison Table:")
print(results_df.round(4).to_string())
results_df.to_csv("models/model_comparison.csv")

# Bar chart comparison
fig, ax = plt.subplots(figsize=(10, 5))
metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score"]
x = np.arange(len(results_df))
width = 0.2
clrs  = ["#3498db", "#2ecc71", "#e67e22", "#e74c3c"]
for i, (col, clr) in enumerate(zip(metric_cols, clrs)):
    ax.bar(x + i*width, results_df[col], width, label=col, color=clr, alpha=0.85)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(results_df.index, rotation=20, ha="right", fontsize=9)
ax.set_ylim(0, 1.08)
ax.set_ylabel("Score"); ax.legend(fontsize=9)
ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/11_model_comparison.png", dpi=120)
plt.close()
print("[EVAL] Saved -> plots/11_model_comparison.png")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 6 — BEST MODEL SELECTION & PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 6 — BEST MODEL SELECTION & SAVING")
print("="*60)

best_name = results_df["F1-Score"].idxmax()
best_model = models[best_name]
print(f"[INFO] Best model: {best_name}  (F1={results_df.loc[best_name,'F1-Score']:.4f})")

joblib.dump(best_model, "models/best_model.pkl")
print("[INFO] Saved -> models/best_model.pkl")

# Also save feature names for the Streamlit app
joblib.dump(utils.FEATURE_COLS, "models/feature_cols.pkl")
joblib.dump(utils.LABEL_MAP,    "models/label_map.pkl")
print("[INFO] Metadata saved.")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 7 — EXPLAINABILITY (SHAP)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 7 — SHAP EXPLAINABILITY")
print("="*60)

try:
    import shap

    # Use a Tree-based model for reliable SHAP values
    rf_model = models["Random Forest"]

    # Use a small background sample for speed
    bg_sample = shap.sample(X_train_sc, 200, random_state=42)
    explainer  = shap.TreeExplainer(rf_model)
    shap_vals  = explainer.shap_values(X_test_sc[:300])

    # ── Summary plot (beeswarm) ────────────────────────────────
    fig_shap, axes_shap = plt.subplots(1, 3, figsize=(18, 5))
    for cls_idx, ax_s in enumerate(axes_shap):
        plt.sca(ax_s)
        shap.summary_plot(
            shap_vals[cls_idx], X_test_sc[:300],
            feature_names=utils.FEATURE_COLS,
            plot_type="bar", show=False,
            color=["#3498db","#f39c12","#e74c3c"][cls_idx]
        )
        ax_s.set_title(label_names[cls_idx], fontsize=12, fontweight="bold")
    plt.suptitle("SHAP Feature Importance (Random Forest)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/12_shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("[SHAP] Saved -> plots/12_shap_summary.png")

    # Save SHAP explainer for the app
    joblib.dump(explainer, "models/shap_explainer.pkl")
    print("[SHAP] Explainer saved -> models/shap_explainer.pkl")

except ImportError:
    print("[SHAP] SHAP not installed — skipping. Run: pip install shap")
except Exception as e:
    print(f"[SHAP] Warning: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# DONE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  ✅  PIPELINE COMPLETE")
print("  Plots saved in  -> plots/")
print("  Models saved in -> models/")
print("  Run the app:      streamlit run app.py")
print("="*60 + "\n")
