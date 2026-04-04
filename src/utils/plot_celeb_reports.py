"""
Celeb-DF Report Visuals
========================
Generates the same charts as the FF++ report visuals but for the
Celeb-DF trained model. All outputs saved to data/report_visuals/celeb_df/.

Charts:
  1. Model comparison (Accuracy, Precision, Recall, F1)
  2. Accuracy-only bar chart
  3. Feature correlation matrix heatmap
  4. Feature-target correlation bar chart
  5. Feature importance (Random Forest)

Run from project root:
    python src/utils/plot_celeb_reports.py
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import os
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
CELEB_FEATURES_CSV = "data/output/celeb_unified_features.csv"
OUTPUT_DIR         = "data/report_visuals/celeb_df"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CELEB_N_REAL = 850
CELEB_N_FAKE = 850
SEED = 42

FEATURES = [
    "chrom_snr", "pos_snr", "chrom_bpm", "pos_bpm",
    "mean_ear", "std_ear", "min_ear",
    "blink_count", "blink_rate_per_min",
    "mean_blink_duration", "std_blink_duration",
    "measured_ita",
]
LABEL = "is_deepfake"

FEATURE_LABELS = {
    "chrom_snr":           "CHROM SNR",
    "pos_snr":             "POS SNR",
    "chrom_bpm":           "CHROM BPM",
    "pos_bpm":             "POS BPM",
    "mean_ear":            "Mean EAR",
    "std_ear":             "Std EAR",
    "min_ear":             "Min EAR",
    "blink_count":         "Blink Count",
    "blink_rate_per_min":  "Blink Rate/min",
    "mean_blink_duration": "Mean Blink Dur.",
    "std_blink_duration":  "Std Blink Dur.",
    "measured_ita":        "ITA (Skin Tone)",
    "is_deepfake":         "Is Deepfake",
}

# Celeb-DF 5-fold CV results (from celeb_classifier.py output)
CELEB_RESULTS = {
    "Logistic\nRegression": {"accuracy": 0.603, "precision": 0.601, "recall": 0.611, "f1": 0.606},
    "Random\nForest":       {"accuracy": 0.598, "precision": 0.592, "recall": 0.631, "f1": 0.610},
    "XGBoost":              {"accuracy": 0.618, "precision": 0.609, "recall": 0.659, "f1": 0.633},
}


def load_celeb_data():
    df = pd.read_csv(CELEB_FEATURES_CSV)
    df = df.dropna(subset=FEATURES + [LABEL])
    df_real = df[df[LABEL] == 0].sample(n=CELEB_N_REAL, random_state=SEED)
    df_fake = df[df[LABEL] == 1].sample(n=CELEB_N_FAKE, random_state=SEED)
    df = pd.concat([df_real, df_fake]).reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# CHART 1: Model Comparison (4 metrics)
# ══════════════════════════════════════════════════════════════════════════════

def plot_model_comparison():
    models   = list(CELEB_RESULTS.keys())
    accuracy  = [CELEB_RESULTS[m]["accuracy"]  for m in models]
    precision = [CELEB_RESULTS[m]["precision"] for m in models]
    recall    = [CELEB_RESULTS[m]["recall"]    for m in models]
    f1        = [CELEB_RESULTS[m]["f1"]        for m in models]

    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - 1.5 * width, accuracy,  width, label="Accuracy",  color="#4C72B0")
    bars2 = ax.bar(x - 0.5 * width, precision, width, label="Precision", color="#55A868")
    bars3 = ax.bar(x + 0.5 * width, recall,    width, label="Recall",    color="#C44E52")
    bars4 = ax.bar(x + 1.5 * width, f1,        width, label="F1 Score",  color="#8172B2")

    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            ax.annotate(f"{bar.get_height():.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Celeb-DF Model Comparison — Deepfake Detection (5-Fold CV, n=1700)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=10)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "model_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 2: Accuracy-only bar chart
# ══════════════════════════════════════════════════════════════════════════════

def plot_accuracy_only():
    models   = list(CELEB_RESULTS.keys())
    accuracy = [CELEB_RESULTS[m]["accuracy"] for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, accuracy, color="#4C72B0", edgecolor="white", width=0.5)

    for bar in bars:
        ax.annotate(f"{bar.get_height():.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Celeb-DF Classification Accuracy by Model (5-Fold CV, n=1700)",
                 fontsize=13, fontweight="bold")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(len(models) - 0.5, 0.515, "Random baseline (0.5)", ha="right", fontsize=9, color="gray")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "model_accuracy_only.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 3: Feature Correlation Matrix
# ══════════════════════════════════════════════════════════════════════════════

def plot_correlation_matrix(df):
    cols = FEATURES + [LABEL]
    corr_df = df[cols].rename(columns=FEATURE_LABELS)
    corr = corr_df.corr()

    plt.figure(figsize=(13, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                linewidths=0.5, annot_kws={"size": 8})
    plt.title(f"Celeb-DF Feature Correlation Matrix (n={len(df)})",
              fontsize=13, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "correlation_matrix.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 4: Feature-Target Correlation
# ══════════════════════════════════════════════════════════════════════════════

def plot_feature_target_correlation(df):
    cols = FEATURES + [LABEL]
    corr_df = df[cols].rename(columns=FEATURE_LABELS)
    corr = corr_df.corr()

    target_corr = corr["Is Deepfake"].drop("Is Deepfake").sort_values()
    colors = ["#C44E52" if v < 0 else "#4C72B0" for v in target_corr]

    plt.figure(figsize=(9, 5))
    plt.barh(target_corr.index, target_corr.values, color=colors, edgecolor="white")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Pearson Correlation with Is Deepfake", fontsize=11)
    plt.title("Celeb-DF Feature Correlation with Deepfake Label",
              fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "feature_target_correlation.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    print("\nCeleb-DF correlations with is_deepfake:")
    print(target_corr.sort_values(key=abs, ascending=False).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# CHART 5: Feature Importance (Random Forest)
# ══════════════════════════════════════════════════════════════════════════════

def plot_feature_importance(df):
    X = df[FEATURES].values
    y = df[LABEL].values

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=FEATURES)
    importances = importances.sort_values(ascending=True)

    labels = [FEATURE_LABELS[f] for f in importances.index]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(labels, importances.values, color="steelblue", edgecolor="white")
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title("Celeb-DF Feature Importances (Random Forest)",
                 fontsize=13, fontweight="bold")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    print("\nCeleb-DF feature importances:")
    for feat, imp in importances.sort_values(ascending=False).items():
        print(f"  {FEATURE_LABELS[feat]}: {imp:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run():
    print("=" * 60)
    print("  CELEB-DF REPORT VISUALS")
    print("=" * 60)

    df = load_celeb_data()
    print(f"Loaded {len(df)} videos ({int(df[LABEL].sum())} fake, {int((df[LABEL]==0).sum())} real)\n")

    plot_model_comparison()
    plot_accuracy_only()
    plot_correlation_matrix(df)
    plot_feature_target_correlation(df)
    plot_feature_importance(df)

    print(f"\nAll charts saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    run()
