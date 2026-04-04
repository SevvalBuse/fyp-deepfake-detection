"""
Celeb-DF Per-Model Heatmaps with ITA Group Breakdown
======================================================
Same style as the FF++ individual_models heatmaps.
Runs 5-fold CV and computes metrics per ITA skin tone group.

Run from project root:
    python src/utils/plot_celeb_individual_models.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import os
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
CELEB_FEATURES_CSV = "data/output/celeb_unified_features.csv"
OUTPUT_DIR         = "data/report_visuals/celeb_df/individual_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CELEB_N_REAL = 850
CELEB_N_FAKE = 850
SEED = 42
N_SPLITS = 5

FEATURES = [
    "chrom_snr", "pos_snr", "chrom_bpm", "pos_bpm",
    "mean_ear", "std_ear", "min_ear",
    "blink_count", "blink_rate_per_min",
    "mean_blink_duration", "std_blink_duration",
    "measured_ita",
]
LABEL = "is_deepfake"
ITA_GROUPS = ["light", "medium", "dark"]

BEST_XGB_PARAMS = {
    "n_estimators":     500,
    "max_depth":        5,
    "learning_rate":    0.01,
    "subsample":        0.7,
    "colsample_bytree": 0.8,
    "min_child_weight": 7,
    "gamma":            0,
    "reg_alpha":        0.1,
    "reg_lambda":       1.5,
    "eval_metric":      "logloss",
    "random_state":     42,
    "verbosity":        0,
}


def ita_to_group(ita):
    if ita <= 10:
        return "dark"
    elif ita <= 41:
        return "medium"
    else:
        return "light"


def load_data():
    df = pd.read_csv(CELEB_FEATURES_CSV)
    df = df.dropna(subset=FEATURES + [LABEL])
    df_real = df[df[LABEL] == 0].sample(n=CELEB_N_REAL, random_state=SEED)
    df_fake = df[df[LABEL] == 1].sample(n=CELEB_N_FAKE, random_state=SEED)
    df = pd.concat([df_real, df_fake]).reset_index(drop=True)
    df["ita_group"] = df["measured_ita"].apply(ita_to_group)
    return df


def compute_group_metrics(df, make_model_fn, scale=False):
    """5-fold CV, compute overall + per-ITA-group metrics."""
    X      = df[FEATURES].values
    y      = df[LABEL].values
    groups = df["ita_group"].values

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    # Collect all predictions
    all_preds  = np.zeros(len(y))
    all_filled = np.zeros(len(y), dtype=bool)

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train = y[train_idx]

        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val   = scaler.transform(X_val)

        model = make_model_fn()
        model.fit(X_train, y_train)
        all_preds[val_idx] = model.predict(X_val)
        all_filled[val_idx] = True

    # Overall metrics
    results = {}
    results["Overall"] = {
        "Accuracy":  accuracy_score(y, all_preds),
        "Precision": precision_score(y, all_preds, zero_division=0),
        "Recall":    recall_score(y, all_preds, zero_division=0),
        "F1 Score":  f1_score(y, all_preds, zero_division=0),
    }

    # Per-group
    for group in ["light", "medium", "dark"]:
        mask = groups == group
        if mask.sum() == 0:
            continue
        g_true = y[mask]
        g_pred = all_preds[mask]
        results[group.capitalize()] = {
            "Accuracy":  accuracy_score(g_true, g_pred),
            "Precision": precision_score(g_true, g_pred, zero_division=0),
            "Recall":    recall_score(g_true, g_pred, zero_division=0),
            "F1 Score":  f1_score(g_true, g_pred, zero_division=0),
        }
        print(f"  {group.capitalize():8s} (n={mask.sum():4d}): Acc={results[group.capitalize()]['Accuracy']:.3f}")

    return results


def save_heatmap(title, data_dict, filename, vmin=0.35, vmax=0.85):
    """Generate heatmap matching the FF++ individual_models style."""
    df = pd.DataFrame(data_dict)
    # Reorder columns
    col_order = [c for c in ["Overall", "Light", "Medium", "Dark"] if c in df.columns]
    df = df[col_order]

    fig, ax = plt.subplots(figsize=(max(5, len(df.columns) * 1.8), max(3, len(df) * 1.1)))
    sns.heatmap(
        df,
        annot=True, fmt=".3f",
        cmap="RdYlGn", vmin=vmin, vmax=vmax,
        linewidths=0.6, linecolor="white",
        annot_kws={"size": 13, "weight": "bold"},
        ax=ax,
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11, rotation=0)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}\n")


def run():
    print("=" * 60)
    print("  CELEB-DF PER-MODEL HEATMAPS")
    print("=" * 60)

    df = load_data()
    n = len(df)
    print(f"Loaded {n} videos")
    print(f"ITA groups: {df['ita_group'].value_counts().to_dict()}\n")

    # XGBoost
    print("--- XGBoost ---")
    xgb_results = compute_group_metrics(
        df, lambda: XGBClassifier(**BEST_XGB_PARAMS))
    save_heatmap(
        f"Celeb-DF XGBoost — 5-Fold CV (n={n})",
        xgb_results, "xgboost.png", vmin=0.35, vmax=0.85)

    # Random Forest
    print("--- Random Forest ---")
    rf_results = compute_group_metrics(
        df, lambda: RandomForestClassifier(n_estimators=200, random_state=42))
    save_heatmap(
        f"Celeb-DF Random Forest — 5-Fold CV (n={n})",
        rf_results, "random_forest.png", vmin=0.35, vmax=0.85)

    # Logistic Regression
    print("--- Logistic Regression ---")
    lr_results = compute_group_metrics(
        df, lambda: LogisticRegression(max_iter=1000, random_state=42),
        scale=True)
    save_heatmap(
        f"Celeb-DF Logistic Regression — 5-Fold CV (n={n})",
        lr_results, "logistic_regression.png", vmin=0.35, vmax=0.85)

    print(f"All heatmaps saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    run()
