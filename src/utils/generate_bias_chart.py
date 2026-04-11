"""
Generates a grouped bar chart showing Accuracy, AUC ROC, and F1 Score
per skin tone group for XGBoost (bias audit visualization).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

FEATURES_CSV = "data/output/unified_features.csv"
BIAS_IDS_CSV = "data/output/bias_audit_ids.csv"
OUTPUT_PATH  = "data/report_visuals/bias_audit_xgb_grouped.png"

FEATURES = [
    "chrom_snr", "pos_snr", "chrom_bpm", "pos_bpm",
    "mean_ear", "std_ear", "min_ear",
    "blink_count", "blink_rate_per_min",
    "mean_blink_duration", "std_blink_duration",
    "measured_ita",
]
LABEL    = "is_deepfake"
N_SPLITS = 5
GROUPS   = ["light", "medium", "dark"]

# Best XGBoost params from tune_xgboost.py — duplicated here to keep this
# script self-contained (no import dependency on bias_auditor.py).
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

# --- Load data ---
features = pd.read_csv(FEATURES_CSV)
bias_ids = pd.read_csv(BIAS_IDS_CSV)

features["join_id"] = features["video_id"].apply(lambda x: str(x).split("/")[-1].replace(".mp4", ""))
bias_ids["join_id"] = bias_ids["video_id"].apply(lambda x: str(x).split("/")[-1].replace(".mp4", ""))

df = features.merge(bias_ids[["join_id", "ita_group"]], on="join_id", how="inner")
df = df.dropna(subset=FEATURES + [LABEL]).reset_index(drop=True)

print(f"Bias audit subset: {len(df)} videos")
print(df["ita_group"].value_counts().to_string())

X      = df[FEATURES].values
y      = df[LABEL].values
groups = df["ita_group"].values

# --- 5-fold CV collecting per-group metrics ---
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
group_results = {g: {"accuracy": [], "auc": [], "f1": []} for g in GROUPS}

for train_idx, val_idx in cv.split(X, y):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    groups_val     = groups[val_idx]

    model = XGBClassifier(**BEST_XGB_PARAMS)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)[:, 1]

    val_df            = df.iloc[val_idx].copy()
    val_df["_pred"]   = preds
    val_df["_prob"]   = probs

    for g in GROUPS:
        mask = val_df["ita_group"] == g
        if mask.sum() < 2:
            continue
        g_true = val_df.loc[mask, LABEL].values
        g_pred = val_df.loc[mask, "_pred"].values
        g_prob = val_df.loc[mask, "_prob"].values
        group_results[g]["accuracy"].append(accuracy_score(g_true, g_pred))
        group_results[g]["f1"].append(f1_score(g_true, g_pred, zero_division=0))
        if len(np.unique(g_true)) > 1:
            group_results[g]["auc"].append(roc_auc_score(g_true, g_prob))

# --- Compute means ---
means = {}
for g in GROUPS:
    means[g] = {
        "accuracy": np.mean(group_results[g]["accuracy"]),
        "auc":      np.mean(group_results[g]["auc"]),
        "f1":       np.mean(group_results[g]["f1"]),
    }
    print(f"  {g.capitalize():8s}: Acc={means[g]['accuracy']:.3f} | AUC={means[g]['auc']:.3f} | F1={means[g]['f1']:.3f}")

# --- Plot ---
group_labels = ["Light\n(ITA > 41)", "Medium\n(10 < ITA ≤ 41)", "Dark\n(ITA ≤ 10)"]
x     = np.arange(len(GROUPS))
width = 0.22

acc_vals = [means[g]["accuracy"] for g in GROUPS]
auc_vals = [means[g]["auc"]      for g in GROUPS]
f1_vals  = [means[g]["f1"]       for g in GROUPS]

fig, ax = plt.subplots(figsize=(10, 6))

b1 = ax.bar(x - width, acc_vals, width, label="ACCURACY",  color="#4C72B0")
b2 = ax.bar(x,         auc_vals, width, label="AUC ROC",   color="#FF7C2A")
b3 = ax.bar(x + width, f1_vals,  width, label="F1 SCORE",  color="#2CA02C")

for bars in [b1, b2, b3]:
    for bar in bars:
        ax.annotate(
            f"{bar.get_height():.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=10
        )

overall_mean = np.mean(acc_vals)
ax.axhline(overall_mean, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
ax.text(x[-1] + width + 0.05, overall_mean + 0.01, f"Overall avg ({overall_mean:.2f})",
        fontsize=8, color="gray", va="bottom")

ax.set_ylim(0, 1.0)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Deepfake Detection Fairness — Performance by Skin Tone (XGBoost, 5-Fold CV)", fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(group_labels, fontsize=11)
ax.legend(fontsize=10)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

plt.tight_layout()
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved to {OUTPUT_PATH}")
