"""
Demographic bias audit on the 300 held-out FF++ videos (100 Dark, 100 Medium,
100 Light by ITA skin tone). Runs 5-fold CV for XGBoost (baseline, class-weighted,
and per-group threshold optimisation), Random Forest, and Logistic Regression,
then reports per-group accuracy and the fairness gap (max - min accuracy across groups).
Also runs a direct held-out test using the pre-trained rf_model.pkl and xgb_model.pkl.
"""
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

try:
    from fairlearn.postprocessing import ThresholdOptimizer
    FAIRLEARN_AVAILABLE = True
except ImportError:
    FAIRLEARN_AVAILABLE = False
    print("fairlearn not installed — skipping ThresholdOptimizer. Run: pip install fairlearn")

# Best params from RandomizedSearchCV (tune_xgboost.py)
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

BEST_RF_PARAMS = {
    "n_estimators": 300,
    "max_depth":    None,
    "random_state": 42,
}

# --- CONFIG ---
FEATURES_CSV = "data/output/unified_features.csv"
BIAS_IDS_CSV = "data/output/bias_audit_ids.csv"
OUTPUT_DIR   = "data/report_visuals"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURES = [
    "chrom_snr", "pos_snr", "chrom_bpm", "pos_bpm",
    "mean_ear", "std_ear", "min_ear",
    "blink_count", "blink_rate_per_min",
    "mean_blink_duration", "std_blink_duration",
    "measured_ita",
]
LABEL      = "is_deepfake"
N_SPLITS   = 5
ITA_GROUPS = ["light", "medium", "dark"]


def load_data():
    features = pd.read_csv(FEATURES_CSV)
    bias_ids = pd.read_csv(BIAS_IDS_CSV)

    features["join_id"] = features["video_id"].apply(
        lambda x: str(x).split("/")[-1].replace(".mp4", "")
    )
    bias_ids["join_id"] = bias_ids["video_id"].apply(
        lambda x: str(x).split("/")[-1].replace(".mp4", "")
    )

    df = features.merge(bias_ids[["join_id", "ita_group"]], on="join_id", how="inner")
    df = df.dropna(subset=FEATURES + [LABEL]).reset_index(drop=True)

    print(f"Bias audit subset: {len(df)} videos")
    print(df["ita_group"].value_counts().to_string())
    print(f"\nReal/Fake split:\n{df[LABEL].value_counts().rename({0:'real',1:'fake'}).to_string()}")
    return df


def run_bias_audit(df, label, make_model_fn):
    """
    make_model_fn(X_train, y_train, groups_train) -> predict_fn(X_val, groups_val)
    """
    X      = df[FEATURES].values
    y      = df[LABEL].values
    groups = df["ita_group"].values

    cv             = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    group_results  = {g: [] for g in ITA_GROUPS}
    overall_results = []

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val     = X[train_idx], X[val_idx]
        y_train, y_val     = y[train_idx], y[val_idx]
        groups_train       = groups[train_idx]
        groups_val         = groups[val_idx]

        predict_fn = make_model_fn(X_train, y_train, groups_train)
        preds_val  = predict_fn(X_val, groups_val)

        overall_results.append({
            "accuracy":  accuracy_score(y_val, preds_val),
            "precision": precision_score(y_val, preds_val, zero_division=0),
            "recall":    recall_score(y_val, preds_val, zero_division=0),
            "f1":        f1_score(y_val, preds_val, zero_division=0),
        })

        val_df          = df.iloc[val_idx].copy()
        val_df["_pred"] = preds_val

        for group in ITA_GROUPS:
            mask = val_df["ita_group"] == group
            if mask.sum() == 0:
                continue
            g_true = val_df.loc[mask, LABEL].values
            g_pred = val_df.loc[mask, "_pred"].values
            group_results[group].append({
                "accuracy":  accuracy_score(g_true, g_pred),
                "precision": precision_score(g_true, g_pred, zero_division=0),
                "recall":    recall_score(g_true, g_pred, zero_division=0),
                "f1":        f1_score(g_true, g_pred, zero_division=0),
                "n":         int(mask.sum()),
            })

    print(f"\n========== BIAS AUDIT — {label.upper()} ==========")
    overall_acc  = np.mean([r["accuracy"]  for r in overall_results])
    overall_prec = np.mean([r["precision"] for r in overall_results])
    overall_rec  = np.mean([r["recall"]    for r in overall_results])
    overall_f1   = np.mean([r["f1"]        for r in overall_results])
    print(f"Overall: Acc={overall_acc:.3f} ± {np.std([r['accuracy'] for r in overall_results]):.3f} | "
          f"Prec={overall_prec:.3f} | Rec={overall_rec:.3f} | F1={overall_f1:.3f}")

    summary = {}
    for group in ITA_GROUPS:
        if not group_results[group]:
            continue
        accs  = [r["accuracy"]  for r in group_results[group]]
        precs = [r["precision"] for r in group_results[group]]
        recs  = [r["recall"]    for r in group_results[group]]
        f1s   = [r["f1"]        for r in group_results[group]]
        ns    = [r["n"]         for r in group_results[group]]
        print(f"  {group.capitalize():8s} (n≈{int(np.mean(ns)):3d}): "
              f"Acc={np.mean(accs):.3f} ± {np.std(accs):.3f} | "
              f"Prec={np.mean(precs):.3f} | Rec={np.mean(recs):.3f} | "
              f"F1={np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
        summary[group] = {
            "accuracy":  np.mean(accs),
            "precision": np.mean(precs),
            "recall":    np.mean(recs),
            "f1":        np.mean(f1s),
        }

    return summary


# --- Model factory helpers ---

def xgb_baseline(X_train, y_train, _groups):
    model = XGBClassifier(**BEST_XGB_PARAMS)
    model.fit(X_train, y_train)
    return lambda X_val, _g: model.predict(X_val)


def xgb_class_weights(X_train, y_train, _groups):
    params = dict(BEST_XGB_PARAMS)
    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    params["scale_pos_weight"] = neg / pos if pos > 0 else 1
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    return lambda X_val, _g: model.predict(X_val)


def xgb_threshold_optimizer(X_train, y_train, groups_train):
    """
    Manual per-group threshold optimisation (equalized accuracy).
    For each ITA group, find the decision threshold on the training set
    that maximises that group's accuracy, then apply per-group thresholds
    at inference time.
    """
    model = XGBClassifier(**BEST_XGB_PARAMS)
    model.fit(X_train, y_train)

    # Find best threshold per group on training set
    train_probs = model.predict_proba(X_train)[:, 1]
    thresholds  = np.linspace(0.1, 0.9, 33)
    group_thresholds = {}
    for group in ITA_GROUPS:
        mask = groups_train == group
        if mask.sum() == 0:
            group_thresholds[group] = 0.5
            continue
        g_probs = train_probs[mask]
        g_true  = y_train[mask]
        best_t, best_acc = 0.5, 0.0
        for t in thresholds:
            acc = accuracy_score(g_true, (g_probs >= t).astype(int))
            if acc > best_acc:
                best_acc, best_t = acc, t
        group_thresholds[group] = best_t

    def predict_fn(X_val, groups_val):
        probs = model.predict_proba(X_val)[:, 1]
        preds = np.zeros(len(probs), dtype=int)
        for group, t in group_thresholds.items():
            mask = groups_val == group
            preds[mask] = (probs[mask] >= t).astype(int)
        return preds

    return predict_fn


def rf_baseline(X_train, y_train, _groups):
    model = RandomForestClassifier(**BEST_RF_PARAMS)
    model.fit(X_train, y_train)
    return lambda X_val, _g: model.predict(X_val)


def lr_baseline(X_train, y_train, _groups):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_scaled, y_train)
    return lambda X_val, _g: model.predict(scaler.transform(X_val))


# --- Plotting ---

def plot_xgb_mitigation(baseline, class_w, thresh_opt):
    groups = [g.capitalize() for g in ITA_GROUPS]
    x      = np.arange(len(groups))
    w      = 0.22

    conditions = [
        ("XGBoost baseline",        baseline,   "#4C72B0"),
        ("XGBoost + class weights", class_w,    "#55A868"),
        ("XGBoost + ThresholdOpt",  thresh_opt, "#C44E52"),
    ]
    n = len(conditions)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, ylabel in [
        (axes[0], "accuracy", "Accuracy"),
        (axes[1], "f1",       "F1 Score"),
    ]:
        for i, (lbl, data, color) in enumerate(conditions):
            offset = (i - (n - 1) / 2) * w
            vals   = [data[g][metric] for g in ITA_GROUPS]
            bars   = ax.bar(x + offset, vals, w, label=lbl, color=color)
            for bar in bars:
                ax.annotate(f"{bar.get_height():.3f}",
                            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{ylabel} by Skin Tone Group", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(groups, fontsize=11)
        ax.legend(fontsize=8)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

    plt.suptitle("Bias Audit: XGBoost Fairness Mitigation by ITA Skin Tone Group (5-Fold CV)",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "bias_audit_xgb_mitigation.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved to {path}")


def plot_model_comparison(xgb_res, rf_res):
    groups = [g.capitalize() for g in ITA_GROUPS]
    x      = np.arange(len(groups))
    w      = 0.3

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, ylabel in [
        (axes[0], "accuracy", "Accuracy"),
        (axes[1], "f1",       "F1 Score"),
    ]:
        b1 = ax.bar(x - w / 2, [xgb_res[g][metric] for g in ITA_GROUPS], w,
                    label="XGBoost (tuned)", color="#4C72B0")
        b2 = ax.bar(x + w / 2, [rf_res[g][metric]  for g in ITA_GROUPS], w,
                    label="Random Forest",   color="#DD8452")
        for bar in list(b1) + list(b2):
            ax.annotate(f"{bar.get_height():.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{ylabel} by Skin Tone Group", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(groups, fontsize=11)
        ax.legend(fontsize=9)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

    plt.suptitle("Bias Audit: XGBoost vs Random Forest by ITA Skin Tone Group (5-Fold CV)",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "bias_audit_model_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved to {path}")


def fairness_gap(summary, label):
    accs = [summary[g]["accuracy"] for g in ITA_GROUPS if g in summary]
    gap  = max(accs) - min(accs)
    print(f"  {label}: gap = {gap:.3f}")
    return gap


def run_held_out(df, model_path, label):
    """Test a saved model (trained on 1,699) directly on the 300 held-out videos."""
    if not os.path.exists(model_path):
        print(f"{model_path} not found — run classifier.py first.")
        return

    model  = joblib.load(model_path)
    X      = df[FEATURES].values
    y      = df[LABEL].values
    groups = df["ita_group"].values
    preds  = model.predict(X)

    print(f"\n========== HELD-OUT BIAS AUDIT — {label.upper()} (trained on 1,699) ==========")
    print(f"Overall accuracy: {accuracy_score(y, preds):.3f}")

    summary = {}
    for group in ITA_GROUPS:
        mask = groups == group
        if mask.sum() == 0:
            continue
        g_true = y[mask]
        g_pred = preds[mask]
        acc  = accuracy_score(g_true, g_pred)
        prec = precision_score(g_true, g_pred, zero_division=0)
        rec  = recall_score(g_true, g_pred, zero_division=0)
        f1   = f1_score(g_true, g_pred, zero_division=0)
        print(f"  {group.capitalize():8s} (n={mask.sum():3d}): "
              f"Acc={acc:.3f} | Prec={prec:.3f} | Rec={rec:.3f} | F1={f1:.3f}")
        summary[group] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    accs = [summary[g]["accuracy"] for g in ITA_GROUPS if g in summary]
    print(f"  Fairness gap (max - min accuracy): {max(accs) - min(accs):.3f}")
    return summary


def run():
    df = load_data()

    print("\n--- Proper held-out evaluation (trained on 1,699, tested on 300) ---")
    run_held_out(df, "data/output/rf_model.pkl",  "Random Forest")
    run_held_out(df, "data/output/xgb_model.pkl", "XGBoost")

    xgb_base = run_bias_audit(df, "XGBoost baseline",       xgb_baseline)
    xgb_cw   = run_bias_audit(df, "XGBoost + class weights", xgb_class_weights)
    xgb_to   = run_bias_audit(df, "XGBoost + ThresholdOpt",  xgb_threshold_optimizer)
    rf_base  = run_bias_audit(df, "Random Forest baseline",  rf_baseline)
    lr_base  = run_bias_audit(df, "Logistic Regression",     lr_baseline)

    print("\n========== FAIRNESS GAPS (max - min accuracy across groups) ==========")
    fairness_gap(xgb_base, "XGBoost baseline       ")
    fairness_gap(xgb_cw,   "XGBoost + class weights")
    fairness_gap(xgb_to,   "XGBoost + ThresholdOpt ")
    fairness_gap(rf_base,  "Random Forest baseline  ")
    fairness_gap(lr_base,  "Logistic Regression     ")

    plot_xgb_mitigation(xgb_base, xgb_cw, xgb_to)
    plot_model_comparison(xgb_base, rf_base)


if __name__ == "__main__":
    run()
