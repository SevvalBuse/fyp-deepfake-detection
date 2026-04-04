"""
Celeb-DF v2 Classifier + Bidirectional Cross-Dataset Test
==========================================================
1. Trains RF + XGBoost on Celeb-DF features (5-fold CV)
2. Cross-tests in both directions:
   - FF++ model  → Celeb-DF test set  (expected: poor — confirms the problem)
   - Celeb-DF model → FF++ test set   (hypothesis: should generalise well)
3. Produces comparison table for the report

Run from project root:
    python src/models/celeb_classifier.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    make_scorer, accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
from xgboost import XGBClassifier
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
CELEB_FEATURES_CSV = "data/output/celeb_unified_features.csv"
FF_FEATURES_CSV    = "data/output/unified_features.csv"
FF_BIAS_IDS_CSV    = "data/output/bias_audit_ids.csv"
OUTPUT_DIR         = "data/output"

# Subsample Celeb-DF to match FF++ training size (850 real + 850 fake)
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
LABEL    = "is_deepfake"
N_SPLITS = 5

# Best XGBoost params (same as bias_auditor.py)
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


# ── DATA LOADING ──────────────────────────────────────────────────────────────

def load_celeb_data():
    df = pd.read_csv(CELEB_FEATURES_CSV)
    df = df.dropna(subset=FEATURES + [LABEL])
    print(f"Celeb-DF available: {len(df)} videos ({int(df[LABEL].sum())} fake, {int((df[LABEL]==0).sum())} real)")

    # Subsample to match FF++ training size (850 real + 850 fake)
    df_real = df[df[LABEL] == 0].sample(n=CELEB_N_REAL, random_state=SEED)
    df_fake = df[df[LABEL] == 1].sample(n=CELEB_N_FAKE, random_state=SEED)
    df = pd.concat([df_real, df_fake]).reset_index(drop=True)

    X = df[FEATURES].values
    y = df[LABEL].values
    print(f"Celeb-DF sampled:   {len(df)} videos ({int(y.sum())} fake, {int((y==0).sum())} real)")
    return X, y, df


def load_ff_data():
    """Load FF++ data. Returns full set and training-only set (excluding bias audit)."""
    df = pd.read_csv(FF_FEATURES_CSV)
    df = df.dropna(subset=FEATURES + [LABEL])

    # Split: training set (excluding bias audit) and full set
    bias_ids = pd.read_csv(FF_BIAS_IDS_CSV)
    bias_join = set(bias_ids["video_id"].apply(
        lambda x: str(x).split("/")[-1].replace(".mp4", "")
    ))
    df["join_id"] = df["video_id"].apply(
        lambda x: str(x).split("/")[-1].replace(".mp4", "")
    )

    df_train = df[~df["join_id"].isin(bias_join)].reset_index(drop=True)
    df_full  = df.reset_index(drop=True)

    X_train = df_train[FEATURES].values
    y_train = df_train[LABEL].values
    X_full  = df_full[FEATURES].values
    y_full  = df_full[LABEL].values

    print(f"FF++ full:  {len(df_full)} videos ({int(y_full.sum())} fake, {int((y_full==0).sum())} real)")
    print(f"FF++ train: {len(df_train)} videos (excl. 300 bias audit)")
    return X_train, y_train, X_full, y_full, df_full


# ── CROSS-VALIDATED EVALUATION ────────────────────────────────────────────────

def evaluate_model(name, model, X, y, scale=False):
    """5-fold stratified CV — same as classifier.py."""
    X_input = X.copy()
    if scale:
        scaler = StandardScaler()
        X_input = scaler.fit_transform(X_input)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    scorers = {
        "accuracy":  make_scorer(accuracy_score),
        "precision": make_scorer(precision_score, zero_division=0),
        "recall":    make_scorer(recall_score, zero_division=0),
        "f1":        make_scorer(f1_score, zero_division=0),
    }
    results = cross_validate(model, X_input, y, cv=cv, scoring=scorers)

    acc  = results["test_accuracy"].mean()
    prec = results["test_precision"].mean()
    rec  = results["test_recall"].mean()
    f1   = results["test_f1"].mean()

    print(f"\n--- {name} ---")
    print(f"  Accuracy:  {acc:.3f} +/- {results['test_accuracy'].std():.3f}")
    print(f"  Precision: {prec:.3f} +/- {results['test_precision'].std():.3f}")
    print(f"  Recall:    {rec:.3f} +/- {results['test_recall'].std():.3f}")
    print(f"  F1 Score:  {f1:.3f} +/- {results['test_f1'].std():.3f}")

    return {"model": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# ── CROSS-DATASET TEST ───────────────────────────────────────────────────────

def cross_test(model, model_name, X_test, y_test, direction_label):
    """Test a trained model on a different dataset."""
    preds = model.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    prec  = precision_score(y_test, preds, zero_division=0)
    rec   = recall_score(y_test, preds, zero_division=0)
    f1    = f1_score(y_test, preds, zero_division=0)

    print(f"\n--- {direction_label}: {model_name} ---")
    print(f"  Accuracy:  {acc:.3f}")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall:    {rec:.3f}")
    print(f"  F1 Score:  {f1:.3f}")
    print(f"\n  Confusion matrix:")
    cm = confusion_matrix(y_test, preds)
    print(f"    TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"    FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")

    return {"direction": direction_label, "model": model_name,
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run():
    print("=" * 70)
    print("  CELEB-DF CLASSIFIER + BIDIRECTIONAL CROSS-DATASET TEST")
    print("=" * 70)

    # Load data
    print("\n--- Loading datasets ---")
    X_celeb, y_celeb, df_celeb = load_celeb_data()
    X_ff_train, y_ff_train, X_ff_full, y_ff_full, df_ff = load_ff_data()

    # ══════════════════════════════════════════════════════════════════════
    # PART 1: Celeb-DF 5-fold CV (same evaluation as FF++ classifier.py)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PART 1: CELEB-DF CLASSIFIER RESULTS (5-fold CV)")
    print("=" * 70)

    models = [
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42), True),
        ("Random Forest",       RandomForestClassifier(n_estimators=200, random_state=42), False),
        ("XGBoost",             XGBClassifier(**BEST_XGB_PARAMS), False),
    ]

    celeb_summary = []
    for name, model, scale in models:
        result = evaluate_model(f"Celeb-DF {name}", model, X_celeb, y_celeb, scale=scale)
        celeb_summary.append(result)

    print("\n--- Celeb-DF Summary ---")
    print(pd.DataFrame(celeb_summary).set_index("model").round(3).to_string())

    # ══════════════════════════════════════════════════════════════════════
    # PART 2: Train final models on full training sets
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PART 2: TRAINING FINAL MODELS")
    print("=" * 70)

    # Train Celeb-DF models
    print("\nTraining on Celeb-DF...")
    celeb_rf = RandomForestClassifier(n_estimators=200, random_state=42)
    celeb_rf.fit(X_celeb, y_celeb)
    joblib.dump(celeb_rf, os.path.join(OUTPUT_DIR, "celeb_rf_model.pkl"))
    print("  Saved celeb_rf_model.pkl")

    celeb_xgb = XGBClassifier(**BEST_XGB_PARAMS)
    celeb_xgb.fit(X_celeb, y_celeb)
    joblib.dump(celeb_xgb, os.path.join(OUTPUT_DIR, "celeb_xgb_model.pkl"))
    print("  Saved celeb_xgb_model.pkl")

    # Load FF++ models (already trained by classifier.py)
    ff_rf_path  = os.path.join(OUTPUT_DIR, "rf_model.pkl")
    ff_xgb_path = os.path.join(OUTPUT_DIR, "xgb_model.pkl")

    if not os.path.exists(ff_rf_path) or not os.path.exists(ff_xgb_path):
        print("\nWARNING: FF++ models not found. Run classifier.py first.")
        print("Training FF++ models now...")
        ff_rf = RandomForestClassifier(n_estimators=200, random_state=42)
        ff_rf.fit(X_ff_train, y_ff_train)
        joblib.dump(ff_rf, ff_rf_path)

        ff_xgb = XGBClassifier(**BEST_XGB_PARAMS)
        ff_xgb.fit(X_ff_train, y_ff_train)
        joblib.dump(ff_xgb, ff_xgb_path)
    else:
        ff_rf  = joblib.load(ff_rf_path)
        ff_xgb = joblib.load(ff_xgb_path)
        print(f"\nLoaded FF++ models from {ff_rf_path} and {ff_xgb_path}")

    # ══════════════════════════════════════════════════════════════════════
    # PART 3: BIDIRECTIONAL CROSS-DATASET TEST
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PART 3: BIDIRECTIONAL CROSS-DATASET TEST")
    print("=" * 70)

    cross_results = []

    # Direction A: FF++ model → Celeb-DF (expected: poor)
    cross_results.append(cross_test(ff_rf,  "Random Forest", X_celeb, y_celeb,
                                    "FF++ -> Celeb-DF"))
    cross_results.append(cross_test(ff_xgb, "XGBoost",       X_celeb, y_celeb,
                                    "FF++ -> Celeb-DF"))

    # Direction B: Celeb-DF model → FF++ full set (hypothesis: should work)
    cross_results.append(cross_test(celeb_rf,  "Random Forest", X_ff_full, y_ff_full,
                                    "Celeb-DF -> FF++"))
    cross_results.append(cross_test(celeb_xgb, "XGBoost",       X_ff_full, y_ff_full,
                                    "Celeb-DF -> FF++"))

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  CROSS-DATASET SUMMARY")
    print("=" * 70)

    cross_df = pd.DataFrame(cross_results)
    cross_df = cross_df[["direction", "model", "accuracy", "precision", "recall", "f1"]]
    print("\n" + cross_df.round(3).to_string(index=False))

    # Save results
    cross_csv = os.path.join(OUTPUT_DIR, "cross_dataset_results.csv")
    cross_df.to_csv(cross_csv, index=False)
    print(f"\nResults saved to {cross_csv}")

    # ── Interpretation ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  INTERPRETATION")
    print("=" * 70)

    for model_name in ["Random Forest", "XGBoost"]:
        ff_to_celeb = cross_df[(cross_df["direction"] == "FF++ -> Celeb-DF") &
                               (cross_df["model"] == model_name)]["accuracy"].values[0]
        celeb_to_ff = cross_df[(cross_df["direction"] == "Celeb-DF -> FF++") &
                               (cross_df["model"] == model_name)]["accuracy"].values[0]

        print(f"\n  {model_name}:")
        print(f"    FF++ -> Celeb-DF:  {ff_to_celeb:.1%}")
        print(f"    Celeb-DF -> FF++:  {celeb_to_ff:.1%}")

        if celeb_to_ff > ff_to_celeb + 0.05:
            print(f"    --> Celeb-DF model generalises significantly better (+{celeb_to_ff - ff_to_celeb:.1%})")
        elif abs(celeb_to_ff - ff_to_celeb) <= 0.05:
            print(f"    --> Similar cross-dataset performance (gap: {abs(celeb_to_ff - ff_to_celeb):.1%})")
        else:
            print(f"    --> FF++ model generalises better in this direction")


if __name__ == "__main__":
    run()
