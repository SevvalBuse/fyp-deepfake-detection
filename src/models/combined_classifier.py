"""
Combined Classifier: FF++ + Celeb-DF (Fixed — No Data Leakage)
================================================================
Proper evaluation with held-out test sets:
  - FF++ test set:     300 bias audit videos (never in any training set)
  - Celeb-DF test set: 150 held-out videos (never in any training set)

Three training conditions tested on BOTH held-out sets:
  1. FF++ only     (trained on ~1,700 FF++ videos)
  2. Celeb-DF only (trained on 1,550 Celeb-DF videos)
  3. Combined      (trained on ~1,700 FF++ + 1,550 Celeb-DF)

Also runs 5-fold CV for each condition as a secondary metric.

Run from project root:
    python src/models/combined_classifier.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    make_scorer, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
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

CELEB_N_REAL = 850
CELEB_N_FAKE = 850
CELEB_TEST_REAL = 75   # held out from training
CELEB_TEST_FAKE = 75   # held out from training
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


# ── DATA LOADING (with proper train/test splits) ─────────────────────────────

def load_all_data():
    """
    Returns:
      ff_train:     FF++ training set (~1,700 — excludes 300 bias audit)
      ff_test:      FF++ held-out test set (300 bias audit videos)
      celeb_train:  Celeb-DF training set (775 real + 775 fake = 1,550)
      celeb_test:   Celeb-DF held-out test set (75 real + 75 fake = 150)
    """
    # ── FF++ ──
    ff_df = pd.read_csv(FF_FEATURES_CSV)
    ff_df = ff_df.dropna(subset=FEATURES + [LABEL])

    bias_ids = pd.read_csv(FF_BIAS_IDS_CSV)
    bias_join = set(bias_ids["video_id"].apply(
        lambda x: str(x).split("/")[-1].replace(".mp4", "")
    ))
    ff_df["join_id"] = ff_df["video_id"].apply(
        lambda x: str(x).split("/")[-1].replace(".mp4", "")
    )

    ff_train = ff_df[~ff_df["join_id"].isin(bias_join)].reset_index(drop=True)
    ff_test  = ff_df[ff_df["join_id"].isin(bias_join)].reset_index(drop=True)

    # ── Celeb-DF ──
    celeb_df = pd.read_csv(CELEB_FEATURES_CSV)
    celeb_df = celeb_df.dropna(subset=FEATURES + [LABEL])

    # Sample 850 real + 850 fake (same as before)
    celeb_real = celeb_df[celeb_df[LABEL] == 0].sample(n=CELEB_N_REAL, random_state=SEED)
    celeb_fake = celeb_df[celeb_df[LABEL] == 1].sample(n=CELEB_N_FAKE, random_state=SEED)
    celeb_sampled = pd.concat([celeb_real, celeb_fake]).reset_index(drop=True)

    # Split into train + held-out test (using a second random split)
    celeb_test_real  = celeb_sampled[celeb_sampled[LABEL] == 0].sample(
        n=CELEB_TEST_REAL, random_state=SEED + 1)
    celeb_test_fake  = celeb_sampled[celeb_sampled[LABEL] == 1].sample(
        n=CELEB_TEST_FAKE, random_state=SEED + 1)
    celeb_test = pd.concat([celeb_test_real, celeb_test_fake]).reset_index(drop=True)

    celeb_test_ids = set(celeb_test.index)
    # Remove test videos from training
    celeb_train_real = celeb_sampled[celeb_sampled[LABEL] == 0].drop(celeb_test_real.index)
    celeb_train_fake = celeb_sampled[celeb_sampled[LABEL] == 1].drop(celeb_test_fake.index)
    celeb_train = pd.concat([celeb_train_real, celeb_train_fake]).reset_index(drop=True)

    return ff_train, ff_test, celeb_train, celeb_test


# ── EVALUATION HELPERS ────────────────────────────────────────────────────────

def cv_evaluate(name, model, X, y, scale=False):
    X_input = X.copy()
    if scale:
        X_input = StandardScaler().fit_transform(X_input)

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

    print(f"\n--- {name} (5-fold CV) ---")
    print(f"  Accuracy:  {acc:.3f} +/- {results['test_accuracy'].std():.3f}")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall:    {rec:.3f}")
    print(f"  F1 Score:  {f1:.3f}")

    return {"model": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def test_model(model, X_test, y_test, train_label, test_label, model_name):
    preds = model.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    prec  = precision_score(y_test, preds, zero_division=0)
    rec   = recall_score(y_test, preds, zero_division=0)
    f1    = f1_score(y_test, preds, zero_division=0)
    cm    = confusion_matrix(y_test, preds)

    print(f"\n--- {train_label} -> {test_label}: {model_name} ---")
    print(f"  Accuracy:  {acc:.3f}")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall:    {rec:.3f}")
    print(f"  F1 Score:  {f1:.3f}")
    print(f"  TN={cm[0,0]:4d}  FP={cm[0,1]:4d}  |  FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")

    return {
        "trained_on": train_label, "tested_on": test_label, "model": model_name,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run():
    print("=" * 70)
    print("  COMBINED CLASSIFIER: FF++ + CELEB-DF")
    print("  (Proper held-out evaluation — no data leakage)")
    print("=" * 70)

    # Load with proper train/test splits
    ff_train, ff_test, celeb_train, celeb_test = load_all_data()

    print(f"\n--- Dataset splits ---")
    print(f"FF++ train:      {len(ff_train)} videos "
          f"({int(ff_train[LABEL].sum())} fake, {int((ff_train[LABEL]==0).sum())} real)")
    print(f"FF++ test:       {len(ff_test)} videos "
          f"({int(ff_test[LABEL].sum())} fake, {int((ff_test[LABEL]==0).sum())} real)  [bias audit held-out]")
    print(f"Celeb-DF train:  {len(celeb_train)} videos "
          f"({int(celeb_train[LABEL].sum())} fake, {int((celeb_train[LABEL]==0).sum())} real)")
    print(f"Celeb-DF test:   {len(celeb_test)} videos "
          f"({int(celeb_test[LABEL].sum())} fake, {int((celeb_test[LABEL]==0).sum())} real)  [held-out]")

    # Combined training set
    combined_train = pd.concat([ff_train, celeb_train], ignore_index=True)
    print(f"Combined train:  {len(combined_train)} videos "
          f"({int(combined_train[LABEL].sum())} fake, {int((combined_train[LABEL]==0).sum())} real)")

    # Prepare arrays
    X_ff_train   = ff_train[FEATURES].values
    y_ff_train   = ff_train[LABEL].values
    X_ff_test    = ff_test[FEATURES].values
    y_ff_test    = ff_test[LABEL].values
    X_cel_train  = celeb_train[FEATURES].values
    y_cel_train  = celeb_train[LABEL].values
    X_cel_test   = celeb_test[FEATURES].values
    y_cel_test   = celeb_test[LABEL].values
    X_comb_train = combined_train[FEATURES].values
    y_comb_train = combined_train[LABEL].values

    # ══════════════════════════════════════════════════════════════════════
    # PART 1: 5-Fold CV for each training condition
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PART 1: 5-FOLD CV (within-dataset evaluation)")
    print("=" * 70)

    for train_name, X, y in [
        ("FF++ only",     X_ff_train,   y_ff_train),
        ("Celeb-DF only", X_cel_train,  y_cel_train),
        ("Combined",      X_comb_train, y_comb_train),
    ]:
        for model_name, model, scale in [
            ("RF",  RandomForestClassifier(n_estimators=200, random_state=42), False),
            ("XGB", XGBClassifier(**BEST_XGB_PARAMS), False),
        ]:
            cv_evaluate(f"{train_name} {model_name}", model, X, y, scale=scale)

    # ══════════════════════════════════════════════════════════════════════
    # PART 2: Held-out test — no model sees its own test data
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PART 2: HELD-OUT TEST (no data leakage)")
    print("=" * 70)

    all_results = []

    for model_name, make_model in [
        ("Random Forest", lambda: RandomForestClassifier(n_estimators=200, random_state=42)),
        ("XGBoost",       lambda: XGBClassifier(**BEST_XGB_PARAMS)),
    ]:
        # Train 3 models
        ff_model = make_model()
        ff_model.fit(X_ff_train, y_ff_train)

        celeb_model = make_model()
        celeb_model.fit(X_cel_train, y_cel_train)

        combined_model = make_model()
        combined_model.fit(X_comb_train, y_comb_train)

        # Save combined models
        if model_name == "Random Forest":
            joblib.dump(combined_model, os.path.join(OUTPUT_DIR, "combined_rf_model.pkl"))
        else:
            joblib.dump(combined_model, os.path.join(OUTPUT_DIR, "combined_xgb_model.pkl"))

        # Test each model on BOTH held-out test sets
        for trained_on, model in [("FF++ only", ff_model),
                                   ("Celeb-DF only", celeb_model),
                                   ("Combined", combined_model)]:
            all_results.append(test_model(model, X_ff_test, y_ff_test,
                                          trained_on, "FF++ test", model_name))
            all_results.append(test_model(model, X_cel_test, y_cel_test,
                                          trained_on, "Celeb-DF test", model_name))

    # ══════════════════════════════════════════════════════════════════════
    # PART 3: SUMMARY TABLE
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  HELD-OUT COMPARISON TABLE")
    print("  (Every number below is on data the model NEVER saw during training)")
    print("=" * 70)

    results_df = pd.DataFrame(all_results)
    results_df = results_df[["trained_on", "tested_on", "model", "accuracy", "precision", "recall", "f1"]]

    for model_name in ["Random Forest", "XGBoost"]:
        print(f"\n--- {model_name} ---")
        sub = results_df[results_df["model"] == model_name].copy()
        sub = sub[["trained_on", "tested_on", "accuracy", "f1"]]
        print(sub.round(3).to_string(index=False))

    # Save
    csv_path = os.path.join(OUTPUT_DIR, "combined_comparison_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # ══════════════════════════════════════════════════════════════════════
    # PART 4: KEY TAKEAWAYS
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  KEY TAKEAWAYS")
    print("=" * 70)

    for model_name in ["Random Forest", "XGBoost"]:
        sub = results_df[results_df["model"] == model_name]

        ff_on_ff    = sub[(sub["trained_on"] == "FF++ only")     & (sub["tested_on"] == "FF++ test")]["accuracy"].values[0]
        ff_on_cel   = sub[(sub["trained_on"] == "FF++ only")     & (sub["tested_on"] == "Celeb-DF test")]["accuracy"].values[0]
        cel_on_ff   = sub[(sub["trained_on"] == "Celeb-DF only") & (sub["tested_on"] == "FF++ test")]["accuracy"].values[0]
        cel_on_cel  = sub[(sub["trained_on"] == "Celeb-DF only") & (sub["tested_on"] == "Celeb-DF test")]["accuracy"].values[0]
        com_on_ff   = sub[(sub["trained_on"] == "Combined")      & (sub["tested_on"] == "FF++ test")]["accuracy"].values[0]
        com_on_cel  = sub[(sub["trained_on"] == "Combined")      & (sub["tested_on"] == "Celeb-DF test")]["accuracy"].values[0]

        print(f"\n  {model_name}:")
        print(f"    {'Trained on':<16} {'-> FF++ test':>12} {'-> Celeb-DF test':>16}")
        print(f"    {'FF++ only':<16} {ff_on_ff:>11.1%} {ff_on_cel:>15.1%}")
        print(f"    {'Celeb-DF only':<16} {cel_on_ff:>11.1%} {cel_on_cel:>15.1%}")
        print(f"    {'Combined':<16} {com_on_ff:>11.1%} {com_on_cel:>15.1%}")

        # Combined should ideally be competitive on BOTH test sets
        print(f"\n    Combined vs best single-dataset on each test set:")
        ff_gain  = com_on_ff  - max(ff_on_ff, cel_on_ff)
        cel_gain = com_on_cel - max(ff_on_cel, cel_on_cel)
        print(f"      FF++ test:     {'+' if ff_gain >= 0 else ''}{ff_gain:.1%}")
        print(f"      Celeb-DF test: {'+' if cel_gain >= 0 else ''}{cel_gain:.1%}")

    print("\n  Saved combined models: combined_rf_model.pkl, combined_xgb_model.pkl")


if __name__ == "__main__":
    run()
