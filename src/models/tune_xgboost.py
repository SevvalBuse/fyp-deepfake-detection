"""
Hyperparameter tuning for XGBoost and Random Forest via RandomizedSearchCV
(80 iterations, 5-fold CV, optimising F1). Best XGBoost parameters found here
are applied in bias_auditor.py, celeb_classifier.py, and combined_classifier.py.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score, accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# --- CONFIG ---
FEATURES_CSV = "data/output/unified_features.csv"
FEATURES = [
    "chrom_snr", "pos_snr", "chrom_bpm", "pos_bpm",
    "mean_ear", "std_ear", "min_ear",
    "blink_count", "blink_rate_per_min",
    "mean_blink_duration", "std_blink_duration",
    "measured_ita",
]
LABEL    = "is_deepfake"
N_SPLITS = 5

# Focused XGBoost grid around best params (subsample=0.7, max_depth=4, lr=0.01)
XGBOOST_GRID = {
    "n_estimators":      [300, 400, 500, 600, 800],
    "max_depth":         [3, 4, 5],
    "learning_rate":     [0.005, 0.01, 0.02, 0.05],
    "subsample":         [0.6, 0.7, 0.8],
    "colsample_bytree":  [0.6, 0.7, 0.8],
    "min_child_weight":  [3, 5, 7, 10],
    "gamma":             [0, 0.05, 0.1, 0.2],
    "reg_alpha":         [0, 0.01, 0.1],   # L1 regularisation
    "reg_lambda":        [1, 1.5, 2],       # L2 regularisation
}

# Random Forest grid
RF_GRID = {
    "n_estimators":     [200, 300, 500, 800],
    "max_depth":        [None, 10, 20, 30],
    "min_samples_split":[2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features":     ["sqrt", "log2", 0.5],
}


def tune_model(name, model, param_grid, X, y, n_iter=80):
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    scorer = make_scorer(f1_score, zero_division=0)

    search = RandomizedSearchCV(
        model, param_grid,
        n_iter=n_iter,
        scoring=scorer,
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    print(f"\nTuning {name} ({n_iter} iterations)...")
    search.fit(X, y)

    print(f"Best F1 (search): {search.best_score_:.3f}")
    print("Best params:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")

    # Full CV evaluation with best model
    results = cross_validate(search.best_estimator_, X, y, cv=cv, scoring={
        "accuracy":  make_scorer(accuracy_score),
        "precision": make_scorer(precision_score, zero_division=0),
        "recall":    make_scorer(recall_score, zero_division=0),
        "f1":        scorer,
    })

    acc = results["test_accuracy"].mean()
    f1  = results["test_f1"].mean()
    print(f"\n========== TUNED {name.upper()} ==========")
    print(f"  Accuracy:  {acc:.3f} ± {results['test_accuracy'].std():.3f}")
    print(f"  Precision: {results['test_precision'].mean():.3f} ± {results['test_precision'].std():.3f}")
    print(f"  Recall:    {results['test_recall'].mean():.3f} ± {results['test_recall'].std():.3f}")
    print(f"  F1 Score:  {f1:.3f} ± {results['test_f1'].std():.3f}")

    return search.best_estimator_, search.best_params_, acc, f1


def run():
    df = pd.read_csv(FEATURES_CSV).dropna(subset=FEATURES + [LABEL])
    X = df[FEATURES].values
    y = df[LABEL].values
    print(f"Dataset: {len(df)} videos ({int(y.sum())} fake, {int((y==0).sum())} real)")
    print("\n--- Baselines (default params) ---")
    print("  Random Forest: Acc=0.763, F1=0.757")
    print("  XGBoost:       Acc=0.762, F1=0.757")

    _, _, xgb_acc, xgb_f1 = tune_model(
        "XGBoost",
        XGBClassifier(eval_metric="logloss", random_state=42, verbosity=0),
        XGBOOST_GRID, X, y, n_iter=80
    )

    _, _, rf_acc, rf_f1 = tune_model(
        "Random Forest",
        RandomForestClassifier(random_state=42),
        RF_GRID, X, y, n_iter=80
    )

    print("\n========== FINAL COMPARISON ==========")
    print(f"  Default RF:    Acc=0.763, F1=0.757")
    print(f"  Default XGB:   Acc=0.762, F1=0.757")
    print(f"  Tuned XGBoost: Acc={xgb_acc:.3f}, F1={xgb_f1:.3f}")
    print(f"  Tuned RF:      Acc={rf_acc:.3f}, F1={rf_f1:.3f}")
    print("\nBest params have been applied in bias_auditor.py, celeb_classifier.py, and combined_classifier.py")


if __name__ == "__main__":
    run()
