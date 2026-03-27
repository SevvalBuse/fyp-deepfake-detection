import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import joblib
import warnings
warnings.filterwarnings("ignore")

# --- CONFIG ---
FEATURES_CSV  = "data/output/unified_features.csv"
BIAS_IDS_CSV  = "data/output/bias_audit_ids.csv"   # 300 held-out videos — excluded from training
FEATURES = [
    "chrom_snr", "pos_snr", "chrom_bpm", "pos_bpm",
    "mean_ear", "std_ear", "min_ear",
    "blink_count", "blink_rate_per_min",
    "mean_blink_duration", "std_blink_duration",
    "measured_ita",
]
LABEL = "is_deepfake"
N_SPLITS = 5


def load_data():
    df = pd.read_csv(FEATURES_CSV)
    df = df.dropna(subset=FEATURES + [LABEL])

    # Exclude the 300 held-out bias audit videos from training
    bias_ids = pd.read_csv(BIAS_IDS_CSV)
    bias_join = set(bias_ids["video_id"].apply(lambda x: str(x).split("/")[-1].replace(".mp4", "")))
    df["join_id"] = df["video_id"].apply(lambda x: str(x).split("/")[-1].replace(".mp4", ""))
    df = df[~df["join_id"].isin(bias_join)].reset_index(drop=True)

    X = df[FEATURES].values
    y = df[LABEL].values
    print(f"Dataset: {len(df)} videos ({int(y.sum())} fake, {int((y==0).sum())} real)")
    print(f"(300 bias audit videos excluded from training)")
    return X, y, df


def evaluate_model(name, model, X, y, scale=False):
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
    print(f"  Accuracy:  {acc:.3f} ± {results['test_accuracy'].std():.3f}")
    print(f"  Precision: {prec:.3f} ± {results['test_precision'].std():.3f}")
    print(f"  Recall:    {rec:.3f} ± {results['test_recall'].std():.3f}")
    print(f"  F1 Score:  {f1:.3f} ± {results['test_f1'].std():.3f}")

    return {"model": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def feature_importance(X, y):
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\n--- Feature Importances (Random Forest) ---")
    for feat, imp in importances.items():
        print(f"  {feat}: {imp:.4f}")


def run():
    X, y, df = load_data()

    models = [
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42),          True),
        ("Random Forest",       RandomForestClassifier(n_estimators=200, random_state=42),   False),
        ("XGBoost",             XGBClassifier(n_estimators=200, learning_rate=0.05,
                                              max_depth=4, eval_metric="logloss",
                                              random_state=42, verbosity=0),                  False),
    ]

    print("\n========== BASELINE CLASSIFIER RESULTS ==========")
    summary = []
    for name, model, scale in models:
        result = evaluate_model(name, model, X, y, scale=scale)
        summary.append(result)

    print("\n========== SUMMARY TABLE ==========")
    summary_df = pd.DataFrame(summary).set_index("model")
    print(summary_df.round(3).to_string())

    print("\n========== FEATURE IMPORTANCES ==========")
    feature_importance(X, y)

    # Train final RF and XGBoost on all 1,699 and save for held-out bias audit
    print("\nTraining final models on all 1,699 videos...")
    rf_final = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_final.fit(X, y)
    joblib.dump(rf_final, "data/output/rf_model.pkl")
    print("Saved to data/output/rf_model.pkl")

    xgb_final = XGBClassifier(n_estimators=200, learning_rate=0.05,
                               max_depth=4, eval_metric="logloss",
                               random_state=42, verbosity=0)
    xgb_final.fit(X, y)
    joblib.dump(xgb_final, "data/output/xgb_model.pkl")
    print("Saved to data/output/xgb_model.pkl")


if __name__ == "__main__":
    run()
