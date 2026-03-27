import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

os.makedirs("data/report_visuals/individual_models", exist_ok=True)


def save_heatmap(title, df, filename, vmin=0.35, vmax=0.85):
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
    path = f"data/report_visuals/individual_models/{filename}"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved: {path}")


# ── XGBoost ───────────────────────────────────────────────────────────────────
# Rows = metrics, Columns = Overall + ITA groups
xgb_df = pd.DataFrame({
    "Overall": [0.797, 0.818, 0.780, 0.798],
    "Light":   [0.780, 0.788, 0.788, 0.788],
    "Medium":  [0.740, 0.780, 0.722, 0.750],
    "Dark":    [0.870, 0.886, 0.830, 0.857],
}, index=["Accuracy", "Precision", "Recall", "F1 Score"])

save_heatmap(
    title    = "XGBoost — Held-Out Bias Audit (n=300)",
    df       = xgb_df,
    filename = "xgboost.png",
)

# ── Random Forest ─────────────────────────────────────────────────────────────
rf_df = pd.DataFrame({
    "Overall": [0.770, 0.780, 0.728, 0.749],
    "Light":   [0.760, 0.780, 0.750, 0.765],
    "Medium":  [0.740, 0.780, 0.722, 0.750],
    "Dark":    [0.810, 0.833, 0.745, 0.787],
}, index=["Accuracy", "Precision", "Recall", "F1 Score"])

save_heatmap(
    title    = "Random Forest — Held-Out Bias Audit (n=300)",
    df       = rf_df,
    filename = "random_forest.png",
)

# ── Logistic Regression ───────────────────────────────────────────────────────
lr_df = pd.DataFrame({
    "Overall": [0.746, 0.748, 0.744, 0.745],
    "Light":   [0.737, 0.764, 0.753, 0.751],
    "Medium":  [0.681, 0.732, 0.725, 0.701],
    "Dark":    [0.837, 0.825, 0.849, 0.828],
}, index=["Accuracy", "Precision", "Recall", "F1 Score"])

save_heatmap(
    title    = "Logistic Regression — Performance Heatmap",
    df       = lr_df,
    filename = "logistic_regression.png",
)

# ── CNN-LSTM ──────────────────────────────────────────────────────────────────
cnn_df = pd.DataFrame({
    "Overall": [0.514, 0.524, 0.399, 0.445],
}, index=["Accuracy", "Precision", "Recall", "F1 Score"])

save_heatmap(
    title    = "CNN-LSTM (raw rPPG signals) — Performance Heatmap",
    df       = cnn_df,
    filename = "cnn_lstm.png",
    vmin     = 0.35,
    vmax     = 0.65,   # tighter range so the low values show up clearly in red
)
