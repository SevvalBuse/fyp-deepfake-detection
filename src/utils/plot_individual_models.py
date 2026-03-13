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
    "Overall": [0.770, 0.774, 0.764, 0.767],
    "Light":   [0.802, 0.861, 0.774, 0.810],
    "Medium":  [0.651, 0.684, 0.674, 0.672],
    "Dark":    [0.790, 0.772, 0.785, 0.773],
}, index=["Accuracy", "Precision", "Recall", "F1 Score"])

save_heatmap(
    title    = "XGBoost (tuned) — Performance Heatmap",
    df       = xgb_df,
    filename = "xgboost.png",
)

# ── Random Forest ─────────────────────────────────────────────────────────────
rf_df = pd.DataFrame({
    "Overall": [0.767, 0.773, 0.760, 0.765],
    "Light":   [0.802, 0.865, 0.774, 0.811],
    "Medium":  [0.690, 0.712, 0.712, 0.709],
    "Dark":    [0.782, 0.761, 0.785, 0.766],
}, index=["Accuracy", "Precision", "Recall", "F1 Score"])

save_heatmap(
    title    = "Random Forest (tuned) — Performance Heatmap",
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
