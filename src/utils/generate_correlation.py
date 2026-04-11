"""
Generates two visualisations from unified_features.csv: a full feature correlation
heatmap and a bar chart showing each feature's Pearson correlation with the
deepfake label. Output saved to data/report_visuals/.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

os.makedirs("data/report_visuals", exist_ok=True)

FEATURES_CSV = "data/output/unified_features.csv"

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

COLS = list(FEATURE_LABELS.keys())


def run():
    df = pd.read_csv(FEATURES_CSV)
    df = df[COLS].dropna()

    print(f"Computing correlation matrix on {len(df)} videos...")

    corr = df.rename(columns=FEATURE_LABELS).corr()

    # --- Full heatmap ---
    plt.figure(figsize=(13, 10))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        annot_kws={"size": 8},
    )
    plt.title(f"Feature Correlation Matrix — Deepfake Detection (n={len(df)})",
              fontsize=13, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    path1 = "data/report_visuals/correlation_matrix_final.png"
    plt.savefig(path1, dpi=300, bbox_inches="tight")
    print(f"Saved to {path1}")

    # --- Correlation with is_deepfake only (bar chart) ---
    target_corr = corr["Is Deepfake"].drop("Is Deepfake").sort_values()

    colors = ["#C44E52" if v < 0 else "#4C72B0" for v in target_corr]

    plt.figure(figsize=(9, 5))
    plt.barh(target_corr.index, target_corr.values, color=colors, edgecolor="white")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Pearson Correlation with Is Deepfake", fontsize=11)
    plt.title("Feature Correlation with Deepfake Label", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path2 = "data/report_visuals/feature_target_correlation.png"
    plt.savefig(path2, dpi=300, bbox_inches="tight")
    print(f"Saved to {path2}")

    print("\nCorrelations with is_deepfake:")
    print(target_corr.sort_values(key=abs, ascending=False).to_string())


if __name__ == "__main__":
    run()
