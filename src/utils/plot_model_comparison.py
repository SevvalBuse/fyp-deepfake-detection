import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("data/report_visuals", exist_ok=True)

# --- Results from 5-fold CV on 1699-video training set (300 held out for bias audit) ---
models = [
    "Logistic\nRegression",
    "Random\nForest",
    "XGBoost",
    "1D CNN\n(raw signals)",
    "CNN-LSTM\n(raw signals)",
]

accuracy  = [0.744, 0.756, 0.739, 0.555, 0.514]
precision = [0.750, 0.770, 0.747, 0.598, 0.524]
recall    = [0.729, 0.728, 0.719, 0.415, 0.399]
f1        = [0.739, 0.749, 0.732, 0.467, 0.445]

x = np.arange(len(models))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 6))

bars1 = ax.bar(x - 1.5 * width, accuracy,  width, label="Accuracy",  color="#4C72B0")
bars2 = ax.bar(x - 0.5 * width, precision, width, label="Precision", color="#55A868")
bars3 = ax.bar(x + 0.5 * width, recall,    width, label="Recall",    color="#C44E52")
bars4 = ax.bar(x + 1.5 * width, f1,        width, label="F1 Score",  color="#8172B2")

# Value labels on top of each bar
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        ax.annotate(
            f"{bar.get_height():.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=7.5
        )

ax.set_ylim(0, 1.0)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Model Comparison — Deepfake Detection (5-Fold CV, n=1699)", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.legend(fontsize=10)
ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="Random baseline")
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)

plt.tight_layout()
save_path = "data/report_visuals/model_comparison.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved to {save_path}")

# --- Accuracy-only bar chart ---
fig2, ax2 = plt.subplots(figsize=(9, 5))
bars = ax2.bar(models, accuracy, color="#4C72B0", edgecolor="white", width=0.5)

for bar in bars:
    ax2.annotate(
        f"{bar.get_height():.3f}",
        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
        xytext=(0, 4), textcoords="offset points",
        ha="center", va="bottom", fontsize=11, fontweight="bold"
    )

ax2.set_ylim(0, 1.0)
ax2.set_ylabel("Accuracy", fontsize=12)
ax2.set_title("Classification Accuracy by Model (5-Fold CV, n=1699)", fontsize=13, fontweight="bold")
ax2.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
ax2.text(4.5, 0.515, "Random baseline (0.5)", ha="right", fontsize=9, color="gray")
ax2.yaxis.grid(True, linestyle="--", alpha=0.4)
ax2.set_axisbelow(True)


plt.tight_layout()
save_path2 = "data/report_visuals/model_accuracy_only.png"
plt.savefig(save_path2, dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved to {save_path2}")
