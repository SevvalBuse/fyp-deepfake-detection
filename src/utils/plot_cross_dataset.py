"""
Cross-Dataset Comparison Chart
===============================
Generates a grouped bar chart comparing within-dataset and cross-dataset
performance for FF++ and Celeb-DF trained models.

Run from project root:
    python src/utils/plot_cross_dataset.py
"""

import matplotlib.pyplot as plt
import numpy as np
import os

OUTPUT_DIR = "data/report_visuals"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Results data ──────────────────────────────────────────────────────────────

models = ["Random Forest", "XGBoost"]

# Accuracy values from the experiments
ff_within    = [0.755, 0.739]   # FF++ model tested on FF++ (5-fold CV)
celeb_within = [0.598, 0.618]   # Celeb-DF model tested on Celeb-DF (5-fold CV)
ff_to_celeb  = [0.544, 0.561]   # FF++ model tested on Celeb-DF
celeb_to_ff  = [0.618, 0.621]   # Celeb-DF model tested on FF++

# ── Chart 1: Full 4-bar comparison ───────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5.5))

x = np.arange(len(models))
w = 0.18

conditions = [
    ("FF++ within-dataset",      ff_within,    "#4C72B0"),
    ("FF++ → Celeb-DF",          ff_to_celeb,  "#4C72B0"),
    ("Celeb-DF within-dataset",  celeb_within, "#DD8452"),
    ("Celeb-DF → FF++",          celeb_to_ff,  "#DD8452"),
]

hatches = ["", "//", "", "//"]

for i, (label, vals, color) in enumerate(conditions):
    offset = (i - 1.5) * w
    bars = ax.bar(x + offset, vals, w, label=label, color=color,
                  hatch=hatches[i], edgecolor="white", linewidth=0.5)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.1%}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_ylim(0, 1.0)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("Within-Dataset vs Cross-Dataset Accuracy",
             fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12)
ax.legend(fontsize=9, loc="upper right")
ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.5, label=None)
ax.text(x[-1] + 0.45, 0.505, "chance level", fontsize=8, color="grey", ha="right")
ax.yaxis.grid(True, linestyle="--", alpha=0.3)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
path1 = os.path.join(OUTPUT_DIR, "cross_dataset_comparison.png")
plt.savefig(path1, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {path1}")


# ── Chart 2: Generalization gap (drop from within to cross) ─────────────────

fig2, ax2 = plt.subplots(figsize=(8, 5))

ff_drop    = [ff_within[i] - ff_to_celeb[i] for i in range(len(models))]
celeb_drop = [celeb_within[i] - celeb_to_ff[i] for i in range(len(models))]

x2 = np.arange(len(models))
w2 = 0.30

b1 = ax2.bar(x2 - w2/2, [d * 100 for d in ff_drop], w2,
             label="FF++ model (drop to Celeb-DF)", color="#e74c3c")
b2 = ax2.bar(x2 + w2/2, [d * 100 for d in celeb_drop], w2,
             label="Celeb-DF model (drop to FF++)", color="#2ecc71")

for bar in list(b1) + list(b2):
    height = bar.get_height()
    if height >= 0:
        sign = "−"
        va = "bottom"
        offset = 4
    else:
        sign = "+"
        va = "top"
        offset = -4
    ax2.annotate(f"{sign}{abs(height):.1f}pp",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, offset), textcoords="offset points",
                ha="center", va=va, fontsize=11, fontweight="bold")

ax2.set_ylabel("Accuracy Change (percentage points)", fontsize=11)
ax2.set_title("Generalization Gap: Accuracy Change When Testing Cross-Dataset",
              fontsize=13, fontweight="bold")
ax2.set_xticks(x2)
ax2.set_xticklabels(models, fontsize=12)
ax2.legend(fontsize=10)
ax2.yaxis.grid(True, linestyle="--", alpha=0.3)
ax2.set_axisbelow(True)
ax2.axhline(0, color="black", linewidth=0.8)
low = min(min(celeb_drop), 0) * 100 - 5
high = max(ff_drop) * 100 + 8
ax2.set_ylim(low, high)
for spine in ["top", "right"]:
    ax2.spines[spine].set_visible(False)

plt.tight_layout()
path2 = os.path.join(OUTPUT_DIR, "generalization_gap.png")
plt.savefig(path2, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {path2}")
