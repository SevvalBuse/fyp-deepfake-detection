"""
Combined Comparison Charts
============================
Generates charts comparing FF++-only, Celeb-DF-only, and Combined models
on both held-out test sets.

Run from project root:
    python src/utils/plot_combined_comparison.py
"""

import matplotlib.pyplot as plt
import numpy as np
import os

OUTPUT_DIR = "data/report_visuals"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Held-out test results (from combined_classifier.py — no data leakage) ────

# Random Forest
rf = {
    "FF++ only":     {"ff_test": 0.780, "celeb_test": 0.613},
    "Celeb-DF only": {"ff_test": 0.623, "celeb_test": 0.653},
    "Combined":      {"ff_test": 0.747, "celeb_test": 0.667},
}

# XGBoost
xgb = {
    "FF++ only":     {"ff_test": 0.793, "celeb_test": 0.607},
    "Celeb-DF only": {"ff_test": 0.630, "celeb_test": 0.647},
    "Combined":      {"ff_test": 0.760, "celeb_test": 0.673},
}

conditions = ["FF++ only", "Celeb-DF only", "Combined"]
colors     = ["#4C72B0", "#DD8452", "#55A868"]


# ══════════════════════════════════════════════════════════════════════════════
# CHART 1: Grouped bar — all 3 conditions x 2 test sets (per classifier)
# ══════════════════════════════════════════════════════════════════════════════

def plot_held_out_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, (model_name, data) in zip(axes, [("Random Forest", rf), ("XGBoost", xgb)]):
        x = np.arange(len(conditions))
        w = 0.30

        ff_vals    = [data[c]["ff_test"]    for c in conditions]
        celeb_vals = [data[c]["celeb_test"] for c in conditions]

        b1 = ax.bar(x - w/2, ff_vals,    w, label="Tested on FF++",     color=colors, edgecolor="white")
        b2 = ax.bar(x + w/2, celeb_vals, w, label="Tested on Celeb-DF", color=colors,
                     edgecolor="white", hatch="//", alpha=0.8)

        for bars in [b1, b2]:
            for bar in bars:
                h = bar.get_height()
                ax.annotate(f"{h:.1%}",
                            xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 4), textcoords="offset points",
                            ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(f"{model_name}", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, fontsize=10)
        ax.legend(fontsize=9)
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.yaxis.grid(True, linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    plt.suptitle("Held-Out Accuracy: Single-Dataset vs Combined Training",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "combined_held_out_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 2: Generalization gap — how far apart FF++ and Celeb-DF test scores are
# ══════════════════════════════════════════════════════════════════════════════

def plot_generalization_gap():
    fig, ax = plt.subplots(figsize=(9, 5.5))

    x = np.arange(len(conditions))
    w = 0.30

    rf_gaps  = [abs(rf[c]["ff_test"]  - rf[c]["celeb_test"])  * 100 for c in conditions]
    xgb_gaps = [abs(xgb[c]["ff_test"] - xgb[c]["celeb_test"]) * 100 for c in conditions]

    b1 = ax.bar(x - w/2, rf_gaps,  w, label="Random Forest", color="#4C72B0", edgecolor="white")
    b2 = ax.bar(x + w/2, xgb_gaps, w, label="XGBoost",       color="#DD8452", edgecolor="white")

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}pp",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Gap Between Test Sets (percentage points)", fontsize=11)
    ax.set_title("Generalization Gap: Difference in Accuracy Between FF++ and Celeb-DF Test Sets\n(Lower = more consistent across datasets)",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=11)
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(max(rf_gaps), max(xgb_gaps)) + 6)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "combined_generalization_gap.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 3: Summary table visual — the "money chart" for the report
# ══════════════════════════════════════════════════════════════════════════════

def plot_summary_table():
    """XGBoost-only clean summary — the main chart for the report."""
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(3)  # 3 conditions
    w = 0.25

    ff_vals    = [xgb[c]["ff_test"]    for c in conditions]
    celeb_vals = [xgb[c]["celeb_test"] for c in conditions]
    # Average as a "robustness" metric
    avg_vals   = [(ff_vals[i] + celeb_vals[i]) / 2 for i in range(3)]

    b1 = ax.bar(x - w, ff_vals,    w, label="FF++ test (n=300)",     color="#4C72B0")
    b2 = ax.bar(x,     celeb_vals, w, label="Celeb-DF test (n=150)", color="#DD8452")
    b3 = ax.bar(x + w, avg_vals,   w, label="Average (robustness)",  color="#55A868")

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1%}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("XGBoost: Dataset Diversity Improves Generalization\n(Held-out test sets — no data leakage)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "combined_summary_xgboost.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    plot_held_out_comparison()
    plot_generalization_gap()
    plot_summary_table()
