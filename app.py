"""
Deepfake Detection Dashboard
Streamlit app — entry point for demo and poster presentation.
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ──────────────────────────────────────────────────────────────────
FEATURES_CSV = "data/output/unified_features.csv"
FEATURES = [
    "chrom_snr", "pos_snr", "chrom_bpm", "pos_bpm",
    "mean_ear", "std_ear", "min_ear",
    "blink_count", "blink_rate_per_min",
    "mean_blink_duration", "std_blink_duration",
    "measured_ita",
]
LABEL = "is_deepfake"

ITA_THRESHOLDS = {"Dark": (-999, 10), "Medium": (10, 41), "Light": (41, 999)}

FEATURE_LABELS = {
    "chrom_snr":           "CHROM SNR (dB)",
    "pos_snr":             "POS SNR (dB)",
    "chrom_bpm":           "CHROM Heart Rate (BPM)",
    "pos_bpm":             "POS Heart Rate (BPM)",
    "mean_ear":            "Mean EAR",
    "std_ear":             "Std EAR",
    "min_ear":             "Min EAR",
    "blink_count":         "Blink Count",
    "blink_rate_per_min":  "Blink Rate (per min)",
    "mean_blink_duration": "Mean Blink Duration (frames)",
    "std_blink_duration":  "Std Blink Duration",
    "measured_ita":        "ITA (Skin Tone)",
}


# ── DATA & MODEL ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(FEATURES_CSV)
    df = df.dropna(subset=FEATURES + [LABEL])
    df["skin_tone"] = df["measured_ita"].apply(ita_to_group)
    return df


@st.cache_resource
def train_model(df):
    X = df[FEATURES].values
    y = df[LABEL].values
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X, y)
    return rf


def ita_to_group(ita):
    if ita <= 10:
        return "Dark"
    elif ita <= 41:
        return "Medium"
    else:
        return "Light"


# ── PAGE SETUP ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deepfake Detection Dashboard",
    page_icon="🔍",
    layout="wide",
)

st.title("Deepfake Detection Dashboard")
st.caption("Physiological signal analysis (rPPG + Eye-Blink) with skin tone bias audit")

df = load_data()
model = train_model(df)

tab1, tab2 = st.tabs(["Analyse Video", "Bias Audit"])


# ── TAB 1: ANALYSE ───────────────────────────────────────────────────────────
with tab1:
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("Select a Video")

        # Filter controls
        label_filter = st.radio("Filter by label", ["All", "Real only", "Fake only"])
        tone_filter = st.multiselect(
            "Filter by skin tone",
            ["Dark", "Medium", "Light"],
            default=["Dark", "Medium", "Light"],
        )

        filtered = df.copy()
        if label_filter == "Real only":
            filtered = filtered[filtered[LABEL] == 0]
        elif label_filter == "Fake only":
            filtered = filtered[filtered[LABEL] == 1]
        if tone_filter:
            filtered = filtered[filtered["skin_tone"].isin(tone_filter)]

        if filtered.empty:
            st.warning("No videos match the selected filters.")
            st.stop()

        video_options = filtered["video_id"].tolist()
        selected_video = st.selectbox("Video ID", video_options)

        row = filtered[filtered["video_id"] == selected_video].iloc[0]

        st.divider()
        st.markdown("**Ground Truth**")
        gt_label = "FAKE" if row[LABEL] == 1 else "REAL"
        gt_colour = "red" if row[LABEL] == 1 else "green"
        st.markdown(f":<{gt_colour}>[{gt_label}]", unsafe_allow_html=False)
        st.markdown(
            f"**Skin Tone:** {row['skin_tone']}  \n"
            f"**ITA value:** {row['measured_ita']:.2f}"
        )

    with col_right:
        st.subheader("Prediction")

        feat_values = row[FEATURES].values.reshape(1, -1)
        prob = model.predict_proba(feat_values)[0]
        pred = model.predict(feat_values)[0]
        confidence = prob[int(pred)]

        pred_label = "FAKE" if pred == 1 else "REAL"
        correct = pred == row[LABEL]

        if pred == 1:
            st.error(f"DEEPFAKE DETECTED — confidence {confidence:.0%}")
        else:
            st.success(f"REAL VIDEO — confidence {confidence:.0%}")

        verdict = "Correct" if correct else "Incorrect"
        st.caption(f"Prediction vs ground truth: **{verdict}**")

        # Probability bar
        prob_df = pd.DataFrame(
            {"Label": ["Real", "Fake"], "Probability": [prob[0], prob[1]]}
        )
        fig_prob, ax_prob = plt.subplots(figsize=(5, 1.4))
        colours = ["#2ecc71", "#e74c3c"]
        ax_prob.barh(prob_df["Label"], prob_df["Probability"], color=colours)
        ax_prob.set_xlim(0, 1)
        ax_prob.axvline(0.5, color="grey", linestyle="--", linewidth=0.8)
        ax_prob.set_xlabel("Probability")
        for spine in ["top", "right"]:
            ax_prob.spines[spine].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_prob)
        plt.close(fig_prob)

        st.divider()
        st.subheader("Extracted Features")

        feat_display = {FEATURE_LABELS[f]: round(row[f], 4) for f in FEATURES}
        feat_df = pd.DataFrame(feat_display.items(), columns=["Feature", "Value"])

        col_a, col_b = st.columns(2)
        half = len(feat_df) // 2
        col_a.dataframe(feat_df.iloc[:half].set_index("Feature"), use_container_width=True)
        col_b.dataframe(feat_df.iloc[half:].set_index("Feature"), use_container_width=True)


# ── TAB 2: BIAS AUDIT ────────────────────────────────────────────────────────
with tab2:
    st.subheader("Bias Audit — Accuracy by Skin Tone Group")
    st.caption(
        "Skin tone determined objectively via ITA (Individual Typology Angle). "
        "Dark: ITA ≤ 10 | Medium: 10 < ITA ≤ 41 | Light: ITA > 41"
    )

    # Compute predictions for whole dataset
    X_all = df[FEATURES].values
    y_all = df[LABEL].values
    preds_all = model.predict(X_all)
    df_audit = df.copy()
    df_audit["predicted"] = preds_all
    df_audit["correct"] = (df_audit["predicted"] == df_audit[LABEL]).astype(int)

    # Overall metrics
    overall_acc = df_audit["correct"].mean()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Overall Accuracy", f"{overall_acc:.1%}")
    col2.metric("Total Videos", len(df_audit))
    col3.metric("Real Videos", int((df_audit[LABEL] == 0).sum()))
    col4.metric("Fake Videos", int((df_audit[LABEL] == 1).sum()))

    st.divider()

    # Per skin tone group
    groups = ["Dark", "Medium", "Light"]
    group_stats = []
    for g in groups:
        sub = df_audit[df_audit["skin_tone"] == g]
        if len(sub) == 0:
            continue
        acc = sub["correct"].mean()
        n = len(sub)
        n_real = int((sub[LABEL] == 0).sum())
        n_fake = int((sub[LABEL] == 1).sum())
        group_stats.append(
            {"Skin Tone": g, "N Videos": n, "Real": n_real, "Fake": n_fake, "Accuracy": acc}
        )

    stats_df = pd.DataFrame(group_stats)

    col_chart, col_table = st.columns([3, 2])

    with col_chart:
        fig_bias, ax = plt.subplots(figsize=(6, 3.5))
        colours_bias = {"Dark": "#5d4e37", "Medium": "#c68642", "Light": "#f5cba7"}
        bar_colours = [colours_bias[g] for g in stats_df["Skin Tone"]]
        bars = ax.bar(stats_df["Skin Tone"], stats_df["Accuracy"], color=bar_colours, width=0.5)
        ax.axhline(overall_acc, color="steelblue", linestyle="--", linewidth=1.2, label=f"Overall ({overall_acc:.1%})")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        ax.set_title("Detection Accuracy by Skin Tone Group")
        ax.legend(fontsize=9)
        for bar, val in zip(bars, stats_df["Accuracy"]):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.1%}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_bias)
        plt.close(fig_bias)

    with col_table:
        st.markdown("**Per-group breakdown**")
        display_df = stats_df.copy()
        display_df["Accuracy"] = display_df["Accuracy"].apply(lambda x: f"{x:.1%}")
        st.dataframe(display_df.set_index("Skin Tone"), use_container_width=True)

        max_acc = stats_df["Accuracy"].max()
        min_acc = stats_df["Accuracy"].min()
        gap = max_acc - min_acc
        best = stats_df.loc[stats_df["Accuracy"].idxmax(), "Skin Tone"]
        worst = stats_df.loc[stats_df["Accuracy"].idxmin(), "Skin Tone"]

        st.markdown("**Fairness summary**")
        st.markdown(
            f"- Best performing group: **{best}** ({max_acc:.1%})  \n"
            f"- Worst performing group: **{worst}** ({min_acc:.1%})  \n"
            f"- Accuracy gap: **{gap:.1%}**"
        )
        if gap < 0.05:
            st.success("Gap < 5% — model is relatively fair across skin tones.")
        elif gap < 0.10:
            st.warning("Gap 5–10% — moderate disparity detected.")
        else:
            st.error("Gap > 10% — significant bias detected.")

    st.divider()
    st.subheader("Feature Importance (Random Forest)")

    importances = pd.Series(model.feature_importances_, index=FEATURES)
    importances = importances.sort_values(ascending=True)

    fig_imp, ax_imp = plt.subplots(figsize=(7, 4))
    bars_imp = ax_imp.barh(
        [FEATURE_LABELS[f] for f in importances.index],
        importances.values,
        color="steelblue",
    )
    ax_imp.set_xlabel("Importance")
    ax_imp.set_title("Feature Importances")
    for spine in ["top", "right"]:
        ax_imp.spines[spine].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig_imp)
    plt.close(fig_imp)
