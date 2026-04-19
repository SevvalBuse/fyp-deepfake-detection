"""
Deepfake Detection Dashboard
Streamlit app — entry point for demo and poster presentation.
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import tempfile
import warnings
warnings.filterwarnings("ignore")

import cv2
import dlib
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
from scipy.spatial import distance as dist

# ── CONFIG ──────────────────────────────────────────────────────────────────
FF_FEATURES_CSV    = "data/output/unified_features.csv"
CELEB_FEATURES_CSV = "data/output/celeb_unified_features.csv"
SHAPE_PREDICTOR    = "src/shape_predictor_68_face_landmarks.dat"

MODEL_PATHS = {
    "FF++ XGBoost":     "data/output/xgb_model.pkl",
    "Celeb-DF XGBoost": "data/output/celeb_xgb_model.pkl",
    "Combined XGBoost": "data/output/combined_xgb_model.pkl",
    "FF++ RF":          "data/output/rf_model.pkl",
    "Celeb-DF RF":      "data/output/celeb_rf_model.pkl",
    "Combined RF":      "data/output/combined_rf_model.pkl",
}

FEATURES = [
    "chrom_snr", "pos_snr", "chrom_bpm", "pos_bpm",
    "mean_ear", "std_ear", "min_ear",
    "blink_count", "blink_rate_per_min",
    "mean_blink_duration", "std_blink_duration",
    "measured_ita",
]
LABEL = "is_deepfake"

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

EAR_THRESHOLD    = 0.21
MIN_BLINK_FRAMES = 2


# ── LOAD MODELS & DATA ─────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    for name, path in MODEL_PATHS.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models


@st.cache_resource
def load_dlib():
    det = dlib.get_frontal_face_detector()
    pred = dlib.shape_predictor(SHAPE_PREDICTOR)
    return det, pred


@st.cache_data
def load_data():
    # FF++
    ff_df = pd.read_csv(FF_FEATURES_CSV)
    ff_df = ff_df.dropna(subset=FEATURES + [LABEL])
    ff_df["dataset"] = "FF++"

    # Celeb-DF
    celeb_df = pd.read_csv(CELEB_FEATURES_CSV)
    celeb_df = celeb_df.dropna(subset=FEATURES + [LABEL])
    celeb_df["dataset"] = "Celeb-DF"

    df = pd.concat([ff_df, celeb_df], ignore_index=True)
    df["skin_tone"] = df["measured_ita"].apply(ita_to_group)
    return df


def ita_to_group(ita):
    if ita <= 10:
        return "Dark"
    elif ita <= 41:
        return "Medium"
    return "Light"


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION — same functions as the training pipeline
# ══════════════════════════════════════════════════════════════════════════════

def get_largest_face(faces):
    return max(faces, key=lambda r: (r.right() - r.left()) * (r.bottom() - r.top()))


def get_refined_rois(frame, shape):
    h, w = frame.shape[:2]
    def clamp(y1, y2, x1, x2):
        return max(0, int(y1)), min(h, int(y2)), max(0, int(x1)), min(w, int(x2))

    face_height = shape.part(8).y - shape.part(27).y
    if face_height <= 0:
        return None, None, None

    f_height = max(4, int(face_height * 0.10))
    f_bottom = min(shape.part(19).y, shape.part(24).y) - int(face_height * 0.02)
    f_top    = f_bottom - f_height
    forehead = clamp(f_top, f_bottom, shape.part(18).x, shape.part(25).x)

    side      = max(4, int(face_height * 0.08))
    cheek_down = int(face_height * 0.02)
    l_x = (shape.part(2).x  + shape.part(31).x) // 2
    l_y = (shape.part(40).y + shape.part(31).y) // 2 + cheek_down
    r_x = (shape.part(14).x + shape.part(35).x) // 2
    r_y = (shape.part(47).y + shape.part(35).y) // 2 + cheek_down

    return forehead, clamp(l_y-side, l_y+side, l_x-side, l_x+side), \
           clamp(r_y-side, r_y+side, r_x-side, r_x+side)


def roi_mean_bgr(frame, coords):
    y1, y2, x1, x2 = coords
    if y2 <= y1 or x2 <= x1:
        return None
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    return cv2.mean(roi)[:3]


def calculate_ita(l_ocv, b_ocv):
    # Correct ITA from raw OpenCV LAB values (8-bit image).
    # OpenCV stores L in [0,255] and b in [0,255] (shifted by +128).
    # Standard ITA formula requires L in [0,100] and b in [-128,127].
    l_std = l_ocv * (100.0 / 255.0)
    b_std = b_ocv - 128.0
    return float(np.arctan2((l_std - 50.0), b_std) * (180.0 / np.pi))


def apply_butterworth(signal, fs=30.0, order=5):
    nyq  = 0.5 * fs
    low  = 0.7 / nyq
    high = 3.0 / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def chrom_method(rgb_signal):
    rgb_mean = np.mean(rgb_signal, axis=0)
    rgb_norm = rgb_signal / (rgb_mean + 1e-8)
    X = 3 * rgb_norm[:, 0] - 2 * rgb_norm[:, 1]
    Y = 1.5 * rgb_norm[:, 0] + rgb_norm[:, 1] - 1.5 * rgb_norm[:, 2]
    alpha = np.std(X) / (np.std(Y) + 1e-8)
    return X - alpha * Y


def pos_method(rgb_signal):
    rgb_mean = np.mean(rgb_signal, axis=0)
    cn = rgb_signal / (rgb_mean + 1e-8)
    S1 = cn[:, 1] - cn[:, 2]
    S2 = cn[:, 1] + cn[:, 2] - 2 * cn[:, 0]
    alpha = np.std(S1) / (np.std(S2) + 1e-8)
    return S1 + alpha * S2


def calculate_snr_pro(signal, fs):
    n = len(signal)
    if n == 0:
        return -99, 0
    yf  = fft(signal)
    xf  = fftfreq(n, 1 / fs)
    psd = np.abs(yf) ** 2
    passband_mask = (xf >= 0.7) & (xf <= 3.0)
    if not any(passband_mask):
        return -99, 0
    peak_idx  = np.argmax(psd[passband_mask])
    peak_freq = xf[passband_mask][peak_idx]
    signal_mask  = (xf >= peak_freq - 0.1) & (xf <= peak_freq + 0.1)
    signal_power = np.sum(psd[signal_mask])
    total_passband_power = np.sum(psd[passband_mask])
    noise_power = total_passband_power - signal_power
    if noise_power <= 0:
        return 20, peak_freq * 60
    snr = 10 * np.log10(signal_power / noise_power)
    return round(snr, 2), round(peak_freq * 60, 2)


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def compute_blink_features(ears, fps):
    if not ears:
        return None
    ears = np.array(ears)
    duration_seconds = len(ears) / fps
    below = ears < EAR_THRESHOLD
    blink_durations = []
    count = 0
    for val in below:
        if val:
            count += 1
        else:
            if count >= MIN_BLINK_FRAMES:
                blink_durations.append(count)
            count = 0
    if count >= MIN_BLINK_FRAMES:
        blink_durations.append(count)
    blink_count = len(blink_durations)
    blink_rate  = (blink_count / duration_seconds) * 60.0 if duration_seconds > 0 else 0.0
    return {
        "mean_ear":            float(np.mean(ears)),
        "std_ear":             float(np.std(ears)),
        "min_ear":             float(np.min(ears)),
        "blink_count":         blink_count,
        "blink_rate_per_min":  round(blink_rate, 4),
        "mean_blink_duration": float(np.mean(blink_durations)) if blink_durations else 0.0,
        "std_blink_duration":  float(np.std(blink_durations))  if blink_durations else 0.0,
    }


def extract_video_features(video_path, detector, predictor, progress_bar=None):
    """
    Single-pass feature extraction from a video file.
    Returns dict with all 12 features, or None on failure.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6 or not np.isfinite(fps):
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    rgb_frames      = []
    ear_sequence    = []
    ita_values      = []
    last_valid_rgb  = None
    last_valid_ear  = 0.3
    frame_idx       = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) > 0:
            face  = get_largest_face(faces)
            shape = predictor(gray, face)

            # ROI extraction
            rois = get_refined_rois(frame, shape)
            if rois[0] is not None:
                f_bgr = roi_mean_bgr(frame, rois[0])
                l_bgr = roi_mean_bgr(frame, rois[1])
                r_bgr = roi_mean_bgr(frame, rois[2])
                if f_bgr is not None and l_bgr is not None and r_bgr is not None:
                    avg_bgr = np.mean([f_bgr, l_bgr, r_bgr], axis=0)
                    avg_rgb = avg_bgr[::-1]
                    rgb_frames.append(avg_rgb)
                    last_valid_rgb = avg_rgb

                # ITA from forehead LAB
                if f_bgr is not None:
                    frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
                    y1, y2, x1, x2 = rois[0]
                    if y2 > y1 and x2 > x1:
                        lab_roi = frame_lab[y1:y2, x1:x2]
                        if lab_roi.size > 0:
                            lab_mean = cv2.mean(lab_roi)[:3]
                            ita_values.append(calculate_ita(lab_mean[0], lab_mean[2]))

            elif last_valid_rgb is not None:
                rgb_frames.append(last_valid_rgb)

            # EAR
            left_eye  = [(shape.part(i).x, shape.part(i).y) for i in range(36, 42)]
            right_eye = [(shape.part(i).x, shape.part(i).y) for i in range(42, 48)]
            avg_ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
            ear_sequence.append(avg_ear)
            last_valid_ear = avg_ear
        else:
            if last_valid_rgb is not None:
                rgb_frames.append(last_valid_rgb)
            ear_sequence.append(last_valid_ear)

        frame_idx += 1
        if progress_bar is not None and total_frames > 0:
            progress_bar.progress(min(frame_idx / total_frames, 1.0))

    cap.release()

    if len(rgb_frames) < 30:
        return None

    # rPPG
    rgb_signal  = np.array(rgb_frames)
    bvp_chrom   = chrom_method(rgb_signal)
    clean_chrom = apply_butterworth(bvp_chrom, fs=fps)
    chrom_snr, chrom_bpm = calculate_snr_pro(clean_chrom, fps)

    bvp_pos   = pos_method(rgb_signal)
    clean_pos = apply_butterworth(bvp_pos, fs=fps)
    pos_snr, pos_bpm = calculate_snr_pro(clean_pos, fps)

    # Blink
    blink_feats = compute_blink_features(ear_sequence, fps)
    if blink_feats is None:
        return None

    # ITA
    measured_ita = float(np.median(ita_values)) if ita_values else 27.0

    return {
        "chrom_snr": chrom_snr, "pos_snr": pos_snr,
        "chrom_bpm": chrom_bpm, "pos_bpm": pos_bpm,
        **blink_feats,
        "measured_ita": measured_ita,
    }


# ── PREDICTION HELPERS ──────────────────────────────────────────────────────

def predict_all_models(models, feat_values):
    """Run all loaded models on a feature vector. Returns list of result dicts."""
    results = []
    for name, model in models.items():
        prob = model.predict_proba(feat_values)[0]
        pred = int(model.predict(feat_values)[0])
        results.append({
            "model": name,
            "prediction": "FAKE" if pred == 1 else "REAL",
            "confidence": prob[pred],
            "prob_real": prob[0],
            "prob_fake": prob[1],
        })
    return results


def compute_ensemble(results):
    """Average probabilities across all models for an ensemble prediction."""
    avg_fake = np.mean([r["prob_fake"] for r in results])
    avg_real = np.mean([r["prob_real"] for r in results])
    pred = "FAKE" if avg_fake > avg_real else "REAL"
    confidence = max(avg_fake, avg_real)
    vote_fake = sum(1 for r in results if r["prediction"] == "FAKE")
    vote_real = len(results) - vote_fake
    return {
        "prediction": pred,
        "confidence": confidence,
        "prob_fake": avg_fake,
        "prob_real": avg_real,
        "vote_fake": vote_fake,
        "vote_real": vote_real,
        "n_models": len(results),
    }


def render_ensemble_result(ensemble, results):
    """Render the ensemble prediction and individual model breakdown."""

    # ── Main verdict ──
    st.markdown("### Ensemble Verdict")
    avg_fake = ensemble["prob_fake"]
    avg_real = ensemble["prob_real"]

    if ensemble["confidence"] < 0.60:
        st.warning(
            f"### UNCERTAIN\n"
            f"**Averaged probability:** {avg_real:.0%} real / {avg_fake:.0%} fake — "
            f"too close to call reliably"
        )
    elif ensemble["prediction"] == "FAKE":
        st.error(
            f"### DEEPFAKE DETECTED\n"
            f"**Averaged probability:** {avg_fake:.0%} fake "
            f"(across {ensemble['n_models']} models)"
        )
    else:
        st.success(
            f"### REAL VIDEO\n"
            f"**Averaged probability:** {avg_real:.0%} real "
            f"(across {ensemble['n_models']} models)"
        )

    # ── Probability gauge ──
    fig_gauge, ax_g = plt.subplots(figsize=(8, 1.2))
    ax_g.barh([0], [ensemble["prob_real"]], color="#2ecc71", height=0.5, label="Real")
    ax_g.barh([0], [ensemble["prob_fake"]], left=[ensemble["prob_real"]],
              color="#e74c3c", height=0.5, label="Fake")
    ax_g.axvline(0.5, color="black", linewidth=1.5)
    ax_g.set_xlim(0, 1)
    ax_g.set_yticks([])
    ax_g.set_xlabel("Ensemble Probability")
    ax_g.legend(fontsize=9, loc="upper right", ncol=2)

    # Annotate
    ax_g.text(ensemble["prob_real"] / 2, 0, f"{ensemble['prob_real']:.0%}",
              ha="center", va="center", fontsize=12, fontweight="bold", color="white")
    ax_g.text(ensemble["prob_real"] + ensemble["prob_fake"] / 2, 0,
              f"{ensemble['prob_fake']:.0%}",
              ha="center", va="center", fontsize=12, fontweight="bold", color="white")
    for spine in ax_g.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    st.pyplot(fig_gauge)
    plt.close(fig_gauge)

    # ── Individual model breakdown ──
    with st.expander("View individual model predictions", expanded=False):
        # Group by dataset
        datasets = ["FF++", "Celeb-DF", "Combined"]
        cols = st.columns(3)
        for col, ds in zip(cols, datasets):
            ds_results = [r for r in results if r["model"].startswith(ds)]
            with col:
                st.markdown(f"**{ds}**")
                for r in ds_results:
                    clf = r["model"].split()[-1]  # "XGBoost" or "RF"
                    icon = "🔴" if r["prediction"] == "FAKE" else "🟢"
                    st.markdown(f"{icon} {clf}: {r['prediction']} ({r['confidence']:.0%})")

        # Bar chart of all models
        fig, ax = plt.subplots(figsize=(9, 3.5))
        model_names = [r["model"] for r in results]
        fake_probs  = [r["prob_fake"] for r in results]

        colors = ["#e74c3c" if r["prediction"] == "FAKE" else "#2ecc71" for r in results]
        y = np.arange(len(model_names))
        ax.barh(y, fake_probs, color=colors, edgecolor="white")
        ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(model_names, fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_xlabel("P(Fake)")
        ax.set_title("Per-Model Fake Probability")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for i, v in enumerate(fake_probs):
            ax.text(v + 0.02 if v < 0.9 else v - 0.06, i,
                    f"{v:.0%}", va="center", fontsize=9, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ── PAGE SETUP ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deepfake Detection Dashboard",
    page_icon="🔍",
    layout="wide",
)

st.title("Deepfake Detection Dashboard")
st.caption("Physiological signal analysis (rPPG + Eye-Blink), multi-model comparison")

models = load_models()
df = load_data()

tab1, tab2, tab3 = st.tabs(["Analyse Video", "Upload & Detect", "Bias Audit"])


# ── TAB 1: ANALYSE (existing dataset) ──────────────────────────────────────
with tab1:
    st.caption(
        "This tab demonstrates the feature extraction pipeline on pre-processed dataset videos. "
        "Select any video to explore its extracted physiological signals and model predictions. "
        "To test the system on an unknown video, use the Upload & Detect tab."
    )
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("Select a Video")

        dataset_filter = st.radio("Dataset", ["All", "FF++", "Celeb-DF"], horizontal=True)
        label_filter = st.radio("Filter by label", ["All", "Real only", "Fake only"])
        tone_filter = st.multiselect(
            "Filter by skin tone",
            ["Dark", "Medium", "Light"],
            default=["Dark", "Medium", "Light"],
        )

        filtered = df.copy()
        if dataset_filter != "All":
            filtered = filtered[filtered["dataset"] == dataset_filter]
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
        st.markdown(f":{gt_colour}[{gt_label}]")
        st.markdown(
            f"**Dataset:** {row['dataset']}  \n"
            f"**Skin Tone:** {row['skin_tone']}  \n"
            f"**ITA value:** {row['measured_ita']:.2f}"
        )

    with col_right:
        st.subheader("Detection Result")

        feat_values = row[FEATURES].values.reshape(1, -1).astype(float)
        results = predict_all_models(models, feat_values)
        ensemble = compute_ensemble(results)

        # Check correctness
        gt_is_fake = row[LABEL] == 1
        ensemble_correct = (ensemble["prediction"] == "FAKE") == gt_is_fake

        render_ensemble_result(ensemble, results)

        if ensemble_correct:
            st.caption("Ensemble prediction vs ground truth: **Correct**")
        else:
            st.caption("Ensemble prediction vs ground truth: **Incorrect**")

        st.divider()
        st.subheader("Extracted Features")
        feat_display = {FEATURE_LABELS[f]: round(row[f], 4) for f in FEATURES}
        feat_df = pd.DataFrame(feat_display.items(), columns=["Feature", "Value"])
        col_a, col_b = st.columns(2)
        half = len(feat_df) // 2
        col_a.dataframe(feat_df.iloc[:half].set_index("Feature"), use_container_width=True)
        col_b.dataframe(feat_df.iloc[half:].set_index("Feature"), use_container_width=True)


# ── TAB 2: UPLOAD & DETECT ─────────────────────────────────────────────────
with tab2:
    st.subheader("Upload a Video for Deepfake Detection")
    st.caption(
        "Upload any video file. The system will extract physiological signals "
        "(rPPG + eye-blink) and run all 3 models to determine if it is a deepfake."
    )

    uploaded = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv"],
        help="Upload a video with a visible face for analysis.",
    )

    if uploaded is not None:
        # Save to temp file for OpenCV
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        st.video(uploaded)

        if st.button("Analyse Video", type="primary"):
            st.divider()

            # Load dlib
            with st.spinner("Loading face detector..."):
                detector, predictor_dlib = load_dlib()

            # Extract features
            st.markdown("**Extracting physiological features...**")
            progress = st.progress(0)
            feats = extract_video_features(tmp_path, detector, predictor_dlib, progress)
            progress.empty()

            if feats is None:
                st.error(
                    "Could not extract features from this video. "
                    "Ensure the video contains a clearly visible face with at least 1 second of footage."
                )
            else:
                st.success("Feature extraction complete!")

                # Show extracted features
                with st.expander("View extracted features", expanded=False):
                    feat_display = {FEATURE_LABELS[f]: round(feats[f], 4) for f in FEATURES}
                    feat_df = pd.DataFrame(feat_display.items(), columns=["Feature", "Value"])
                    col_a, col_b = st.columns(2)
                    half = len(feat_df) // 2
                    col_a.dataframe(feat_df.iloc[:half].set_index("Feature"), use_container_width=True)
                    col_b.dataframe(feat_df.iloc[half:].set_index("Feature"), use_container_width=True)

                    skin_tone = ita_to_group(feats["measured_ita"])
                    st.markdown(
                        f"**Detected skin tone:** {skin_tone} "
                        f"(ITA: {feats['measured_ita']:.2f})"
                    )

                # Run predictions
                st.divider()
                st.subheader("Detection Results")

                feat_array = np.array([[feats[f] for f in FEATURES]])
                results = predict_all_models(models, feat_array)
                ensemble = compute_ensemble(results)

                render_ensemble_result(ensemble, results)

                st.divider()
                st.caption(
                    "Ensemble of 6 models (XGBoost + Random Forest, each trained on "
                    "FF++, Celeb-DF, and Combined datasets). "
                    "Probabilities are averaged across all models for a robust prediction."
                )

        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── TAB 3: BIAS AUDIT ──────────────────────────────────────────────────────
with tab3:
    st.subheader("Bias Audit — Accuracy by Skin Tone Group")
    st.caption(
        "Evaluated on the **300 held-out FF++ videos** never seen during training. "
        "Model: Random Forest trained on 1,699 FF++ videos. "
        "Skin tone determined objectively via ITA (Individual Typology Angle). "
        "Dark: ITA ≤ 10 | Medium: 10 < ITA ≤ 41 | Light: ITA > 41"
    )

    # Use FF++ RF model on held-out test set only (consistent with report)
    audit_model = models.get("FF++ RF")
    if audit_model is None:
        st.error("FF++ RF model not found. Run classifier.py first.")
        st.stop()

    # Filter to FF++ held-out 300 videos only
    bias_ids = pd.read_csv("data/output/bias_audit_ids.csv")
    bias_join = set(bias_ids["video_id"].apply(
        lambda x: str(x).split("/")[-1].replace(".mp4", "")
    ))
    df_ff = df[df["dataset"] == "FF++"].copy()
    df_ff["join_id"] = df_ff["video_id"].apply(
        lambda x: str(x).split("/")[-1].replace(".mp4", "")
    )
    df_holdout = df_ff[df_ff["join_id"].isin(bias_join)].copy()

    preds_holdout = audit_model.predict(df_holdout[FEATURES].values)
    df_audit = df_holdout.copy()
    df_audit["predicted"] = preds_holdout
    df_audit["correct"] = (df_audit["predicted"] == df_audit[LABEL]).astype(int)

    overall_acc = df_audit["correct"].mean()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Overall Accuracy", f"{overall_acc:.1%}")
    col2.metric("Held-Out Videos", len(df_audit))
    col3.metric("Real Videos", int((df_audit[LABEL] == 0).sum()))
    col4.metric("Fake Videos", int((df_audit[LABEL] == 1).sum()))

    st.divider()

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
        ax.axhline(overall_acc, color="steelblue", linestyle="--", linewidth=1.2,
                   label=f"Overall ({overall_acc:.1%})")
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
            st.warning("Gap 5-10% — moderate disparity detected.")
        else:
            st.error("Gap > 10% — significant bias detected.")

