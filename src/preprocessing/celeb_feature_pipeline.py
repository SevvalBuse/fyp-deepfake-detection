"""
Celeb-DF v2 Feature Extraction Pipeline
========================================
Extracts all 12 features (rPPG + EAR + ITA) from Celeb-DF v2 videos.

Uses the EXACT same processing logic as the FF++ pipeline:
  - ROI extraction     → physio_extractor.py
  - CHROM / POS + BPF  → dual_algo_processor.py
  - SNR / BPM (FFT)    → signal_analyser.py
  - EAR / blink feats  → ear_extractor.py
  - ITA                → celebdf_ita_inventory.csv (pre-computed)

Combined into a single pass per video for efficiency (face detection
is the bottleneck — no need to open each video twice).

Run from project root:
    python src/preprocessing/celeb_feature_pipeline.py
"""

import cv2
import dlib
import numpy as np
import pandas as pd
import os
import random
import time
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
from scipy.spatial import distance as dist

# ── CONFIG ────────────────────────────────────────────────────────────────────
CELEB_REAL_DIRS = [
    "data/celeb_df_v2/Celeb-real",
    "data/celeb_df_v2/YouTube-real",
]
CELEB_FAKE_DIR   = "data/celeb_df_v2/Celeb-synthesis"
ITA_CSV          = "data/output/celebdf_ita_inventory.csv"
SHAPE_PREDICTOR  = "src/shape_predictor_68_face_landmarks.dat"
OUTPUT_CSV       = "data/output/celeb_unified_features.csv"
SELECTION_CSV    = "data/output/celeb_selected_videos.csv"   # saved for reproducibility

N_REAL = 889        # all available real videos
N_FAKE = 889        # balanced with real count
SEED   = 42

# EAR blink detection (same as ear_extractor.py)
EAR_THRESHOLD    = 0.21
MIN_BLINK_FRAMES = 2

# ── LOAD DLIB ─────────────────────────────────────────────────────────────────
print("Loading dlib face detector + landmark predictor...")
detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR)
print("Done.\n")


# ══════════════════════════════════════════════════════════════════════════════
# FUNCTIONS — identical logic to the FF++ pipeline scripts
# ══════════════════════════════════════════════════════════════════════════════

# ── From physio_extractor.py ──────────────────────────────────────────────────

def get_largest_face(faces):
    return max(faces, key=lambda r: (r.right() - r.left()) * (r.bottom() - r.top()))


def get_refined_rois(frame, shape):
    """Same ROI extraction as physio_extractor.py — forehead + left/right cheeks."""
    h, w = frame.shape[:2]

    def clamp(y1, y2, x1, x2):
        return max(0, int(y1)), min(h, int(y2)), max(0, int(x1)), min(w, int(x2))

    face_height = shape.part(8).y - shape.part(27).y
    if face_height <= 0:
        return None, None, None

    # Forehead ROI
    f_height = max(4, int(face_height * 0.10))
    f_bottom = min(shape.part(19).y, shape.part(24).y) - int(face_height * 0.02)
    f_top    = f_bottom - f_height
    forehead = clamp(f_top, f_bottom, shape.part(18).x, shape.part(25).x)

    # Cheek ROIs
    side      = max(4, int(face_height * 0.08))
    cheek_down = int(face_height * 0.02)

    l_x = (shape.part(2).x  + shape.part(31).x) // 2
    l_y = (shape.part(40).y + shape.part(31).y) // 2 + cheek_down
    r_x = (shape.part(14).x + shape.part(35).x) // 2
    r_y = (shape.part(47).y + shape.part(35).y) // 2 + cheek_down

    left_cheek  = clamp(l_y - side, l_y + side, l_x - side, l_x + side)
    right_cheek = clamp(r_y - side, r_y + side, r_x - side, r_x + side)

    return forehead, left_cheek, right_cheek


def roi_mean_bgr(frame, coords):
    """Extract mean BGR from a ROI. Returns None if invalid."""
    y1, y2, x1, x2 = coords
    if y2 <= y1 or x2 <= x1:
        return None
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    m = cv2.mean(roi)[:3]
    return m  # BGR order


# ── From dual_algo_processor.py ───────────────────────────────────────────────

def apply_butterworth(signal, fs=30.0, order=5):
    """Butterworth bandpass 0.7–3.0 Hz (same as dual_algo_processor.py)."""
    nyq  = 0.5 * fs
    low  = 0.7 / nyq
    high = 3.0 / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def chrom_method(rgb_signal):
    """CHROM rPPG extraction (same as dual_algo_processor.py)."""
    rgb_mean = np.mean(rgb_signal, axis=0)
    rgb_norm = rgb_signal / (rgb_mean + 1e-8)
    X = 3 * rgb_norm[:, 0] - 2 * rgb_norm[:, 1]
    Y = 1.5 * rgb_norm[:, 0] + rgb_norm[:, 1] - 1.5 * rgb_norm[:, 2]
    alpha = np.std(X) / (np.std(Y) + 1e-8)
    return X - alpha * Y


def pos_method(rgb_signal):
    """POS rPPG extraction (same as dual_algo_processor.py)."""
    rgb_mean = np.mean(rgb_signal, axis=0)
    cn = rgb_signal / (rgb_mean + 1e-8)
    S1 = cn[:, 1] - cn[:, 2]
    S2 = cn[:, 1] + cn[:, 2] - 2 * cn[:, 0]
    alpha = np.std(S1) / (np.std(S2) + 1e-8)
    return S1 + alpha * S2


# ── From signal_analyser.py ──────────────────────────────────────────────────

def calculate_snr_pro(signal, fs):
    """
    SNR calculation — identical to signal_analyser.py's calculate_snr_pro.
    Uses narrow band (+/- 0.1 Hz) around peak in the 0.7–3.0 Hz passband.
    """
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

    # Narrow signal band (+/- 0.1 Hz)
    signal_mask  = (xf >= peak_freq - 0.1) & (xf <= peak_freq + 0.1)
    signal_power = np.sum(psd[signal_mask])

    total_passband_power = np.sum(psd[passband_mask])
    noise_power = total_passband_power - signal_power

    if noise_power <= 0:
        return 20, peak_freq * 60

    snr = 10 * np.log10(signal_power / noise_power)
    return round(snr, 2), round(peak_freq * 60, 2)


# ── From ear_extractor.py ────────────────────────────────────────────────────

def eye_aspect_ratio(eye):
    """EAR formula — identical to ear_extractor.py."""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def compute_blink_features(ears, fps):
    """Blink feature extraction — identical to ear_extractor.py."""
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


# ══════════════════════════════════════════════════════════════════════════════
# COMBINED PER-VIDEO EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def safe_fps(cap, default=30.0):
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6 or not np.isfinite(fps):
        return float(default)
    return float(fps)


def extract_video_features(video_path):
    """
    Single-pass extraction: opens video once, extracts both rPPG and EAR data
    per frame, then computes all derived features.

    Returns a dict with all 11 features (ITA added separately from CSV), or None.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = safe_fps(cap, default=30.0)

    # Collect per-frame data
    rgb_frames = []       # for rPPG (list of 3-element RGB arrays)
    ear_sequence = []     # for blink features
    last_valid_rgb = None
    last_valid_ear = 0.3  # neutral open-eye EAR as fallback

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) > 0:
            face  = get_largest_face(faces)
            shape = predictor(gray, face)

            # ── rPPG: ROI colour extraction (physio_extractor logic) ──
            rois = get_refined_rois(frame, shape)
            if rois[0] is not None:
                f_bgr = roi_mean_bgr(frame, rois[0])
                l_bgr = roi_mean_bgr(frame, rois[1])
                r_bgr = roi_mean_bgr(frame, rois[2])

                if f_bgr is not None and l_bgr is not None and r_bgr is not None:
                    # Average 3 ROIs, convert BGR→RGB (same as dual_algo_processor)
                    avg_bgr = np.mean([f_bgr, l_bgr, r_bgr], axis=0)
                    avg_rgb = avg_bgr[::-1]  # BGR → RGB
                    rgb_frames.append(avg_rgb)
                    last_valid_rgb = avg_rgb
                elif last_valid_rgb is not None:
                    rgb_frames.append(last_valid_rgb)
            elif last_valid_rgb is not None:
                rgb_frames.append(last_valid_rgb)

            # ── EAR: eye landmark extraction (ear_extractor logic) ──
            left_eye  = [(shape.part(i).x, shape.part(i).y) for i in range(36, 42)]
            right_eye = [(shape.part(i).x, shape.part(i).y) for i in range(42, 48)]
            avg_ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
            ear_sequence.append(avg_ear)
            last_valid_ear = avg_ear
        else:
            # No face detected — repeat last valid (same as FF++ pipeline)
            if last_valid_rgb is not None:
                rgb_frames.append(last_valid_rgb)
            ear_sequence.append(last_valid_ear)

    cap.release()

    # Need minimum frames for rPPG processing
    if len(rgb_frames) < 30:
        return None

    # ── rPPG: CHROM + POS + Butterworth + SNR/BPM ────────────────────────
    rgb_signal = np.array(rgb_frames)

    bvp_chrom   = chrom_method(rgb_signal)
    clean_chrom = apply_butterworth(bvp_chrom, fs=fps)
    chrom_snr, chrom_bpm = calculate_snr_pro(clean_chrom, fps)

    bvp_pos   = pos_method(rgb_signal)
    clean_pos = apply_butterworth(bvp_pos, fs=fps)
    pos_snr, pos_bpm = calculate_snr_pro(clean_pos, fps)

    # ── EAR: blink features ──────────────────────────────────────────────
    blink_feats = compute_blink_features(ear_sequence, fps)
    if blink_feats is None:
        return None

    return {
        "chrom_snr":           chrom_snr,
        "pos_snr":             pos_snr,
        "chrom_bpm":           chrom_bpm,
        "pos_bpm":             pos_bpm,
        "mean_ear":            blink_feats["mean_ear"],
        "std_ear":             blink_feats["std_ear"],
        "min_ear":             blink_feats["min_ear"],
        "blink_count":         blink_feats["blink_count"],
        "blink_rate_per_min":  blink_feats["blink_rate_per_min"],
        "mean_blink_duration": blink_feats["mean_blink_duration"],
        "std_blink_duration":  blink_feats["std_blink_duration"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# VIDEO SELECTION + MAIN
# ══════════════════════════════════════════════════════════════════════════════

def select_videos():
    """Select 889 real + 889 fake Celeb-DF videos. Saves selection for reproducibility."""

    # Collect all real videos
    real_videos = []
    for d in CELEB_REAL_DIRS:
        folder_label = "celeb_real" if "Celeb-real" in d else "youtube_real"
        for f in sorted(os.listdir(d)):
            if f.endswith(".mp4"):
                real_videos.append({
                    "video_id": f,
                    "path": os.path.join(d, f),
                    "is_deepfake": 0,
                    "folder": folder_label,
                })

    # Collect all fake videos
    fake_videos = []
    for f in sorted(os.listdir(CELEB_FAKE_DIR)):
        if f.endswith(".mp4"):
            fake_videos.append({
                "video_id": f,
                "path": os.path.join(CELEB_FAKE_DIR, f),
                "is_deepfake": 1,
                "folder": "celeb_synthesis",
            })

    print(f"Available: {len(real_videos)} real, {len(fake_videos)} fake")

    # Select
    selected_real = real_videos[:N_REAL]  # take all available
    random.seed(SEED)
    selected_fake = random.sample(fake_videos, min(N_FAKE, len(fake_videos)))

    selected = selected_real + selected_fake
    random.seed(SEED)
    random.shuffle(selected)

    # Save selection
    sel_df = pd.DataFrame(selected)
    sel_df.to_csv(SELECTION_CSV, index=False)
    print(f"Selected {len(selected_real)} real + {len(selected_fake)} fake = {len(selected)} total")
    print(f"Selection saved to {SELECTION_CSV}\n")

    return selected


def run():
    start_time = time.time()

    # Load ITA values (pre-computed)
    ita_df  = pd.read_csv(ITA_CSV)
    ita_map = dict(zip(ita_df["video_id"], ita_df["measured_ita"]))
    print(f"Loaded {len(ita_map)} ITA values from {ITA_CSV}")

    # Select videos
    videos = select_videos()

    # Resume support: load existing results and skip already-processed videos
    if os.path.exists(OUTPUT_CSV):
        existing_df = pd.read_csv(OUTPUT_CSV)
        already_done = set(existing_df["video_id"].tolist())
        results = existing_df.to_dict("records")
        print(f"Resuming — {len(already_done)} videos already processed, skipping.\n")
    else:
        already_done = set()
        results = []

    total    = len(videos)
    skipped  = 0
    failed   = 0
    new_done = 0

    print(f"{'='*70}")
    print(f"  Celeb-DF v2 Feature Extraction — {total} videos")
    print(f"  Processing logic identical to FF++ pipeline")
    print(f"{'='*70}\n")

    for i, video in enumerate(videos):
        vid   = video["video_id"]
        path  = video["path"]
        label = video["is_deepfake"]

        # Skip if already processed (resume)
        if vid in already_done:
            skipped += 1
            continue

        # Progress display
        elapsed = time.time() - start_time
        done_so_far = new_done + len(already_done) - skipped  # actually newly processed
        if done_so_far > 0:
            avg_per_video = elapsed / done_so_far
            remaining = (total - i) * avg_per_video
            eta_min = remaining / 60
            eta_str = f"ETA: {eta_min:.0f}min"
        else:
            eta_str = "ETA: calculating..."

        label_str = "fake" if label == 1 else "real"
        print(f"[{i+1}/{total}] {vid} ({label_str}) — {eta_str}", end="", flush=True)

        if not os.path.exists(path):
            print(" — SKIP (file not found)")
            failed += 1
            continue

        # Extract features (single pass)
        feats = extract_video_features(path)

        if feats is None:
            print(" — SKIP (extraction failed)")
            failed += 1
            continue

        # Add ITA from pre-computed inventory
        ita_val = ita_map.get(vid, np.nan)
        feats["measured_ita"] = ita_val

        # Add metadata
        feats["video_id"]     = vid
        feats["is_deepfake"]  = label

        results.append(feats)
        new_done += 1

        print(f" — OK (SNR: {feats['chrom_snr']:.1f}/{feats['pos_snr']:.1f}, "
              f"blinks: {feats['blink_count']})")

        # Save checkpoint every 25 videos
        if new_done % 25 == 0:
            _save_results(results)
            elapsed_min = (time.time() - start_time) / 60
            print(f"\n  ✓ Checkpoint: {len(results)} videos saved "
                  f"({elapsed_min:.1f} min elapsed)\n")

    # Final save
    _save_results(results)

    elapsed_min = (time.time() - start_time) / 60
    print(f"\n{'='*70}")
    print(f"  EXTRACTION COMPLETE")
    print(f"  Total processed: {len(results)}")
    print(f"  New this run:    {new_done}")
    print(f"  Skipped (done):  {skipped}")
    print(f"  Failed:          {failed}")
    print(f"  Time:            {elapsed_min:.1f} minutes")
    print(f"  Output:          {OUTPUT_CSV}")
    print(f"{'='*70}")


def _save_results(results):
    """Save current results to CSV with columns in the same order as unified_features.csv."""
    df = pd.DataFrame(results)
    cols = [
        "video_id", "is_deepfake", "measured_ita",
        "chrom_snr", "pos_snr", "chrom_bpm", "pos_bpm",
        "mean_ear", "std_ear", "min_ear",
        "blink_count", "blink_rate_per_min",
        "mean_blink_duration", "std_blink_duration",
    ]
    df = df[cols]
    df.to_csv(OUTPUT_CSV, index=False)


if __name__ == "__main__":
    run()
