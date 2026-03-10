"""
Scans ITA on temp_scan videos, selects light/medium/dark candidates not already
in the audit set, copies them to audit_set/, and appends to dataset_bias_audit.csv.

Targets ~50 per ITA group (light/medium/dark), accounting for what already exists.
"""
import cv2
import dlib
import numpy as np
import pandas as pd
import os
import shutil
from tqdm import tqdm

# --- CONFIG ---
AUDIT_CSV       = "data/output/dataset_bias_audit.csv"
ITA_CSV         = "data/output/ita_objective_audit.csv"
SHAPE_PREDICTOR = "src/shape_predictor_68_face_landmarks.dat"

TEMP_REAL = "data/temp_scan/original_sequences/youtube/c23/videos"
TEMP_FAKE = "data/temp_scan/manipulated_sequences/Deepfakes/c23/videos"

AUDIT_REAL = "data/audit_set/original_sequences/youtube/c23/videos"
AUDIT_FAKE = "data/audit_set/manipulated_sequences/Deepfakes/c23/videos"

TARGET_PER_GROUP = 50   # target total per ITA group across the full dataset

ITA_LIGHT_THRESHOLD = 41.0
ITA_DARK_THRESHOLD  = 10.0

detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR)


def calculate_ita(l_ocv, b_ocv):
    l_std = l_ocv * (100.0 / 255.0)
    b_std = b_ocv - 128.0
    return float(np.arctan2((l_std - 50.0), b_std) * (180.0 / np.pi))


def ita_group(ita):
    if ita > ITA_LIGHT_THRESHOLD:  return "light"
    elif ita > ITA_DARK_THRESHOLD: return "medium"
    else:                           return "dark"


def scan_ita(v_path, n_samples=10):
    cap = cv2.VideoCapture(v_path)
    if not cap.isOpened():
        return None
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, min(total - 1, 200), n_samples).astype(int)
    samples = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if not faces:
            continue
        face  = max(faces, key=lambda r: (r.right()-r.left())*(r.bottom()-r.top()))
        shape = predictor(gray, face)
        f_bottom = min(shape.part(19).y, shape.part(24).y) - 5
        f_top    = max(0, f_bottom - 20)
        f_left, f_right = shape.part(18).x, shape.part(25).x
        roi = frame[f_top:f_bottom, f_left:f_right]
        if roi.size == 0:
            continue
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2Lab)
        l_mean, _, b_mean = cv2.mean(lab)[:3]
        samples.append(calculate_ita(l_mean, b_mean))
    cap.release()
    return float(np.mean(samples)) if samples else None


def get_base_id(filename):
    """Real: '123.mp4' -> '123'. Fake: '123_456.mp4' -> '123'."""
    return filename.replace('.mp4', '').split('_')[0]


def run():
    audit_df = pd.read_csv(AUDIT_CSV)
    existing = set(audit_df['video_id'].apply(lambda x: x.split('/')[-1]))
    print(f"Already have {len(existing)} audited videos — will skip these.")

    # Count current ITA group sizes from ita_objective_audit.csv
    ita_df = pd.read_csv(ITA_CSV)
    ita_df['ita_grp'] = ita_df['measured_ita'].apply(ita_group)
    current_counts = ita_df['ita_grp'].value_counts().to_dict()
    print(f"\nCurrent ITA group counts (from existing ITA CSV):")
    for g in ['light', 'medium', 'dark']:
        print(f"  {g}: {current_counts.get(g, 0)}")

    # How many more needed per group
    need = {g: max(0, TARGET_PER_GROUP - current_counts.get(g, 0))
            for g in ['light', 'medium', 'dark']}
    print(f"\nNeed to add (targeting {TARGET_PER_GROUP} per group):")
    for g, n in need.items():
        print(f"  {g}: {n}")

    # --- Scan real videos in temp_scan ---
    real_files = [f for f in os.listdir(TEMP_REAL)
                  if f.endswith('.mp4') and f not in existing]
    print(f"\nScanning {len(real_files)} new real videos for ITA...")

    real_ita = {}
    for f in tqdm(real_files):
        ita = scan_ita(os.path.join(TEMP_REAL, f))
        if ita is not None:
            real_ita[f] = ita

    # Categorise candidates
    candidates = {'light': {}, 'medium': {}, 'dark': {}}
    for f, v in real_ita.items():
        candidates[ita_group(v)][f] = v

    for g in ['light', 'medium', 'dark']:
        print(f"  {g} candidates available: {len(candidates[g])}")

    # Select up to needed counts per group
    selected_real = {}
    for g in ['light', 'medium', 'dark']:
        n = need[g]
        if n == 0:
            continue
        # Sort: light = highest ITA first, dark = lowest first, medium = closest to midpoint
        if g == 'light':
            sorted_c = sorted(candidates[g].items(), key=lambda x: x[1], reverse=True)
        elif g == 'dark':
            sorted_c = sorted(candidates[g].items(), key=lambda x: x[1])
        else:
            sorted_c = sorted(candidates[g].items(), key=lambda x: abs(x[1] - 25.5))
        selected = dict(sorted_c[:n])
        selected_real.update(selected)
        print(f"  Selected {len(selected)} {g} real videos")

    # Find matching fake videos
    selected_base_ids = {get_base_id(f) for f in selected_real}
    fake_files = [f for f in os.listdir(TEMP_FAKE) if f.endswith('.mp4')]
    matching_fakes = [f for f in fake_files
                      if get_base_id(f) in selected_base_ids
                      and f not in existing]
    print(f"\nFound {len(matching_fakes)} matching fake videos")

    # --- Copy to audit_set and build new rows ---
    os.makedirs(AUDIT_REAL, exist_ok=True)
    os.makedirs(AUDIT_FAKE, exist_ok=True)

    new_rows = []

    for filename, ita_val in selected_real.items():
        src = os.path.join(TEMP_REAL, filename)
        dst = os.path.join(AUDIT_REAL, filename)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        new_rows.append({
            "video_id":            f"c23/{filename}",
            "gender_presentation": "unknown",
            "skin_tone_group":     ita_group(ita_val),
            "is_deepfake":         0
        })

    for filename in matching_fakes:
        src = os.path.join(TEMP_FAKE, filename)
        dst = os.path.join(AUDIT_FAKE, filename)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        base_id    = get_base_id(filename)
        real_match = next((f for f in selected_real if get_base_id(f) == base_id), None)
        grp        = ita_group(selected_real[real_match]) if real_match else "medium"
        new_rows.append({
            "video_id":            f"c23/{filename}",
            "gender_presentation": "unknown",
            "skin_tone_group":     grp,
            "is_deepfake":         1
        })

    if new_rows:
        new_df  = pd.DataFrame(new_rows)
        full_df = pd.concat([audit_df, new_df], ignore_index=True)
        full_df.to_csv(AUDIT_CSV, index=False)
        print(f"\nAdded {len(new_rows)} new rows to {AUDIT_CSV}")
        print(f"Total videos now: {len(full_df)}")
        print(f"\nNew skin_tone_group distribution:")
        print(full_df['skin_tone_group'].value_counts())
    else:
        print("\nNo new videos to add — targets already met.")


if __name__ == "__main__":
    run()
