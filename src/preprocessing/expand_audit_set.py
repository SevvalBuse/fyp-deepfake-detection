"""
Copies all remaining temp_scan videos (not already in audit_set) to audit_set/
and appends their metadata to dataset_bias_audit.csv.

No ITA filtering — takes all available videos to maximise dataset size.
The original 260 ITA-balanced videos are preserved for bias audit.
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
SHAPE_PREDICTOR = "src/shape_predictor_68_face_landmarks.dat"

TEMP_REAL  = "data/temp_scan/original_sequences/youtube/c23/videos"
TEMP_FAKE  = "data/temp_scan/manipulated_sequences/Deepfakes/c23/videos"

AUDIT_REAL = "data/audit_set/original_sequences/youtube/c23/videos"
AUDIT_FAKE = "data/audit_set/manipulated_sequences/Deepfakes/c23/videos"

detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR)


def calculate_ita(l_ocv, b_ocv):
    l_std = l_ocv * (100.0 / 255.0)
    b_std = b_ocv - 128.0
    return float(np.arctan2((l_std - 50.0), b_std) * (180.0 / np.pi))


def ita_group(ita):
    if ita > 41.0:  return "light"
    elif ita > 10.0: return "medium"
    else:            return "dark"


def scan_ita(v_path, n_samples=5):
    """Quick ITA scan — 5 frames only, just for metadata labelling."""
    cap = cv2.VideoCapture(v_path)
    if not cap.isOpened():
        return None
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, min(total - 1, 100), n_samples).astype(int)
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

    # Find all new real videos not already in audit_set
    real_files = [f for f in os.listdir(TEMP_REAL)
                  if f.endswith('.mp4') and f not in existing]
    print(f"Found {len(real_files)} new real videos to add.")

    # Find matching fake videos for those real videos
    real_base_ids = {get_base_id(f) for f in real_files}
    fake_files = [f for f in os.listdir(TEMP_FAKE)
                  if f.endswith('.mp4')
                  and f not in existing
                  and get_base_id(f) in real_base_ids]
    print(f"Found {len(fake_files)} matching fake videos.")

    os.makedirs(AUDIT_REAL, exist_ok=True)
    os.makedirs(AUDIT_FAKE, exist_ok=True)

    new_rows = []

    # Copy real videos
    print("\nCopying real videos and scanning ITA for metadata...")
    for filename in tqdm(real_files):
        src = os.path.join(TEMP_REAL, filename)
        dst = os.path.join(AUDIT_REAL, filename)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

        ita_val = scan_ita(src)
        grp = ita_group(ita_val) if ita_val is not None else "unknown"

        new_rows.append({
            "video_id":            f"c23/{filename}",
            "gender_presentation": "unknown",
            "skin_tone_group":     grp,
            "is_deepfake":         0
        })

    # Copy fake videos
    print("\nCopying fake videos...")
    for filename in tqdm(fake_files):
        src = os.path.join(TEMP_FAKE, filename)
        dst = os.path.join(AUDIT_FAKE, filename)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

        base_id    = get_base_id(filename)
        real_match = next((f for f in real_files if get_base_id(f) == base_id), None)
        grp        = "unknown"
        if real_match:
            ita_val = scan_ita(os.path.join(TEMP_REAL, real_match))
            if ita_val is not None:
                grp = ita_group(ita_val)

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
    else:
        print("\nNo new videos to add.")


if __name__ == "__main__":
    run()
