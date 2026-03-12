"""
Recomputes ITA for the FF++ audit set videos using the correct LAB normalisation.
Overwrites data/output/ita_objective_audit.csv.

The original physio_extractor.py used raw OpenCV LAB values (L in [0,255], b in [0,255])
which produces incorrect ITA values. This script applies the correct normalisation.
"""
import cv2
import dlib
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

AUDIT_CSV       = "data/output/dataset_bias_audit.csv"
SHAPE_PREDICTOR = "src/shape_predictor_68_face_landmarks.dat"
OUTPUT_CSV      = "data/output/ita_objective_audit.csv"

detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR)


def calculate_ita(l_ocv, b_ocv):
    """
    Correct ITA from raw OpenCV LAB values (8-bit image).
    OpenCV stores L in [0,255] and b in [0,255] (shifted by +128).
    Standard ITA formula requires L in [0,100] and b in [-128,127].
    """
    l_std = l_ocv * (100.0 / 255.0)
    b_std = b_ocv - 128.0
    return float(np.arctan2((l_std - 50.0), b_std) * (180.0 / np.pi))


def scan_ita(v_path, n_samples=10):
    cap = cv2.VideoCapture(v_path)
    if not cap.isOpened():
        return None

    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, min(total - 1, 200), n_samples).astype(int)

    ita_samples = []
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
        f_left   = shape.part(18).x
        f_right  = shape.part(25).x

        roi = frame[f_top:f_bottom, f_left:f_right]
        if roi.size == 0:
            continue

        lab    = cv2.cvtColor(roi, cv2.COLOR_BGR2Lab)
        l_mean, _, b_mean = cv2.mean(lab)[:3]
        ita_samples.append(calculate_ita(l_mean, b_mean))

    cap.release()
    return float(np.mean(ita_samples)) if ita_samples else None


def run():
    df      = pd.read_csv(AUDIT_CSV)
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing ITA"):
        v_id     = str(row["video_id"])
        label    = int(row["is_deepfake"])
        filename = v_id.split("/")[-1]

        v_path = os.path.join(
            "data/audit_set/original_sequences/youtube/c23/videos" if label == 0
            else "data/audit_set/manipulated_sequences/Deepfakes/c23/videos",
            filename
        )

        if not os.path.exists(v_path):
            print(f"  [SKIP] {filename}")
            continue

        ita = scan_ita(v_path)
        if ita is not None:
            results.append({"video_id": v_id, "measured_ita": round(ita, 4)})

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone. {len(out)} ITA values saved to {OUTPUT_CSV}")
    print(f"ITA range: {out['measured_ita'].min():.2f} to {out['measured_ita'].max():.2f}")
    print(out["measured_ita"].describe().round(2))


if __name__ == "__main__":
    run()
