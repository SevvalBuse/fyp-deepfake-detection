import cv2
import dlib
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# --- CONFIG ---
AUDIT_CSV = "data/output/dataset_bias_audit.csv"  # must contain columns: video_id, is_deepfake (0/1)
SHAPE_PREDICTOR = "src/shape_predictor_68_face_landmarks.dat"
OUTPUT_DIR = "data/signals/audit_ff/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR)

META_OUT = "data/output/raw_metadata.csv"


def calculate_ita(l_val, b_val):
    return np.arctan2((l_val - 50), b_val) * (180 / np.pi)


def get_largest_face(faces):
    return max(faces, key=lambda r: (r.right() - r.left()) * (r.bottom() - r.top()))


def get_refined_rois(frame, shape):
    h, w = frame.shape[:2]

    def clamp(y1, y2, x1, x2):
        y1 = max(0, int(y1))
        y2 = min(h, int(y2))
        x1 = max(0, int(x1))
        x2 = min(w, int(x2))
        return (y1, y2, x1, x2)

    face_height = shape.part(8).y - shape.part(27).y
    if face_height <= 0:
        return (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)

    # Forehead ROI (protected against tiny faces)
    f_height = max(4, int(face_height * 0.10))
    f_bottom = min(shape.part(19).y, shape.part(24).y) - int(face_height * 0.02)
    f_top = f_bottom - f_height
    f_left = shape.part(18).x
    f_right = shape.part(25).x

    # Cheek ROIs
    side = max(4, int(face_height * 0.08))
    cheek_down = int(face_height * 0.02)

    l_x = (shape.part(2).x + shape.part(31).x) // 2
    l_y = (shape.part(40).y + shape.part(31).y) // 2 + cheek_down

    r_x = (shape.part(14).x + shape.part(35).x) // 2
    r_y = (shape.part(47).y + shape.part(35).y) // 2 + cheek_down

    forehead = clamp(f_top, f_bottom, f_left, f_right)
    left_cheek = clamp(l_y - side, l_y + side, l_x - side, l_x + side)
    right_cheek = clamp(r_y - side, r_y + side, r_x - side, r_x + side)

    return forehead, left_cheek, right_cheek


def extract_mean_color(frame_bgr, frame_lab, coords):
    y1, y2, x1, x2 = coords

    # prevent invalid/empty slices even if clamped coords cross
    if y2 <= y1 or x2 <= x1:
        return None, None

    roi_bgr = frame_bgr[y1:y2, x1:x2]
    if roi_bgr.size == 0:
        return None, None
    bgr_mean = cv2.mean(roi_bgr)[:3]

    roi_lab = frame_lab[y1:y2, x1:x2]
    if roi_lab.size == 0:
        return None, None
    lab_mean = cv2.mean(roi_lab)[:3]

    return bgr_mean, lab_mean


def safe_fps(cap, default=30.0):
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6 or not np.isfinite(fps):
        return float(default)
    return float(fps)


def run_extraction():
    df = pd.read_csv(AUDIT_CSV)

    meta_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        v_id = str(row["video_id"])
        label = int(row["is_deepfake"])  # 0=original, 1=deepfake
        filename = v_id.split("/")[-1]

        if label == 0:
            v_path = os.path.join("data/audit_set/original_sequences/youtube/c23/videos", filename)
        else:
            v_path = os.path.join("data/audit_set/manipulated_sequences/Deepfakes/c23/videos", filename)

        if not os.path.exists(v_path):
            continue

        cap = cv2.VideoCapture(v_path)
        if not cap.isOpened():  # guard failed open
            continue

        fps = safe_fps(cap, default=30.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        video_signals = []
        last_valid = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            if len(faces) > 0:
                target_face = get_largest_face(faces)
                shape = predictor(gray, target_face)

                f_roi, l_roi, r_roi = get_refined_rois(frame, shape)

                # Only compute Lab once per frame (still okay to do every frame)
                frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)

                f_bgr, f_lab = extract_mean_color(frame, frame_lab, f_roi)
                l_bgr, _ = extract_mean_color(frame, frame_lab, l_roi)
                r_bgr, _ = extract_mean_color(frame, frame_lab, r_roi)

                if f_bgr is not None and f_lab is not None and l_bgr is not None and r_bgr is not None:
                    ita_val = float(calculate_ita(f_lab[0], f_lab[2]))
                    current = {"rgb": [f_bgr, l_bgr, r_bgr], "ita": ita_val}  # OpenCV order is BGR
                    video_signals.append(current)
                    last_valid = current
                else:
                    # avoid zero padding; keep placeholder None until we can fill
                    video_signals.append(last_valid if last_valid is not None else None)
            else:
                # avoid zero padding
                video_signals.append(last_valid if last_valid is not None else None)

        cap.release()

        # parity with reported frame count (pad with last observed entry / None)
        while total_frames > 0 and len(video_signals) < total_frames:
            video_signals.append(video_signals[-1] if video_signals else None)

        # replace Nones with first valid (no artificial zeros)
        first_valid = next((v for v in video_signals if v is not None), None)
        if first_valid is None:
            continue  # no usable frames for this video

        video_signals = [v if v is not None else first_valid for v in video_signals]

        out_name = f"{filename.replace('.mp4', '')}_raw.npy"
        np.save(os.path.join(OUTPUT_DIR, out_name), video_signals)

        meta_rows.append({
            "video_id": v_id,
            "filename": filename,
            "is_deepfake": label,
            "fps": fps,
            "frames_saved": len(video_signals),
            "frames_reported": total_frames
        })

    if meta_rows:
        pd.DataFrame(meta_rows).to_csv(META_OUT, index=False)

    print("\nExtraction complete. Raw signals saved in:", OUTPUT_DIR)
    print("Metadata saved in:", META_OUT)


if __name__ == "__main__":
    run_extraction()
