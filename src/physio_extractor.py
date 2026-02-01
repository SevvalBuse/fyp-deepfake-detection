import cv2
import dlib
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# --- CONFIGURATION ---
AUDIT_CSV = "dataset_bias_audit.csv"
SHAPE_PREDICTOR = "src/shape_predictor_68_face_landmarks.dat"
OUTPUT_DIR = "data/signals/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR)


def calculate_ita(l_val, b_val):
    """Individual Typology Angle (ITA) from Lab values."""
    return np.arctan2((l_val - 50), b_val) * (180 / np.pi)


def get_refined_rois(frame, shape):
    """
    Refined ROIs:
    - Forehead: above eyebrows, proportional to face height
    - Cheeks: malar region, slightly lowered to avoid eye shadow/blink artifacts
    Returns coords as (y1, y2, x1, x2) for each ROI, clamped to frame bounds.
    """
    h, w = frame.shape[:2]

    def clamp(y1, y2, x1, x2):
        y1 = max(0, int(y1))
        y2 = min(h, int(y2))
        x1 = max(0, int(x1))
        x2 = min(w, int(x2))
        return (y1, y2, x1, x2)

    face_height = shape.part(8).y - shape.part(27).y
    if face_height <= 0:
        # Fallback: return empty ROIs (will be handled upstream)
        return (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)

    # Forehead: place box above eyebrows
    f_height = int(face_height * 0.10)         # forehead box height
    f_bottom = min(shape.part(19).y, shape.part(24).y) - int(face_height * 0.02)
    f_top = f_bottom - f_height
    f_left = shape.part(18).x
    f_right = shape.part(25).x

    # Cheeks: malar bone area (small square), lowered a bit
    side = int(face_height * 0.08)             # half-size of cheek square
    cheek_down = int(face_height * 0.02)       # ~2% face height downward shift

    l_x = (shape.part(2).x + shape.part(31).x) // 2
    l_y = (shape.part(40).y + shape.part(31).y) // 2 + cheek_down

    r_x = (shape.part(14).x + shape.part(35).x) // 2
    r_y = (shape.part(47).y + shape.part(35).y) // 2 + cheek_down

    forehead = clamp(f_top, f_bottom, f_left, f_right)
    left_cheek = clamp(l_y - side, l_y + side, l_x - side, l_x + side)
    right_cheek = clamp(r_y - side, r_y + side, r_x - side, r_x + side)

    return forehead, left_cheek, right_cheek


def extract_mean_color(frame_bgr, frame_lab, coords):
    """
    Returns:
      - bgr_mean: (b, g, r)
      - lab_mean: (L, a, b)
    If ROI is empty, returns (None, None).
    """
    y1, y2, x1, x2 = coords
    roi_bgr = frame_bgr[y1:y2, x1:x2]
    if roi_bgr.size == 0:
        return None, None

    bgr_mean = cv2.mean(roi_bgr)[:3]

    roi_lab = frame_lab[y1:y2, x1:x2]
    if roi_lab.size == 0:
        return None, None
    lab_mean = cv2.mean(roi_lab)[:3]

    return bgr_mean, lab_mean


def run_extraction():
    df = pd.read_csv(AUDIT_CSV)
    results_summary = []
    print(f"Starting extraction for {len(df)} videos...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        v_id = row["video_id"]
        filename = v_id.split("/")[-1]

        # Try Original first, then Deepfake
        path_options = [
            os.path.join("data/original_sequences/youtube/c23/videos", filename),
            os.path.join("data/manipulated_sequences/Deepfakes/c23/videos", filename),
        ]
        v_path = next((p for p in path_options if os.path.exists(p)), None)
        if not v_path:
            continue

        cap = cv2.VideoCapture(v_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_signals = []
        last_valid = None

        for _ in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)

            if len(faces) > 0:
                # Use the first detected face
                shape = predictor(gray, faces[0])

                f_roi, l_roi, r_roi = get_refined_rois(frame, shape)

                f_bgr, f_lab = extract_mean_color(frame, frame_lab, f_roi)
                l_bgr, _ = extract_mean_color(frame, frame_lab, l_roi)
                r_bgr, _ = extract_mean_color(frame, frame_lab, r_roi)

                # Important: check None explicitly (tuples are always truthy)
                if f_bgr is not None and f_lab is not None and l_bgr is not None and r_bgr is not None:
                    current = {
                        "rgb": [f_bgr, l_bgr, r_bgr],  # note: OpenCV order is BGR
                        "ita": float(calculate_ita(f_lab[0], f_lab[2])),
                    }
                    video_signals.append(current)
                    last_valid = current
                elif last_valid is not None:
                    video_signals.append(last_valid)  # forward fill
                else:
                    video_signals.append({"rgb": [(0, 0, 0)] * 3, "ita": 0})
            elif last_valid is not None:
                video_signals.append(last_valid)  # forward fill
            else:
                video_signals.append({"rgb": [(0, 0, 0)] * 3, "ita": 0})

        cap.release()

        # Ensure parity with reported frame count
        while len(video_signals) < total_frames:
            video_signals.append(video_signals[-1] if video_signals else {"rgb": [(0, 0, 0)] * 3, "ita": 0})

        np.save(os.path.join(OUTPUT_DIR, f"{filename.replace('.mp4', '')}_raw.npy"), video_signals)

        itas = [d["ita"] for d in video_signals if d.get("ita", 0) != 0]
        results_summary.append({"video_id": v_id, "measured_ita": float(np.mean(itas)) if itas else 0})

    pd.DataFrame(results_summary).to_csv("ita_objective_audit.csv", index=False)
    print("\nExtraction complete. Results saved in data/signals/raw/")


if __name__ == "__main__":
    run_extraction()
