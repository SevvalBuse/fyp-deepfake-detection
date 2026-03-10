import cv2
import dlib
import pandas as pd
import numpy as np
import os
from scipy.spatial import distance as dist
from tqdm import tqdm

# --- CONFIG ---
AUDIT_CSV = "data/output/dataset_bias_audit.csv"
SHAPE_PREDICTOR = "src/shape_predictor_68_face_landmarks.dat"
OUTPUT_CSV = "data/output/ear_features.csv"
EAR_THRESHOLD = 0.21   # below this = eye closed (blink)
MIN_BLINK_FRAMES = 2   # minimum consecutive frames below threshold to count as blink

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR)


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def get_largest_face(faces):
    return max(faces, key=lambda r: (r.right() - r.left()) * (r.bottom() - r.top()))


def extract_ear_sequence(video_path):
    """Extract per-frame EAR values from a video. Returns list of EAR values."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    ears = []
    last_valid = 0.3  # neutral open-eye EAR as fallback

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) > 0:
            face = get_largest_face(faces)
            shape = predictor(gray, face)
            left  = [(shape.part(i).x, shape.part(i).y) for i in range(36, 42)]
            right = [(shape.part(i).x, shape.part(i).y) for i in range(42, 48)]
            avg_ear = (eye_aspect_ratio(left) + eye_aspect_ratio(right)) / 2.0
            ears.append(avg_ear)
            last_valid = avg_ear
        else:
            ears.append(last_valid)

    cap.release()
    return ears, float(fps)


def compute_blink_features(ears, fps, threshold=EAR_THRESHOLD, min_frames=MIN_BLINK_FRAMES):
    """
    From a sequence of EAR values, compute:
    - mean_ear: average eye openness
    - std_ear: variability of eye openness
    - min_ear: deepest eye closure observed
    - blink_count: number of detected blinks
    - blink_rate: blinks per minute
    - mean_blink_duration: average frames per blink
    - std_blink_duration: variability of blink duration
    """
    if not ears:
        return None

    ears = np.array(ears)
    duration_seconds = len(ears) / fps

    # Detect blinks: consecutive runs below threshold
    below = ears < threshold
    blink_durations = []
    count = 0
    for val in below:
        if val:
            count += 1
        else:
            if count >= min_frames:
                blink_durations.append(count)
            count = 0
    if count >= min_frames:
        blink_durations.append(count)

    blink_count = len(blink_durations)
    blink_rate = (blink_count / duration_seconds) * 60.0 if duration_seconds > 0 else 0.0

    return {
        "mean_ear":            float(np.mean(ears)),
        "std_ear":             float(np.std(ears)),
        "min_ear":             float(np.min(ears)),
        "blink_count":         blink_count,
        "blink_rate_per_min":  round(blink_rate, 4),
        "mean_blink_duration": float(np.mean(blink_durations)) if blink_durations else 0.0,
        "std_blink_duration":  float(np.std(blink_durations))  if blink_durations else 0.0,
    }


def run_ear_extraction():
    df = pd.read_csv(AUDIT_CSV)

    # Resume: load existing results and skip already-processed videos
    if os.path.exists(OUTPUT_CSV):
        existing_df = pd.read_csv(OUTPUT_CSV)
        already_done = set(existing_df["video_id"].tolist())
        results = existing_df.to_dict("records")
        print(f"Resuming — {len(already_done)} videos already done, skipping.")
    else:
        already_done = set()
        results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting EAR"):
        v_id = str(row["video_id"])
        if v_id in already_done:
            continue
        label = int(row["is_deepfake"])
        filename = v_id.split("/")[-1]

        if label == 0:
            v_path = os.path.join("data/audit_set/original_sequences/youtube/c23/videos", filename)
        else:
            v_path = os.path.join("data/audit_set/manipulated_sequences/Deepfakes/c23/videos", filename)

        if not os.path.exists(v_path):
            print(f"  [SKIP] Not found: {v_path}")
            continue

        ears, fps = extract_ear_sequence(v_path)
        if ears is None or len(ears) == 0:
            print(f"  [SKIP] No frames extracted: {filename}")
            continue

        features = compute_blink_features(ears, fps)
        if features is None:
            continue

        features["video_id"] = v_id
        results.append(features)

    if not results:
        print("No EAR features extracted.")
        return

    out_df = pd.DataFrame(results)
    # reorder columns
    cols = ["video_id", "mean_ear", "std_ear", "min_ear",
            "blink_count", "blink_rate_per_min",
            "mean_blink_duration", "std_blink_duration"]
    out_df = out_df[cols]
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone. EAR features saved to: {OUTPUT_CSV}")
    print(out_df.describe())


if __name__ == "__main__":
    run_ear_extraction()
