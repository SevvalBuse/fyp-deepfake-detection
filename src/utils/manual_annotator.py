import cv2
import pandas as pd
import os
import random

# U
DATASET_PATHS = [
    "data/audit_set/original_sequences/youtube/c23/videos",
    "data/audit_set/manipulated_sequences/Deepfakes/c23/videos"
]

LOG_FILE = "data/output/dataset_bias_audit.csv"

GENDER_OPTIONS = {"m", "f", "other", "unclear"}
TONE_OPTIONS = {"light", "medium", "dark", "unclear"}

def get_representative_frame(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target = total // 2 if total > 0 else 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)

    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def load_existing_ids():
    if not os.path.exists(LOG_FILE):
        return set()
    try:
        df = pd.read_csv(LOG_FILE)
        return set(df["video_id"].astype(str).tolist())
    except Exception:
        return set()

def audit_dataset(num_samples=100):
    existing = load_existing_ids()

    all_videos = []
    for folder in DATASET_PATHS:
        if not os.path.exists(folder):
            print(f" Folder not found: {folder}")
            continue
        for f in os.listdir(folder):
            if f.endswith(".mp4"):
                vid_id = f"{os.path.basename(os.path.dirname(folder))}/{f}"
                all_videos.append((vid_id, os.path.join(folder, f)))

    # remove already-audited
    remaining = [(vid_id, path) for (vid_id, path) in all_videos if vid_id not in existing]
    if not remaining:
        print(" Nothing new to audit.")
        return

    samples = random.sample(remaining, min(num_samples, len(remaining)))
    results = []

    for vid_id, path in samples:
        frame = get_representative_frame(path)
        if frame is None:
            print(f" Could not read frame: {vid_id}")
            continue

        window = "Audit (press any key to label, ESC to skip)"
        cv2.imshow(window, frame)
        key = cv2.waitKey(0)
        cv2.destroyWindow(window)

        if key == 27:  # ESC
            print(f" Skipped: {vid_id}")
            continue

        print(f"\n--- Auditing: {vid_id} ---")
        gender = input("Perceived gender presentation (m/f/other/unclear): ").strip().lower()
        if gender not in GENDER_OPTIONS:
            gender = "unclear"

        tone = input("Perceived skin tone group (light/medium/dark/unclear): ").strip().lower()
        if tone not in TONE_OPTIONS:
            tone = "unclear"

        results.append({"video_id": vid_id, "gender_presentation": gender, "skin_tone_group": tone})

    if not results:
        print(" No entries saved.")
        return

    df_new = pd.DataFrame(results)

    # append
    if os.path.exists(LOG_FILE):
        df_old = pd.read_csv(LOG_FILE)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_csv(LOG_FILE, index=False)
    print(f"\n Audit complete. Saved/updated: {LOG_FILE}")

if __name__ == "__main__":
    audit_dataset(num_samples=100)
