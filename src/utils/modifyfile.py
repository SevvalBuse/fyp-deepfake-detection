import pandas as pd
import os

ORIG_DIR = "data/audit_set/original_sequences/youtube/c23/videos"
FAKE_DIR = "data/audit_set/manipulated_sequences/Deepfakes/c23/videos"



def label_video(video_id):
    filename = str(video_id).split("/")[-1]
    if os.path.exists(os.path.join(FAKE_DIR, filename)):
        return 1
    if os.path.exists(os.path.join(ORIG_DIR, filename)):
        return 0
    return -1  # not found → investigate

if __name__ == "__main__":
    df = pd.read_csv("data/output/dataset_bias_audit.csv")
    df["is_deepfake"] = df["video_id"].apply(label_video)

    # sanity check
    print(df["is_deepfake"].value_counts())

    df.to_csv("data/output/dataset_bias_audit.csv", index=False)
    print("CSV updated safely.")
