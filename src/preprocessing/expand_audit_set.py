"""
Copies all remaining temp_scan videos (not already in audit_set) to audit_set/
and appends their metadata to dataset_bias_audit.csv.

No ITA filtering — takes all available videos to maximise dataset size.
The original 260 ITA-balanced videos are preserved for bias audit.
"""
import pandas as pd
import os
import shutil
from tqdm import tqdm

# --- CONFIG ---
AUDIT_CSV  = "data/output/dataset_bias_audit.csv"

TEMP_REAL  = "data/temp_scan/original_sequences/youtube/c23/videos"
TEMP_FAKE  = "data/temp_scan/manipulated_sequences/Deepfakes/c23/videos"

AUDIT_REAL = "data/audit_set/original_sequences/youtube/c23/videos"
AUDIT_FAKE = "data/audit_set/manipulated_sequences/Deepfakes/c23/videos"


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
    print("\nCopying real videos...")
    for filename in tqdm(real_files):
        src = os.path.join(TEMP_REAL, filename)
        dst = os.path.join(AUDIT_REAL, filename)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        new_rows.append({"video_id": f"c23/{filename}", "is_deepfake": 0})

    # Copy fake videos
    print("\nCopying fake videos...")
    for filename in tqdm(fake_files):
        src = os.path.join(TEMP_FAKE, filename)
        dst = os.path.join(AUDIT_FAKE, filename)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        new_rows.append({"video_id": f"c23/{filename}", "is_deepfake": 1})

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
