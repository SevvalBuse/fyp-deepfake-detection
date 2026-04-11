"""
Merges ita_objective_audit.csv, rppg_method_comparison.csv, and ear_features.csv
with the video labels from dataset_bias_audit.csv into a single unified_features.csv.
This CSV is the main input for all classifiers. Joins on video_id and reports
how many rows have complete data before saving.
"""
import pandas as pd
import os

# --- CONFIG ---
AUDIT_CSV    = "data/output/dataset_bias_audit.csv"
ITA_CSV      = "data/output/ita_objective_audit.csv"
RPPG_CSV     = "data/output/rppg_method_comparison.csv"
EAR_CSV      = "data/output/ear_features.csv"
OUTPUT_CSV   = "data/output/unified_features.csv"


def merge_features():
    # --- Load all CSVs ---
    audit = pd.read_csv(AUDIT_CSV)
    ita   = pd.read_csv(ITA_CSV)
    rppg  = pd.read_csv(RPPG_CSV)
    ear   = pd.read_csv(EAR_CSV)

    # --- Pivot rPPG: one row per video with CHROM and POS columns ---
    rppg_pivot = rppg.pivot(index="video_id", columns="method", values=["measured_snr", "measured_bpm"])
    rppg_pivot.columns = [f"{method.lower()}_{metric}" for metric, method in rppg_pivot.columns]
    rppg_pivot = rppg_pivot.reset_index()
    # Columns: video_id, chrom_measured_snr, pos_measured_snr, chrom_measured_bpm, pos_measured_bpm
    rppg_pivot.rename(columns={
        "chrom_measured_snr": "chrom_snr",
        "pos_measured_snr":   "pos_snr",
        "chrom_measured_bpm": "chrom_bpm",
        "pos_measured_bpm":   "pos_bpm"
    }, inplace=True)

    # --- Merge all on video_id ---
    df = audit[["video_id", "is_deepfake"]].copy()
    df = df.merge(ita,        on="video_id", how="left")
    df = df.merge(rppg_pivot, on="video_id", how="left")
    df = df.merge(ear,        on="video_id", how="left")

    # --- Report ---
    print(f"Total videos:      {len(df)}")
    print(f"Complete rows:     {df.dropna().shape[0]}")
    print(f"Missing any data:  {df[df.isnull().any(axis=1)].shape[0]}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nClass balance:\n{df['is_deepfake'].value_counts().rename({0: 'real', 1: 'fake'})}")

    # --- Save ---
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved to {OUTPUT_CSV}")


if __name__ == "__main__":
    merge_features()