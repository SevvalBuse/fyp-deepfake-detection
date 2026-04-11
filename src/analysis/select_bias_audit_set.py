"""
Selects 300 videos (100 Dark, 100 Medium, 100 Light) from ita_objective_audit.csv
for the held-out bias audit set and saves their IDs to bias_audit_ids.csv.
Selection prioritises the most extreme ITA values in each group to maximise
skin tone separation. The IDs in this file are permanently excluded from classifier
training in classifier.py and bias_auditor.py.
"""
import pandas as pd

# --- CONFIG ---
ITA_CSV    = "data/output/ita_objective_audit.csv"
AUDIT_CSV  = "data/output/dataset_bias_audit.csv"
OUTPUT_CSV = "data/output/bias_audit_ids.csv"

TARGET_PER_GROUP = 100  # 100 light + 100 medium + 100 dark = 300 total


def ita_group(ita):
    """Classify an ITA value into a skin tone group.
    Dark: ITA <= 10 | Medium: 10 < ITA <= 41 | Light: ITA > 41.
    """
    if ita <= 10:
        return "dark"
    elif ita <= 41:
        return "medium"
    else:
        return "light"


def run():
    ita = pd.read_csv(ITA_CSV).dropna()
    audit = pd.read_csv(AUDIT_CSV)

    # Merge to get is_deepfake label
    df = ita.merge(audit[["video_id", "is_deepfake"]], on="video_id", how="inner")
    df["ita_group"] = df["measured_ita"].apply(ita_group)

    print("Available per group:")
    print(df["ita_group"].value_counts())

    selected = []

    # Light: highest ITA first
    light = df[df["ita_group"] == "light"].sort_values("measured_ita", ascending=False)
    selected.append(light.head(TARGET_PER_GROUP))

    # Dark: lowest ITA first
    dark = df[df["ita_group"] == "dark"].sort_values("measured_ita", ascending=True)
    selected.append(dark.head(TARGET_PER_GROUP))

    # Medium: closest to midpoint (25.5)
    medium = df[df["ita_group"] == "medium"].copy()
    medium["dist"] = (medium["measured_ita"] - 25.5).abs()
    medium = medium.sort_values("dist")
    selected.append(medium.head(TARGET_PER_GROUP))

    result = pd.concat(selected, ignore_index=True)

    print(f"\nSelected {len(result)} videos for bias audit:")
    print(result["ita_group"].value_counts())
    print(f"\nReal/Fake split:")
    print(result["is_deepfake"].value_counts().rename({0: "real", 1: "fake"}))

    result[["video_id", "measured_ita", "ita_group", "is_deepfake"]].to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved to {OUTPUT_CSV}")


if __name__ == "__main__":
    run()
