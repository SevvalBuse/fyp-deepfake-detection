# FYP – Deepfake Video Detection via Physiological Signals

This Final Year Project (BSc Computer Science) investigates deepfake video
detection through the analysis of physiological signals, with a particular
focus on algorithmic fairness across skin tone groups.

Unlike traditional artifact-based detectors, this project leverages
biological cues — eye-blink dynamics and remote photoplethysmography (rPPG) —
to improve robustness against unseen manipulations and real-world compression.

A key objective is to evaluate whether physiological-based deepfake detection
systems behave consistently across diverse skin tone groups, using the
Individual Typology Angle (ITA) scale as a fairness-aware audit mechanism.

---

## Setup & Requirements

### 1. Install Dependencies

Activate your virtual environment and install required libraries:

```bash
pip install -r requirements.txt
```

### 2. Download External Models

This project requires the Dlib 68-point Face Landmark Predictor.

Download `shape_predictor_68_face_landmarks.dat` and place it inside the `src/` folder.

> Note: This file is ~100MB and is not tracked by Git. You must download it manually to run any landmark-dependent module.

---

## Dataset

- **Source:** FaceForensics++ (FF++), c23 compression, Deepfakes method
- **Size:** 1,026 videos (513 real, 513 fake)
- **Construction:**
  - Initial 66 videos manually curated and labelled using `src/utils/manual_annotator.py`
  - Expanded to 1,026 using automated ITA-based skin tone selection (`src/preprocessing/expand_audit_set.py`)
  - Selection balanced across light (ITA > 41), medium (10 < ITA ≤ 41), and dark (ITA ≤ 10) groups
  - Skin tone measured objectively from forehead region using the ITA formula — no manual labelling for expanded set
- **Labels:** `data/output/dataset_bias_audit.csv` — `video_id`, `is_deepfake`, `gender_presentation`, `skin_tone_group`
- **Bias audit subset:** `data/output/bias_audit_ids.csv` — 300 videos with balanced ITA groups (100 per group) used for fairness evaluation

---

## Pipeline

All scripts are run from the **project root**. Run in the following order:

### 1. `src/preprocessing/physio_extractor.py`
Detects faces, extracts forehead and cheek ROIs per frame, records BGR colour signals.
- **Input:** `data/audit_set/` videos, `data/output/dataset_bias_audit.csv`
- **Output:** `data/signals/audit_ff/raw/*.npy`, `data/output/raw_metadata.csv`

### 2. `src/preprocessing/dual_algo_processor.py`
Applies CHROM and POS rPPG algorithms + Butterworth bandpass filter (0.7–3.0 Hz).
- **Input:** `data/signals/audit_ff/raw/*.npy`, `data/output/raw_metadata.csv`
- **Output:** `data/signals/audit_ff/clean/*_chrom.npy`, `*_pos.npy`

### 3. `src/analysis/signal_analyser.py`
Computes SNR and estimated BPM for each clean signal via FFT.
- **Input:** `data/signals/audit_ff/clean/*.npy`, `data/output/raw_metadata.csv`
- **Output:** `data/output/rppg_method_comparison.csv`

### 4. `src/preprocessing/ear_extractor.py`
Extracts 7 eye-blink features per video using dlib 68-point landmarks and Eye Aspect Ratio (EAR).
- **Input:** `data/audit_set/` videos, `data/output/dataset_bias_audit.csv`
- **Output:** `data/output/ear_features.csv`

### 5. `src/preprocessing/compute_audit_ita.py`
Recomputes ITA values for all videos using the correct OpenCV LAB normalisation.
- **Input:** `data/audit_set/` videos, `data/output/dataset_bias_audit.csv`
- **Output:** `data/output/ita_objective_audit.csv`

### 6. `src/analysis/feature_merger.py`
Merges rPPG features, EAR features, and ITA values into a single feature matrix.
- **Input:** `data/output/rppg_method_comparison.csv`, `data/output/ear_features.csv`, `data/output/ita_objective_audit.csv`
- **Output:** `data/output/unified_features.csv`

---

## Output Files

| File | Description |
|---|---|
| `data/output/dataset_bias_audit.csv` | Master list of 1,026 videos with labels and demographic metadata |
| `data/output/raw_metadata.csv` | FPS and frame count per video |
| `data/output/rppg_method_comparison.csv` | SNR and BPM per video per rPPG method |
| `data/output/ear_features.csv` | 7 blink features per video |
| `data/output/ita_objective_audit.csv` | Objective ITA skin tone values for all videos |
| `data/output/unified_features.csv` | Merged feature matrix for classifier training (1,026 rows, 12 features) |
| `data/output/bias_audit_ids.csv` | 300-video ITA-balanced subset for fairness evaluation |

---

## Report Visuals

All charts are saved to `data/report_visuals/`:

| File | Description |
|---|---|
| `model_comparison.png` | All models — accuracy, precision, recall, F1 |
| `model_accuracy_only.png` | Accuracy-only bar chart |
| `correlation_matrix_final.png` | 13×13 feature correlation heatmap |
| `feature_target_correlation.png` | Per-feature correlation with deepfake label |
| `bias_audit_model_comparison.png` | XGBoost vs RF by ITA skin tone group |
| `bias_audit_xgb_mitigation.png` | XGBoost baseline vs class weights vs ThresholdOpt |
| `individual_models/*.png` | Per-model heatmaps with ITA group breakdown |

---

## ITA Skin Tone Grouping

Skin tone groups are derived from the ITA (Individual Typology Angle) scale:

| Group | ITA Range |
|---|---|
| Dark | ITA ≤ 10 |
| Medium | 10 < ITA ≤ 41 |
| Light | ITA > 41 |

ITA is measured directly from each video's forehead region using the LAB colour space. For fake videos, ITA reflects the face shown in the video (the source identity), not the target.

---

## Status

In progress – Final Year Project (BSc Computer Science)

**Completed:** Dataset construction, rPPG extraction (CHROM + POS), SNR/BPM analysis, EAR blink extraction, ITA audit

**In progress:** Feature merging → classifier training → bias-stratified evaluation → Streamlit dashboard
