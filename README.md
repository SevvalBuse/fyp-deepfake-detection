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

Download `shape_predictor_68_face_landmarks.dat` from:
https://github.com/ageitgey/face_recognition_models

Place it inside the `src/` folder.

> Note: This file is ~100MB and is not tracked by Git. You must download it manually to run any landmark-dependent module.

---

## Dataset

- **Source:** FaceForensics++ (FF++), c23 compression
- **Size:** 260 videos (130 real, 130 fake)
- **Construction:**
  - Initial 66 videos manually curated and labelled using `bias_auditor.py`
  - Expanded to 260 using automated ITA-based skin tone selection (`expand_audit_set.py`)
  - Selection balanced across light (ITA > 41), medium (10 < ITA ≤ 41), and dark (ITA ≤ 10) groups
  - Skin tone measured objectively from forehead region using the ITA formula — no manual labelling for expanded set
- **Labels:** `dataset_bias_audit.csv` — `video_id`, `is_deepfake`, `gender_presentation`, `skin_tone_group`

---

## Pipeline

Run scripts in the following order:

### 1. `src/physio_extractor.py`
Detects faces, extracts forehead and cheek ROIs per frame, records BGR colour signals and ITA.
- **Input:** `data/audit_set/` videos, `data/output/dataset_bias_audit.csv`
- **Output:** `data/signals/audit_ff/raw/*.npy`, `data/output/raw_metadata.csv`

### 2. `src/dual_algo_processor.py`
Applies CHROM and POS rPPG algorithms + Butterworth bandpass filter (0.7–3.0 Hz).
- **Input:** `data/signals/audit_ff/raw/*.npy`, `data/output/raw_metadata.csv`
- **Output:** `data/signals/audit_ff/clean/*_chrom.npy`, `*_pos.npy`

### 3. `src/signal_analyser.py`
Computes SNR and estimated BPM for each clean signal via FFT.
- **Input:** `data/signals/audit_ff/clean/*.npy`, `data/output/raw_metadata.csv`
- **Output:** `data/output/rppg_method_comparison.csv`

### 4. `src/ear_extractor.py`
Extracts 7 eye-blink features per video using dlib 68-point landmarks and Eye Aspect Ratio (EAR).
- **Input:** `data/audit_set/` videos, `data/output/dataset_bias_audit.csv`
- **Output:** `data/output/ear_features.csv`

### 5. `src/compute_audit_ita.py`
Recomputes ITA values for all 260 videos using the correct OpenCV LAB normalisation.
- **Input:** `data/audit_set/` videos, `data/output/dataset_bias_audit.csv`
- **Output:** `data/output/ita_objective_audit.csv`

---

## Output Files

| File | Description |
|---|---|
| `data/output/dataset_bias_audit.csv` | Master list of 260 videos with labels and demographic metadata |
| `data/output/raw_metadata.csv` | FPS and frame count per video |
| `data/output/rppg_method_comparison.csv` | SNR and BPM per video per rPPG method (520 rows) |
| `data/output/ear_features.csv` | 7 blink features per video |
| `data/output/ita_objective_audit.csv` | Objective ITA skin tone values for all 260 videos |
| `data/output/unified_features.csv` | *(to be generated)* Merged feature matrix for classifier training |

---

## Scripts Overview

| Script | Purpose |
|---|---|
| `src/bias_auditor.py` | Manual annotation UI — used to label the initial 66 videos |
| `src/expand_audit_set.py` | Automated ITA-based dataset expansion from temp_scan to audit_set |
| `src/physio_extractor.py` | Raw rPPG signal extraction (forehead + cheeks, BGR per frame) |
| `src/dual_algo_processor.py` | CHROM and POS rPPG algorithms + Butterworth filtering |
| `src/signal_analyser.py` | FFT-based SNR and BPM computation |
| `src/ear_extractor.py` | Batch eye-blink (EAR) feature extraction |
| `src/compute_audit_ita.py` | Correct ITA computation for all audit set videos |
| `src/ita_scanner.py` | ITA scanning utility for large video collections |
| `src/generate_report_plots.py` | Comparative rPPG signal visualisations (Raw vs CHROM vs POS) |
| `src/generate_correlation.py` | Correlation matrix heatmap across features and demographics |
| `src/check_rois.py` | Diagnostic tool for verifying facial ROI placement |

---

## ITA Skin Tone Grouping

Skin tone groups are derived from the ITA (Individual Typology Angle) scale:

| Group | ITA Range |
|---|---|
| Dark | ITA ≤ 10 |
| Medium | 10 < ITA ≤ 41 |
| Light | ITA > 41 |

ITA is measured directly from each video's forehead region. For fake videos, ITA reflects the face shown in the video (the source identity), not the target.

---

## Status

In progress – Final Year Project (BSc Computer Science)

**Completed:** Dataset construction, rPPG extraction (CHROM + POS), SNR/BPM analysis, EAR blink extraction, ITA audit

**In progress:** Feature merging → classifier training → bias-stratified evaluation → Streamlit dashboard
