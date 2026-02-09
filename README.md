# FYP – Deepfake Video Detection via Physiological Signals

This Final Year Project (BSc Computer Science) investigates deepfake video
detection through the analysis of physiological signals, with a particular
focus on algorithmic fairness and cross-dataset generalisation.

Unlike traditional artifact-based detectors, this project leverages
biological cues, eye-blink dynamics and remote photoplethysmography (rPPG),
to improve robustness against unseen manipulations and real-world bias.

A key objective of this work is to evaluate whether physiological-based
deepfake detection systems behave consistently across diverse skin tone
groups, using the Individual Typology Angle (ITA) scale as a fairness-aware
audit mechanism.

## Setup & Requirements

### 1. Install Dependencies

Activate your virtual environment and install required libraries:

```bash
pip install -r requirements.txt
```

### 2. Download External Models
This project requires the Dlib 68-point Face Landmark Predictor.

Download the file from: https://github.com/ageitgey/face_recognition_models/blob/master/face_recognition_models/models/shape_predictor_68_face_landmarks.dat


Place the shape_predictor_68_face_landmarks.dat file inside the src/ folder.

Note: This file is ~100MB and is intentionally not added to Git. You must download it manually to run the blink detection module.





## Preliminary Results Data

Preliminary eye-blink and rPPG analyses were conducted on a limited subset of
videos to validate the signal extraction pipeline. The resulting plots and
qualitative analyses are presented in the project report but are not publicly
released due to dataset licensing constraints.


## Data & Methodology

### Datasets
- Training Dataset: 
  Celeb-DF v2 — used for large-scale learning of physiological signal patterns.

- Independent Audit Dataset:
  A curated subset of 66 videos from FaceForensics++ (FF++), manually balanced
  across *light*, *medium*, and *dark* skin tone groups to enable controlled
  bias analysis.

### Signal Extraction
- rPPG:
  Extracted using both **CHROM** and **POS** algorithms, followed by signal
  cleaning and frequency filtering.

- Eye-Blink Dynamics:  
  Eye Aspect Ratio (EAR) computed using Dlib’s 68-point facial landmark model.

Algorithm selection was informed through comparative visual analysis of signal
quality across different skin tone groups.



## Scripts Overview

  - `src/physio_extractor.py`  
  Core signal extraction engine performing face tracking, ROI extraction,
  and raw physiological signal collection.

- `src/signal_analyser.py`  
  Computes signal quality and forensic metrics such as SNR and BPM to assess
  rPPG reliability.

- `src/bias_auditor.py`  
  Evaluation module designed to analyse model performance across skin tone
  groups using the independent audit dataset.

- `src/check_rois.py`  
  Diagnostic utility for visually verifying facial landmark alignment and ROI
  placement (forehead and cheeks) across different head poses and skin tones.

- `src/dual_algo_processor.py`  
  Implements signal processing pipelines including Butterworth filtering and
  normalisation for CHROM and POS outputs.

- `src/generate_report_plots.py`  
  Generates comparative visualisations of rPPG signals extracted using CHROM
  and POS algorithms. These plots were used to inform algorithm selection
  across different skin tone groups.


## Status
In progress – Final Year Project (BSc Computer Science)
