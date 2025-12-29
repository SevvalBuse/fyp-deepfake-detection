# FYP – Deepfake Video Detection via Physiological Signals

This Final Year Project focuses on detecting deepfake videos by analysing
physiological signals, specifically eye-blink dynamics and remote
photoplethysmography (rPPG).

The project aims to improve robustness, interpretability, and cross-dataset
generalisation compared to traditional artifact-based deepfake detectors.

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

Note: This file is ~100MB and is intentionally not added to Git. You must download it manually to run the blink detection module.k detection module.



## Status
In progress – Final Year Project (BSc Computer Science)
