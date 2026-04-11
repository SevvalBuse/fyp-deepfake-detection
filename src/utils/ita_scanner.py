"""
Scans all Celeb-DF v2 videos and computes ITA (skin tone) for each using the
forehead ROI with correct LAB normalisation. Saves results to
data/output/celebdf_ita_inventory.csv, which is consumed by
celeb_feature_pipeline.py to attach ITA values to the extracted features.
"""
import cv2
import dlib
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# --- CONFIGURATION ---
# Update these paths to match your local Celeb-DF v2 directory structure
DATA_ROOT = "data/celeb_df_v2"
PATHS = {
    "celeb_real":      (os.path.join(DATA_ROOT, "Celeb-real"),      0),
    "youtube_real":    (os.path.join(DATA_ROOT, "YouTube-real"),     0),
    "celeb_synthesis": (os.path.join(DATA_ROOT, "Celeb-synthesis"),  1),
}

SHAPE_PREDICTOR = "src/shape_predictor_68_face_landmarks.dat"
OUTPUT_CSV = "data/output/celebdf_ita_inventory.csv"

# Initialize Dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR)

def calculate_ita(l_val, b_val):
    """Calculates Individual Typology Angle from normalized Lab values."""
    ita = np.arctan2((l_val - 50), b_val) * (180 / np.pi)
    return ita

def scan_ita_fast(v_path):
    cap = cv2.VideoCapture(v_path)
    if not cap.isOpened():
        return None
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Optimization: Sample 10 frames across the first 200 frames to ensure
    # lighting variety while maintaining high processing speed.
    sample_indices = np.linspace(0, min(total_frames - 1, 200), 10).astype(int)
    
    ita_samples = []
    
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. Convert to Gray for Landmark Accuracy
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        if len(faces) > 0:
            # Focus on the primary subject (largest face)
            target_face = max(faces, key=lambda r: (r.right()-r.left()) * (r.bottom()-r.top()))
            shape = predictor(gray, target_face)
            
            # 2. Extract Forehead ROI (Stable for skin tone)
            f_bottom = min(shape.part(19).y, shape.part(24).y) - 5
            f_top = max(0, f_bottom - 20)
            f_left, f_right = shape.part(18).x, shape.part(25).x
            
            roi = frame[f_top:f_bottom, f_left:f_right]
            if roi.size > 0:
                # 3. Convert ROI to LAB space
                lab = cv2.cvtColor(roi, cv2.COLOR_BGR2Lab)
                l_mean, _, b_mean = cv2.mean(lab)[:3]
                
                # OpenCV stores L in [0,255] and b in [0,255].
                # Standard ITA formula requires L in [0,100] and b in [-128,127].
                l_norm = l_mean * (100 / 255)
                b_std  = b_mean - 128
                
                ita_val = calculate_ita(l_norm, b_std)
                ita_samples.append(ita_val)
    
    cap.release()
    return np.mean(ita_samples) if ita_samples else None

def run_inventory_scan():
    all_results = []
    
    for label_type, (folder_path, is_deepfake) in PATHS.items():
        if not os.path.exists(folder_path):
            print(f"Warning: Folder not found: {folder_path}")
            continue

        print(f"Scanning directory: {label_type} (is_deepfake={is_deepfake})...")
        files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]

        for f in tqdm(files):
            v_full_path = os.path.join(folder_path, f)
            measured_ita = scan_ita_fast(v_full_path)

            if measured_ita is not None:
                all_results.append({
                    "video_id": f,
                    "folder": label_type,
                    "is_deepfake": is_deepfake,
                    "measured_ita": round(measured_ita, 2)
                })

    # Save results to CSV for stratified sampling tomorrow
    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone. {len(df)} videos scanned and saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    run_inventory_scan()