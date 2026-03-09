import numpy as np
import os
import pandas as pd
from scipy.signal import butter, filtfilt
from tqdm import tqdm

# --- CONFIGURATION ---
RAW_DIR = "data/signals/audit_ff/raw"
CLEAN_DIR = "data/signals/audit_ff/clean"
META_PATH = "data/output/raw_metadata.csv"  # Per-video FPS information
os.makedirs(CLEAN_DIR, exist_ok=True)

# 1. Filtering: Butterworth Bandpass Filter
def apply_butterworth(signal, fs=30, order=5):
    nyq = 0.5 * fs
    # Ideal heart rate range: 42–180 BPM (0.7 – 3.0 Hz)
    low = 0.7 / nyq
    high = 3.0 / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# 2. Signal Extraction: CHROM Method
def chrom_method(rgb_signal):
    rgb_mean = np.mean(rgb_signal, axis=0)
    rgb_norm = rgb_signal / (rgb_mean + 1e-8)
    
    X = 3 * rgb_norm[:, 0] - 2 * rgb_norm[:, 1]
    Y = 1.5 * rgb_norm[:, 0] + rgb_norm[:, 1] - 1.5 * rgb_norm[:, 2]
    
    alpha = np.std(X) / (np.std(Y) + 1e-8)
    return X - alpha * Y

# 3. Signal Extraction: POS Method
def pos_method(rgb_signal):
    rgb_mean = np.mean(rgb_signal, axis=0)
    cn = rgb_signal / (rgb_mean + 1e-8)
    
    S1 = cn[:, 1] - cn[:, 2]        # G - B
    S2 = cn[:, 1] + cn[:, 2] - 2 * cn[:, 0]  # G + B - 2R
    
    alpha = np.std(S1) / (np.std(S2) + 1e-8)
    return S1 + alpha * S2

def run_benchmark():
    # Load metadata (for FPS information)
    if not os.path.exists(META_PATH):
        print(f"Error: {META_PATH} not found!")
        return
        
    meta_df = pd.read_csv(META_PATH)
    files = [f for f in os.listdir(RAW_DIR) if f.endswith('_raw.npy')]
    
    print(f"Processing the 66-video audit set (CHROM vs POS)...")
    
    for f in tqdm(files):
        # Match video ID with metadata
        v_id = f.replace('_raw.npy', '.mp4')
        video_meta = meta_df[meta_df['filename'] == v_id]
        
        if video_meta.empty:
            fps = 30.0  # Default FPS
        else:
            fps = float(video_meta.iloc[0]['fps'])

        # Load raw signal data
        raw_data = np.load(os.path.join(RAW_DIR, f), allow_pickle=True)
        
        # Triple-ROI Averaging (Forehead + Cheeks)
        # Convert BGR to RGB (OpenCV reads images in BGR format)
        rgb_signal = np.array(
            [np.mean(frame['rgb'], axis=0) for frame in raw_data]
        )[:, ::-1]

        # --- PROCESSING: CHROM + Butterworth ---
        bvp_chrom = chrom_method(rgb_signal)
        clean_chrom = apply_butterworth(bvp_chrom, fs=fps)
        np.save(
            os.path.join(CLEAN_DIR, f.replace('_raw.npy', '_chrom.npy')),
            clean_chrom
        )
        
        # --- PROCESSING: POS + Butterworth ---
        bvp_pos = pos_method(rgb_signal)
        clean_pos = apply_butterworth(bvp_pos, fs=fps)
        np.save(
            os.path.join(CLEAN_DIR, f.replace('_raw.npy', '_pos.npy')),
            clean_pos
        )

if __name__ == "__main__":
    run_benchmark()