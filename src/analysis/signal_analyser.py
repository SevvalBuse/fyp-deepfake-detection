"""
Computes SNR and estimated BPM from the CHROM/POS filtered rPPG signals using
FFT-based spectral analysis. Reads all .npy files from data/signals/audit_ff/clean/
and saves per-video, per-method results to data/output/rppg_method_comparison.csv.

SNR is measured as signal power in a narrow band (±0.1 Hz) around the dominant
frequency peak, relative to total passband power (0.7–3.0 Hz).
"""
import numpy as np
import os
import pandas as pd
from scipy.fft import fft, fftfreq
from tqdm import tqdm

# --- CONFIG ---
CLEAN_DIR = "data/signals/audit_ff/clean"
FS_DEFAULT = 30.0

def calculate_snr_pro(signal, fs):
    """
    Advanced SNR Calculation:
    1. Removes DC component.
    2. Measures signal power in a narrow band (+/- 0.1 Hz) around peak.
    3. Measures noise power in the broader passband (0.7 - 3.0 Hz).
    """
    n = len(signal)
    if n == 0: return -99, 0
    
    yf = fft(signal)
    xf = fftfreq(n, 1/fs)
    psd = np.abs(yf)**2
    
    # Define Human Heart Rate Range (0.7 to 3.0 Hz / 42-180 BPM)
    passband_mask = (xf >= 0.7) & (xf <= 3.0)
    
    if not any(passband_mask): return -99, 0
    
    peak_idx = np.argmax(psd[passband_mask])
    peak_freq = xf[passband_mask][peak_idx]
    
    # Narrow Signal Band (+/- 0.1 Hz)
    signal_mask = (xf >= peak_freq - 0.1) & (xf <= peak_freq + 0.1)
    signal_power = np.sum(psd[signal_mask])
    
    total_passband_power = np.sum(psd[passband_mask])
    noise_power = total_passband_power - signal_power
    
    if noise_power <= 0: return 20, peak_freq * 60 
    
    snr = 10 * np.log10(signal_power / noise_power)
    return round(snr, 2), round(peak_freq * 60, 2)

def analyze_benchmarks():
    files = [f for f in os.listdir(CLEAN_DIR) if f.endswith('.npy')]
    results = []

    # Load metadata for specific video FPS
    meta_path = "data/output/raw_metadata.csv"
    meta_df = pd.read_csv(meta_path) if os.path.exists(meta_path) else None

    print(f"Benchmarking {len(files)} signals...")
    for f in tqdm(files):
        # Determine method (CHROM or POS)
        method = "CHROM" if "_chrom" in f else "POS"
        base_filename = f.replace("_chrom.npy", ".mp4").replace("_pos.npy", ".mp4")
        
        # Get correct FPS
        fps = FS_DEFAULT
        if meta_df is not None:
            match = meta_df[meta_df['filename'] == base_filename]
            if not match.empty:
                fps = float(match.iloc[0]['fps'])

        signal = np.load(os.path.join(CLEAN_DIR, f))
        snr_val, bpm_val = calculate_snr_pro(signal, fps)
        
        results.append({
            "video_id": f"c23/{base_filename}",
            "method": method,
            "measured_snr": snr_val,
            "measured_bpm": bpm_val
        })

    df_analysis = pd.DataFrame(results)
    df_analysis.to_csv("data/output/rppg_method_comparison.csv", index=False)

    # Summary Statistics
    summary = df_analysis.groupby("method")[["measured_snr"]].mean()
    print("\n--- Benchmark Summary ---")
    print(summary)
    print("\nResults saved to data/output/rppg_method_comparison.csv")

if __name__ == "__main__":
    analyze_benchmarks()