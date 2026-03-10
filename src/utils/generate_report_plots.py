import numpy as np
import matplotlib.pyplot as plt
import os

# --- PATHS ---
RAW_DIR = "data/signals/audit_ff/raw"
CLEAN_DIR = "data/signals/audit_ff/clean"
REPORT_DIR = "data/report_visuals"
os.makedirs(REPORT_DIR, exist_ok=True)

def create_algorithm_benchmark_plot(video_id, ita_value, label):
    """
    Generates a 3-tier comparison: Raw vs. CHROM vs. POS.
    Used to justify algorithm selection in the final report.
    """
    clean_name = video_id.split("/")[-1].replace(".mp4", "")
    
    raw_path = os.path.join(RAW_DIR, f"{clean_name}_raw.npy")
    chrom_path = os.path.join(CLEAN_DIR, f"{clean_name}_chrom.npy")
    pos_path = os.path.join(CLEAN_DIR, f"{clean_name}_pos.npy")

    # Guard: Ensure all files exist
    if not all(os.path.exists(p) for p in [raw_path, chrom_path, pos_path]):
        print(f"SKIPPING: Missing files for {clean_name}")
        return

    # 1. Load Data
    raw_data = np.load(raw_path, allow_pickle=True)
    # Extract Raw Green Channel from Forehead (Index 0, Green is Index 1 in BGR)
    raw_green = np.array([frame['rgb'][0][1] for frame in raw_data]) 
    
    clean_chrom = np.load(chrom_path)
    clean_pos = np.load(pos_path)

    # 2. Create the Visual (3 rows)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    fig.suptitle(f"Algorithmic Comparison: {clean_name} ({label})\nObjective ITA: {ita_value:.2f}°", 
                 fontsize=16, fontweight='bold')

    # --- Subplot 1: RAW INPUT ---
    ax1.plot(raw_green, color='#2ecc71', alpha=0.7, label='Raw Green Channel (Forehead)')
    ax1.set_title("Input: Raw Pixel Intensity Fluctuations", loc='left', fontsize=12, fontweight='bold')
    ax1.set_ylabel("Intensity")
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(loc='upper right')

    # --- Subplot 2: CHROM METHOD ---
    ax2.plot(clean_chrom, color='#e74c3c', linewidth=1.5, label='CHROM + Butterworth')
    ax2.set_title("Method A: Chrominance-based (CHROM) Pulse", loc='left', fontsize=12, fontweight='bold')
    ax2.set_ylabel("Amplitude")
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend(loc='upper right')

    # --- Subplot 3: POS METHOD ---
    ax3.plot(clean_pos, color='#3498db', linewidth=1.5, label='POS + Butterworth')
    ax3.set_title("Method B: Plane-Orthogonal-to-Skin (POS) Pulse", loc='left', fontsize=12, fontweight='bold')
    ax3.set_xlabel("Time (Frames)")
    ax3.set_ylabel("Amplitude")
    ax3.grid(True, linestyle='--', alpha=0.3)
    ax3.legend(loc='upper right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the comparison
    save_file = os.path.join(REPORT_DIR, f"benchmark_{clean_name}.png")
    plt.savefig(save_file, dpi=300)
    plt.close()
    print(f"SUCCESS: Comparison plot saved to {save_file}")

if __name__ == "__main__":
    # Example benchmark cases (using your previous ITA results)
    benchmarks = [
        ("c23/469.mp4", 51.62, "Light Skin Case"),
        ("c23/111_094.mp4", 12.65, "Dark Skin Case")
    ]
    
    for vid, ita, lbl in benchmarks:
        create_algorithm_benchmark_plot(vid, ita, lbl)