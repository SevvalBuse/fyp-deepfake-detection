import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- CONFIG ---
REPORT_DIR = "data/report_visuals"
os.makedirs(REPORT_DIR, exist_ok=True)

def generate_bias_correlation():
    print("Generating Forensic Correlation Matrix...")

    # 1. Load your results
    try:
        manual_df = pd.read_csv("data/output/dataset_bias_audit.csv")
        ita_df = pd.read_csv("data/output/ita_objective_audit.csv")
        snr_df = pd.read_csv("data/output/rppg_method_comparison.csv")
    except FileNotFoundError as e:
        print(f"Error: Missing files. {e}")
        return

    # Use CHROM SNR only (one row per video)
    snr_df = snr_df[snr_df['method'] == 'CHROM'].copy()

    # 2. Harmonize IDs
    for df in [manual_df, ita_df, snr_df]:
        df['join_id'] = df['video_id'].apply(lambda x: str(x).split('/')[-1].replace('.mp4', ''))

    # 3. Merge
    master_df = pd.merge(manual_df, ita_df[['join_id', 'measured_ita']], on="join_id")
    master_df = pd.merge(master_df, snr_df[['join_id', 'measured_snr', 'measured_bpm']], on="join_id")

    # 4. Convert Skin Tone to Numeric
    skin_mapping = {'dark': 0, 'medium': 1, 'light': 2}
    master_df['skin_numeric'] = master_df['skin_tone_group'].map(skin_mapping)

    # 5. NEW LOGIC: Underscore (_) means Manipulated/Fake
    # Example: '111' is Real (0), '111_094' is Fake (1)
    master_df['is_deepfake'] = master_df['join_id'].apply(
        lambda x: 1 if "_" in str(x) else 0
    )

    # 6. Correlation Math
    corr_cols = ['skin_numeric', 'measured_ita', 'measured_snr', 'measured_bpm', 'is_deepfake']
    correlation_matrix = master_df[corr_cols].corr()

    # 7. Heatmap Generation
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Forensic Feature Correlation Matrix\n(Real vs. Deepfake Signal Analysis)", fontsize=14, fontweight='bold')
    
    save_path = os.path.join(REPORT_DIR, "correlation_matrix_final.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"SUCCESS: Analysis complete. Plot saved to {save_path}")
    print(correlation_matrix[['is_deepfake']])

if __name__ == "__main__":
    generate_bias_correlation()