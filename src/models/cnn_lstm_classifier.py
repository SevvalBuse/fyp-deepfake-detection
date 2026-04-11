"""
Experimental CNN-LSTM classifier trained directly on raw rPPG signal sequences
(CHROM + POS as a 2-channel input). Achieved 51.4% accuracy with severe
overfitting across all 5 folds, performing at random-chance level. This approach
was abandoned in favour of the classical ML classifiers (classifier.py) which
use hand-crafted features extracted from the same signals.
"""
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings("ignore")

# --- CONFIG ---
CLEAN_DIR   = "data/signals/audit_ff/clean"
AUDIT_CSV   = "data/output/dataset_bias_audit.csv"
SEQ_LEN     = 300   # fixed sequence length (truncate or pad all signals to this)
BATCH_SIZE  = 16
EPOCHS      = 30
LR          = 1e-3
N_SPLITS    = 5
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- DATASET ---
class RPPGDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chrom_path, pos_path, label = self.samples[idx]

        chrom = np.load(chrom_path).astype(np.float32)
        pos   = np.load(pos_path).astype(np.float32)

        chrom = (chrom - chrom.mean()) / (chrom.std() + 1e-8)
        pos   = (pos   - pos.mean())   / (pos.std()   + 1e-8)

        chrom = self._fix_length(chrom)
        pos   = self._fix_length(pos)

        # Stack as 2-channel signal: shape (2, SEQ_LEN)
        x = np.stack([chrom, pos], axis=0)
        return torch.tensor(x), torch.tensor(label, dtype=torch.long)

    def _fix_length(self, signal):
        if len(signal) >= SEQ_LEN:
            return signal[:SEQ_LEN]
        pad = SEQ_LEN - len(signal)
        return np.pad(signal, (0, pad), mode="constant")


# --- MODEL: CNN-LSTM ---
class CNNLSTM(nn.Module):
    """
    CNN-LSTM hybrid for rPPG signal classification.
    CNN blocks extract local temporal features from 2-channel input.
    Bidirectional LSTM then captures long-range sequential dependencies.
    """
    def __init__(self, lstm_hidden=64, lstm_layers=2):
        super().__init__()

        # CNN blocks — no global pooling, keep time dimension for LSTM
        self.cnn = nn.Sequential(
            # Block 1: (2, 300) → (32, 150)
            nn.Conv1d(2, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # Block 2: (32, 150) → (64, 75)
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # Block 3: (64, 75) → (128, 37)
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # Bidirectional LSTM: input (batch, T=37, 128)
        # Single layer to reduce overfitting on small dataset
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Classifier head: bidirectional doubles hidden size
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(lstm_hidden * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        # x: (batch, 2, SEQ_LEN)
        features = self.cnn(x)                # (batch, 128, T)
        features = features.permute(0, 2, 1)  # (batch, T, 128)
        _, (h_n, _) = self.lstm(features)     # h_n: (layers*2, batch, hidden)
        # Concat forward + backward final hidden states
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (batch, hidden*2)
        return self.classifier(h)


# --- TRAINING ---
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            out = model(x)
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    return np.array(all_preds), np.array(all_labels)


# --- LOAD SAMPLES ---
def load_samples():
    df = pd.read_csv(AUDIT_CSV)
    samples = []
    skipped = 0
    for _, row in df.iterrows():
        v_id  = str(row["video_id"])
        label = int(row["is_deepfake"])
        base  = v_id.split("/")[-1].replace(".mp4", "")

        chrom_path = os.path.join(CLEAN_DIR, f"{base}_chrom.npy")
        pos_path   = os.path.join(CLEAN_DIR, f"{base}_pos.npy")

        if not os.path.exists(chrom_path) or not os.path.exists(pos_path):
            skipped += 1
            continue
        samples.append((chrom_path, pos_path, label))

    print(f"Loaded {len(samples)} samples ({skipped} skipped — missing signal files)")
    return samples


# --- MAIN ---
def run():
    print(f"Device: {DEVICE}")
    samples = load_samples()

    labels  = np.array([s[2] for s in samples])
    indices = np.arange(len(samples))

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(indices, labels)):
        print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")

        train_samples = [samples[i] for i in train_idx]
        val_samples   = [samples[i] for i in val_idx]

        train_loader = DataLoader(RPPGDataset(train_samples), batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(RPPGDataset(val_samples),   batch_size=BATCH_SIZE, shuffle=False)

        model     = CNNLSTM().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        best_val_f1 = 0.0
        patience, no_improve = 7, 0

        for epoch in range(EPOCHS):
            loss = train_epoch(model, train_loader, optimizer, criterion)
            scheduler.step()

            # Early stopping check on val F1
            preds_ep, true_ep = evaluate(model, val_loader)
            val_f1 = f1_score(true_ep, preds_ep, zero_division=0)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                no_improve = 0
            else:
                no_improve += 1

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{EPOCHS} — Loss: {loss:.4f} | Val F1: {val_f1:.3f}")
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        preds, true = evaluate(model, val_loader)
        acc  = accuracy_score(true, preds)
        prec = precision_score(true, preds, zero_division=0)
        rec  = recall_score(true, preds, zero_division=0)
        f1   = f1_score(true, preds, zero_division=0)

        print(f"  Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")
        fold_results.append({"accuracy": acc, "precision": prec, "recall": rec, "f1": f1})

    results_df = pd.DataFrame(fold_results)
    print("\n========== CNN-LSTM RESULTS ==========")
    print(f"  Accuracy:  {results_df['accuracy'].mean():.3f} ± {results_df['accuracy'].std():.3f}")
    print(f"  Precision: {results_df['precision'].mean():.3f} ± {results_df['precision'].std():.3f}")
    print(f"  Recall:    {results_df['recall'].mean():.3f} ± {results_df['recall'].std():.3f}")
    print(f"  F1 Score:  {results_df['f1'].mean():.3f} ± {results_df['f1'].std():.3f}")


if __name__ == "__main__":
    run()
