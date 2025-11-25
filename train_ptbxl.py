import os
import numpy as np
import pandas as pd
import wfdb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time
from ast import literal_eval

# ============================================================
# 1. CONFIGURATION — CHANGE ONLY THIS PART IF NEEDED
# ============================================================

BASE_DIR = "/Users/arnavbhatnagar/Downloads/ECG-PTBXL-Capstone-main/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/"
NPPY_DIR = "ecg_npy"
EPOCHS = 10
BATCH_SIZE = 32
SEQ_LEN = 2000  # shorten for speed

# ============================================================
# 2. MODEL DEFINITION
# ============================================================

class FastECGModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(12, 64, 7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = self.net(x)
        x = x.squeeze(-1)
        return self.fc(x)

# ============================================================
# 3. FAST DATASET
# ============================================================

class FastECGDataset(Dataset):
    def __init__(self, df, npy_dir=NPPY_DIR):
        self.df = df
        self.npy_dir = npy_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ecg_id = row["ecg_id"]
        signal = np.load(f"{self.npy_dir}/{ecg_id}.npy")
        signal = torch.tensor(signal, dtype=torch.float32)
        label = 1 if row["binary_target"] == "Risk" else 0
        return signal, torch.tensor(label, dtype=torch.long)

# ============================================================
# PROGRAM ENTRY POINT (CRITICAL FOR WINDOWS)
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # LOAD METADATA
    # ========================================================
    print("Loading metadata...")

    df = pd.read_csv(BASE_DIR + "ptbxl_database.csv")
    scp_df = pd.read_csv(BASE_DIR + "scp_statements.csv", index_col=0)

    df["scp_codes"] = df["scp_codes"].apply(literal_eval)

    def map_classes(code_dict):
        labels = []
        for key in code_dict.keys():
            if key in scp_df.index:
                labels.append(scp_df.loc[key, "diagnostic_class"])
        return list(set(labels))

    df["diagnostic_class"] = df["scp_codes"].apply(map_classes)
    df["binary_target"] = df["diagnostic_class"].apply(
        lambda x: "Risk" if any(c in ["MI", "STTC", "ISC"] for c in x) else "Normal"
    )

    df = df[["ecg_id", "filename_hr", "binary_target"]]
    print(df["binary_target"].value_counts())

    # ========================================================
    # PREPROCESS & CACHE SIGNALS
    # ========================================================

    def preprocess_signals(df, base_path, save_dir=NPPY_DIR):
        os.makedirs(save_dir, exist_ok=True)
        print("Caching ECG signals...")
        for i, row in tqdm(df.iterrows(), total=len(df)):
            record_path = base_path + row["filename_hr"].replace(".mat", "")
            npy_file = f"{save_dir}/{row['ecg_id']}.npy"

            if os.path.exists(npy_file):
                continue

            signal, _ = wfdb.rdsamp(record_path)
            signal = signal.T[:, :SEQ_LEN]

            mn = signal.min(axis=1, keepdims=True)
            mx = signal.max(axis=1, keepdims=True)
            signal = (signal - mn) / (mx - mn + 1e-8)

            np.save(npy_file, signal.astype(np.float32))

        print("Caching complete.")

    preprocess_signals(df, BASE_DIR)

    # ========================================================
    # DATA LOADERS
    # ========================================================

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_ds = FastECGDataset(train_df)
    test_ds = FastECGDataset(test_df)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ========================================================
    # TRAINING SETUP
    # ========================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = FastECGModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        start_time = time.time()

        print(f"\n====== Epoch {epoch+1}/{EPOCHS} ======")
        batch_bar = tqdm(train_loader, desc="Training", ncols=100)

        for signals, labels in batch_bar:
            signals, labels = signals.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        print("Epoch Loss:", total_loss / len(train_loader))
        print("Epoch Time:", round(time.time() - start_time, 2), "sec")

    # ========================================================
    # EVALUATION
    # ========================================================

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for signals, labels in test_loader:
            signals, labels = signals.to(device), labels.to(device)
            outputs = model(signals)
            predicted = outputs.argmax(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print("\n=========== FINAL TEST RESULTS ===========")
    print("Accuracy:", correct / total)
    print("==========================================")
