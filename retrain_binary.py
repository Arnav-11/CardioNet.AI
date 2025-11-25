import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from ast import literal_eval
from tqdm import tqdm

# ============================
# CONFIG
# ============================
BASE_DIR = "/Users/arnavbhatnagar/Downloads/ECG-PTBXL-Capstone-main/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/"
NPPY_DIR = "ecg_npy"
EPOCHS = 3      # ONLY 3 EPOCHS needed to regenerate weights
BATCH_SIZE = 16
SEQ_LEN = 2000  # used during preprocessing

# ============================
# MODEL (BINARY)
# ============================
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
        self.fc = nn.Linear(128, 2)   # BINARY CLASSIFICATION

    def forward(self, x):
        x = self.net(x)
        x = x.squeeze(-1)
        return self.fc(x)

# ============================
# DATASET
# ============================
class FastECGDataset(Dataset):
    def __init__(self, df, npy_dir="ecg_npy"):
        self.df = df
        self.npy_dir = npy_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        signal = np.load(f"{self.npy_dir}/{row['ecg_id']}.npy")
        signal = torch.tensor(signal, dtype=torch.float32)

        label = 1 if row["binary_target"] == "Risk" else 0
        return signal, torch.tensor(label, dtype=torch.long)

# ============================
# RETRAIN ONLY
# ============================
if __name__ == "__main__":

    print("Loading binary labels (fast)...")

    df = pd.read_csv(BASE_DIR + "ptbxl_database.csv")
    df["scp_codes"] = df["scp_codes"].apply(literal_eval)

    risk_classes = ["MI", "STTC", "ISC"]

    def is_risk(code_dict):
        for key in code_dict.keys():
            if key in risk_classes:
                return True
        return False

    df["binary_target"] = df["scp_codes"].apply(lambda x: "Risk" if is_risk(x) else "Normal")

    df = df[["ecg_id", "filename_hr", "binary_target"]]

    # Train/Test split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Dataset and Loader
    train_ds = FastECGDataset(train_df, NPPY_DIR)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # Setup model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = FastECGModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # ============================
    # RETRAIN (FAST)
    # ============================
    print("\nStarting fast retraining to regenerate weights...\n")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")

        for signals, labels in tqdm(train_loader):
            signals, labels = signals.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print("Epoch Loss:", total_loss / len(train_loader))

    # ============================
    # SAVE MODEL
    # ============================
    torch.save(model.state_dict(), "ecg_model_binary.pth")
    print("\nDONE! Saved model as ecg_model_binary.pth")
