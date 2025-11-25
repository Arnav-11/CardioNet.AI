import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from ast import literal_eval
from tqdm import tqdm

# =================================
# CONFIG
# =================================
BASE_DIR = "/Users/arnavbhatnagar/Downloads/ECG-PTBXL-Capstone-main/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/"
NPPY_DIR = "ecg_npy"
EPOCHS = 5        # quick retrain
BATCH_SIZE = 16
SEQ_LEN = 2000

# =================================
# MULTICLASS LABEL MAPPING
# =================================
diagnostic_map = {
    "NORM": 0,
    "MI": 1,
    "HYP": 2,
    "STTC": 3,
    "CD": 4
}

# =================================
# MODEL (MULTI-CLASS)
# =================================
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
        self.fc = nn.Linear(128, 5)  # MULTI-CLASS

    def forward(self, x):
        x = self.net(x)
        x = x.squeeze(-1)
        return self.fc(x)

# =================================
# DATASET
# =================================
class FastECGDataset(Dataset):
    def __init__(self, df, npy_dir=NPPY_DIR):
        self.df = df
        self.npy_dir = npy_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        signal = np.load(f"{self.npy_dir}/{row['ecg_id']}.npy")
        signal = torch.tensor(signal, dtype=torch.float32)
        return signal, torch.tensor(row["label"], dtype=torch.long)

# =================================
# TRAIN MULTI-CLASS MODEL
# =================================
if __name__ == "__main__":
    print("Loading PTB-XL metadata...")

    df = pd.read_csv(BASE_DIR + "ptbxl_database.csv")
    scp_df = pd.read_csv(BASE_DIR + "scp_statements.csv", index_col=0)

    df["scp_codes"] = df["scp_codes"].apply(literal_eval)

    # Map diagnostic classes
    def map_class(codes):
        classes = []
        for key in codes.keys():
            if key in scp_df.index:
                diag = scp_df.loc[key, "diagnostic_class"]
                if diag in diagnostic_map:
                    classes.append(diag)
        if len(classes) == 0:
            return "NORM"
        return classes[0]  # choose first

    df["main_class"] = df["scp_codes"].apply(map_class)
    df["label"] = df["main_class"].map(diagnostic_map)

    df = df[["ecg_id", "label"]]

    print(df["label"].value_counts())

    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_ds = FastECGDataset(train_df, NPPY_DIR)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = FastECGModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # Training
    for epoch in range(EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")
        model.train()
        total_loss = 0

        for signals, labels in tqdm(train_loader):
            signals, labels = signals.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print("Epoch Loss:", total_loss / len(train_loader))

    # SAVE MODEL
    torch.save(model.state_dict(), "ecg_model_multiclass.pth")
    print("\nModel saved as ecg_model_multiclass.pth")
