import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from ast import literal_eval
import matplotlib.pyplot as plt
import seaborn as sns

# ====== CONFIG (same as retrain_multiclass.py) ======
BASE_DIR = "/Users/arnavbhatnagar/Downloads/ECG-PTBXL-Capstone-main/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/"
NPPY_DIR = "ecg_npy"
diagnostic_map = {
    "NORM": 0,
    "MI":   1,
    "HYP":  2,
    "STTC": 3,
    "CD":   4
}
idx_to_class = {v: k for k, v in diagnostic_map.items()}

# ====== MODEL DEFINITION (same as training) ======
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
        self.fc = nn.Linear(128, 5)

    def forward(self, x):
        x = self.net(x)
        x = x.squeeze(-1)
        return self.fc(x)

# ====== DATASET (loads cached .npy) ======
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
        label = int(row["label"])
        return signal, torch.tensor(label, dtype=torch.long)

if __name__ == "__main__":
    # ====== Load metadata & labels ======
    df = pd.read_csv(BASE_DIR + "ptbxl_database.csv")
    scp_df = pd.read_csv(BASE_DIR + "scp_statements.csv", index_col=0)
    df["scp_codes"] = df["scp_codes"].apply(literal_eval)

    def map_class(codes):
        classes = []
        for key in codes.keys():
            if key in scp_df.index:
                diag = scp_df.loc[key, "diagnostic_class"]
                if diag in diagnostic_map:
                    classes.append(diag)
        if len(classes) == 0:
            return "NORM"
        return classes[0]

    df["main_class"] = df["scp_codes"].apply(map_class)
    df["label"] = df["main_class"].map(diagnostic_map)
    df = df[["ecg_id", "label"]]

    # ====== Split and dataloader ======
    _, test_df = train_test_split(df, test_size=0.2, random_state=42)
    test_ds = FastECGDataset(test_df, NPPY_DIR)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    # ====== Load trained model ======
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FastECGModel().to(device)
    model.load_state_dict(torch.load("ecg_model_multiclass.pth", map_location=device))
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for signals, labels in test_loader:
            signals, labels = signals.to(device), labels.to(device)
            outputs = model(signals)
            preds = outputs.argmax(1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # ====== Metrics ======
    print("Classification Report:")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=["NORM", "MI", "HYP", "STTC", "CD"]
    ))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["NORM", "MI", "HYP", "STTC", "CD"],
        yticklabels=["NORM", "MI", "HYP", "STTC", "CD"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
