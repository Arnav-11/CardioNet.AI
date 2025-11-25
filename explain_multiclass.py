import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ====== same config ======
NPPY_DIR = "ecg_npy"

# ====== same model as before ======
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

# ====== Grad-CAM for 1D Conv ======
class GradCAM1D:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None

        # hook into the last Conv1d layer: index 6 in self.net
        target_layer = self.model.net[6]
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()          # (B, C, T')

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()    # (B, C, T')

    def generate(self, signal_tensor, class_idx=None):
        """
        signal_tensor: (12, T)
        class_idx: optional target class (0-4). If None → argmax.
        """
        self.model.zero_grad()
        x = signal_tensor.unsqueeze(0)           # (1, 12, T)
        out = self.model(x)                      # (1, 5)

        if class_idx is None:
            class_idx = out.argmax(1).item()

        target = out[0, class_idx]
        target.backward()

        # gradients: (1, C, T'), activations: (1, C, T')
        grads = self.gradients.mean(dim=2, keepdim=True)  # (1, C, 1)
        cam = (grads * self.activations).sum(dim=1).squeeze(0)  # (T')

        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        # upsample CAM to original length using interpolation
        T_cam = cam.shape[0]
        T_signal = signal_tensor.shape[1]
        x_cam = np.linspace(0, 1, T_cam)
        x_sig = np.linspace(0, 1, T_signal)
        cam_upsampled = np.interp(x_sig, x_cam, cam)

        return cam_upsampled, class_idx

if __name__ == "__main__":
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FastECGModel().to(device)
    model.load_state_dict(torch.load("ecg_model_multiclass.pth", map_location=device))
    model.eval()

    cam_gen = GradCAM1D(model)

    # Pick an ECG id you know exists (from df.head())
    ecg_id = int(input("Enter ECG ID (e.g. 1, 10, 100): "))

    signal = np.load(f"{NPPY_DIR}/{ecg_id}.npy")   # (12, T)
    lead1 = signal[0]                              # first lead
    signal_t = torch.tensor(signal, dtype=torch.float32).to(device)

    cam, cls_idx = cam_gen.generate(signal_t)

    classes = ["NORM", "MI", "HYP", "STTC", "CD"]
    print("Predicted class:", classes[cls_idx])

    plt.figure(figsize=(12, 4))
    plt.plot(lead1, label="ECG Lead 1")
    plt.plot(cam * lead1.max(), label="Grad-CAM (scaled)", alpha=0.7)
    plt.legend()
    plt.title(f"ECG ID {ecg_id} - GradCAM highlight ({classes[cls_idx]})")
    plt.xlabel("Time (samples)")
    plt.tight_layout()
    plt.show()
