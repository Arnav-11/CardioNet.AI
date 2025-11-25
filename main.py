import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from ast import literal_eval

# ========================================
# PAGE CONFIG (MUST BE FIRST)
# ========================================
st.set_page_config(
    page_title="ECG Diagnostic AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# CUSTOM CSS FOR BEAUTIFUL UI
# ========================================
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin: 10px 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 10px 30px;
        font-size: 16px;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }
    h1, h2, h3 {
        color: #2d3748;
    }
</style>
""", unsafe_allow_html=True)

# ========================================
# CONFIG
# ========================================
BASE_DIR = "/Users/arnavbhatnagar/Downloads/ECG-PTBXL-Capstone-main/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/"
NPPY_DIR = "ecg_npy"

diagnostic_map = {
    "NORM": 0,
    "MI": 1,
    "HYP": 2,
    "STTC": 3,
    "CD": 4
}

idx_to_class = {v: k for k, v in diagnostic_map.items()}

class_descriptions = {
    "NORM": "Normal ECG - No abnormalities detected",
    "MI": "Myocardial Infarction - Heart attack indicators",
    "HYP": "Hypertrophy - Enlarged heart muscle",
    "STTC": "ST/T Changes - Abnormal heart rhythm",
    "CD": "Conduction Disturbance - Electrical signal issues"
}

class_colors = {
    "NORM": "#10b981",
    "MI": "#ef4444",
    "HYP": "#f59e0b",
    "STTC": "#8b5cf6",
    "CD": "#3b82f6"
}

# ========================================
# MODEL DEFINITION
# ========================================
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

# ========================================
# GRAD-CAM FOR EXPLAINABILITY
# ========================================
class GradCAM1D:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer = self.model.net[6]
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, signal_tensor, class_idx=None):
        self.model.train()  # Enable gradient computation
        self.model.zero_grad()
        x = signal_tensor.unsqueeze(0)
        out = self.model(x)
        
        if class_idx is None:
            class_idx = out.argmax(1).item()
        
        target = out[0, class_idx]
        target.backward()
        
        self.model.eval()  # Switch back to eval mode
        
        grads = self.gradients.mean(dim=2, keepdim=True)
        cam = (grads * self.activations).sum(dim=1).squeeze(0)
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        T_cam = cam.shape[0]
        T_signal = signal_tensor.shape[1]
        x_cam = np.linspace(0, 1, T_cam)
        x_sig = np.linspace(0, 1, T_signal)
        cam_upsampled = np.interp(x_sig, x_cam, cam)
        
        return cam_upsampled, class_idx

# ========================================
# LOAD MODEL (CACHED)
# ========================================
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FastECGModel().to(device)
    model.load_state_dict(torch.load("ecg_model_multiclass.pth", map_location=device, weights_only=True))
    model.eval()
    return model, device

# ========================================
# LOAD METADATA (CACHED)
# ========================================
@st.cache_data
def load_metadata():
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
    
    return df

# ========================================
# PREDICTION FUNCTION
# ========================================
def predict_ecg(model, signal, device):
    signal_t = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(signal_t)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_class = outputs.argmax(1).item()
    return pred_class, probs

# ========================================
# MAIN APP
# ========================================
def main():
    # Header
    st.markdown("<h1 style='text-align: center; color: #2d3748;'>🫀 ECG Diagnostic AI Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px; color: #4a5568;'>Powered by Deep Learning • PTB-XL Dataset</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Load model and data
    model, device = load_model()
    df = load_metadata()
    
    # Sidebar
    st.sidebar.title("🔍 ECG Selection")
    st.sidebar.markdown("---")
    
    # Filter options
    diagnosis_filter = st.sidebar.multiselect(
        "Filter by Diagnosis",
        options=list(diagnostic_map.keys()),
        default=list(diagnostic_map.keys())
    )
    
    filtered_df = df[df["main_class"].isin(diagnosis_filter)]
    
    st.sidebar.markdown(f"**{len(filtered_df)}** ECGs available")
    
    # ECG selection with session state for random button
    if 'random_ecg' not in st.session_state:
        st.session_state['random_ecg'] = None
    
    # Random selection button
    if st.sidebar.button("🎲 Random ECG"):
        st.session_state['random_ecg'] = np.random.choice(filtered_df["ecg_id"].values)
    
    # Determine default index
    if st.session_state['random_ecg'] is not None and st.session_state['random_ecg'] in filtered_df["ecg_id"].values:
        default_idx = list(filtered_df["ecg_id"].values).index(st.session_state['random_ecg'])
    else:
        default_idx = 0
    
    ecg_id = st.sidebar.selectbox(
        "Select ECG ID",
        options=filtered_df["ecg_id"].values,
        format_func=lambda x: f"ECG {x}",
        index=default_idx
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Dataset Stats")
    stats_df = df["main_class"].value_counts().reset_index()
    stats_df.columns = ["Diagnosis", "Count"]
    st.sidebar.dataframe(stats_df, use_container_width=True)
    
    # Main content
    if ecg_id:
        # Load signal
        signal_path = f"{NPPY_DIR}/{ecg_id}.npy"
        if not os.path.exists(signal_path):
            st.error(f"❌ ECG file not found: {signal_path}")
            return
        
        signal = np.load(signal_path)
        actual_class = df[df["ecg_id"] == ecg_id]["main_class"].values[0]
        
        # Make prediction
        pred_class, probs = predict_ecg(model, signal, device)
        pred_label = idx_to_class[pred_class]
        
        # Top section: Prediction
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown(f"""
            <div class='prediction-box'>
                <div style='font-size: 16px; margin-bottom: 10px;'>PREDICTED DIAGNOSIS</div>
                <div style='font-size: 36px; margin: 10px 0;'>{pred_label}</div>
                <div style='font-size: 14px; opacity: 0.9;'>{class_descriptions[pred_label]}</div>
                <div style='font-size: 18px; margin-top: 15px;'>Confidence: {probs[pred_class]*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Metrics row
        st.markdown("### 📈 Prediction Confidence")
        cols = st.columns(5)
        for i, (label, prob) in enumerate(zip(idx_to_class.values(), probs)):
            with cols[i]:
                delta = "✓ Predicted" if i == pred_class else ""
                st.metric(
                    label=label,
                    value=f"{prob*100:.1f}%",
                    delta=delta
                )
        
        st.markdown("---")
        
        # Confidence Bar Chart
        st.markdown("### 🎯 Confidence Distribution")
        fig_bar = go.Figure()
        colors = [class_colors[idx_to_class[i]] for i in range(5)]
        
        fig_bar.add_trace(go.Bar(
            x=list(idx_to_class.values()),
            y=probs * 100,
            marker_color=colors,
            text=[f"{p*100:.1f}%" for p in probs],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Confidence: %{y:.1f}%<extra></extra>'
        ))
        
        fig_bar.update_layout(
            title="Prediction Confidence by Class",
            xaxis_title="Diagnosis",
            yaxis_title="Confidence (%)",
            yaxis_range=[0, 100],
            height=400,
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
        
        st.markdown("---")
        
        # ECG Visualization
        st.markdown("### 📉 12-Lead ECG Visualization")
        
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        
        fig = make_subplots(
            rows=4, cols=3,
            subplot_titles=lead_names,
            vertical_spacing=0.08,
            horizontal_spacing=0.05
        )
        
        for i in range(12):
            row = i // 3 + 1
            col = i % 3 + 1
            
            fig.add_trace(
                go.Scatter(
                    y=signal[i],
                    mode='lines',
                    line=dict(color='#667eea', width=1.5),
                    name=lead_names[i],
                    showlegend=False,
                    hovertemplate=f'<b>Lead {lead_names[i]}</b><br>Time: %{{x}}<br>Amplitude: %{{y:.3f}}<extra></extra>'
                ),
                row=row, col=col
            )
        
        fig.update_xaxes(showticklabels=False, showgrid=False)
        fig.update_yaxes(showticklabels=False, showgrid=False)
        
        fig.update_layout(
            height=800,
            title_text=f"ECG ID: {ecg_id} | Actual: {actual_class} | Predicted: {pred_label}",
            hovermode='closest',
            plot_bgcolor='white',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Grad-CAM Explainability
        st.markdown("### 🔬 AI Explainability (Grad-CAM)")
        st.markdown("*Highlighted regions show which parts of the ECG influenced the AI's decision*")
        
        cam_gen = GradCAM1D(model)
        signal_t = torch.tensor(signal, dtype=torch.float32).to(device)
        cam, _ = cam_gen.generate(signal_t, pred_class)
        
        # Select lead for Grad-CAM
        selected_lead = st.selectbox("Select Lead for Grad-CAM Analysis", lead_names, index=1)
        lead_idx = lead_names.index(selected_lead)
        
        fig_cam = go.Figure()
        
        # Original signal
        fig_cam.add_trace(go.Scatter(
            y=signal[lead_idx],
            mode='lines',
            name='ECG Signal',
            line=dict(color='#2d3748', width=2),
            hovertemplate='<b>ECG Signal</b><br>Time: %{x}<br>Amplitude: %{y:.3f}<extra></extra>'
        ))
        
        # Grad-CAM overlay
        fig_cam.add_trace(go.Scatter(
            y=cam * signal[lead_idx].max(),
            mode='lines',
            name='AI Focus Areas',
            fill='tozeroy',
            fillcolor='rgba(239, 68, 68, 0.3)',
            line=dict(color='rgba(239, 68, 68, 0.8)', width=2),
            hovertemplate='<b>AI Attention</b><br>Time: %{x}<br>Importance: %{y:.3f}<extra></extra>'
        ))
        
        fig_cam.update_layout(
            title=f"Grad-CAM Analysis - Lead {selected_lead}",
            xaxis_title="Time (samples)",
            yaxis_title="Amplitude",
            height=400,
            hovermode='x unified',
            legend=dict(x=0.01, y=0.99),
            plot_bgcolor='white',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_cam, use_container_width=True)
        
        # Comparison Section
        st.markdown("---")
        st.markdown("### ⚖️ Prediction Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Actual Diagnosis")
            st.markdown(f"""
            <div style='background: {class_colors[actual_class]}; color: white; padding: 20px; 
                        border-radius: 10px; text-align: center; font-size: 20px;'>
                <b>{actual_class}</b><br>
                <span style='font-size: 14px;'>{class_descriptions[actual_class]}</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### Model Prediction")
            is_correct = actual_class == pred_label
            bg_color = "#10b981" if is_correct else "#ef4444"
            result_text = "✓ CORRECT" if is_correct else "✗ INCORRECT"
            
            st.markdown(f"""
            <div style='background: {bg_color}; color: white; padding: 20px; 
                        border-radius: 10px; text-align: center; font-size: 20px;'>
                <b>{pred_label}</b><br>
                <span style='font-size: 14px;'>{class_descriptions[pred_label]}</span><br>
                <span style='font-size: 16px; margin-top: 10px;'>{result_text}</span>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()