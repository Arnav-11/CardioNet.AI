# 🫀 CardioNet AI  
### Deep Learning–based ECG Abnormality Classification with Explainability

CardioNet AI is a deep learning system that analyzes **12-lead ECG signals** and classifies them into **five cardiac conditions** using a **1D-CNN model** trained on the **PTB-XL dataset**.  
The project also includes **Grad-CAM–based explainability** and a **real-time Streamlit dashboard** for ECG upload, prediction, and visualization.

---

##Features
- ✔ **1D-CNN model** for multi-class ECG classification  
- ✔ Achieves **80% accuracy** across **5 cardiac conditions**  
- ✔ **Grad-CAM heatmaps** to visualize important ECG regions  
- ✔ **21,000+ ECG signals** processed and normalized  
- ✔ **Real-time Streamlit dashboard** for prediction & visualization  
- ✔ End-to-end ML pipeline: preprocessing → training → explainability → deployment  

---

##Cardiac Conditions Classified
1. **NORM** – Normal  
2. **MI** – Myocardial Infarction  
3. **HYP** – Hypertrophy  
4. **STTC** – ST/T Wave Changes  
5. **CD** – Conduction Disturbance  

---

##Model Architecture (1D-CNN)

Input (12 × 2000)
│
├── Conv1D (64 filters, kernel=7)
├── Conv1D (128 filters, kernel=5)
├── Conv1D (128 filters, kernel=3)
│
└── Dense → Softmax (5 classes)

Optimizer: **AdamW**  
Loss: **CrossEntropy**  
Train/Test Split: **80/20**

---

##Explainability – Grad-CAM

CardioNet AI uses **Grad-CAM** to highlight which ECG regions influence the model’s decision.

##Streamlit Dashboard

A simple and interactive UI to:

- Upload ECG signals  
- Run inference in real-time  
- View predicted cardiac class  
- Visualize Grad-CAM heatmaps  




