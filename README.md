# Unified XAI Interface: Deepfake & Pathology Detection
**M2-DS Final Project - 2025**
**DIA 2**

## Team Members
* Badr Bouchabchoub
* Seifeddine Ben Rhouma
* Eric Bernhart
* Serdar Charyyev
* Noé Bourdin

---

## 1. Abstract
The primary objective of this project is to unify two distinct classification workflows into a single, cohesive interactive platform. By integrating state-of-the-art architectures for **Audio Deepfake Detection** and **Lung Cancer Diagnosis**, we aim to provide not just accurate predictions, but also robust interpretability. The application leverages a suite of Explainable AI (XAI) techniques—Grad-CAM, LIME, and SHAP—bridging the gap between "black box" model performance and user trust.

### Technical Scope & Features

**Multi-Modal Classification Engines:**
*   **Audio Pipeline**: Detection of synthetic voice manipulation (Deepfakes) using **MobileNetV2** trained on Mel-Spectrogram features.
*   **Image Pipeline**: Pathological screening for lung conditions using a transfer-learning approach with **DenseNet121** (TorchXRayVision).

**Explainability Module:**
We implemented three complementary interpretation layers to validate model behaviors:
*   **Grad-CAM**: Generates class-discriminative localization maps (Heatmaps) for Convolutional Neural Networks.
*   **LIME**: Provides local, interpretable approximations of specific predictions.
*   **SHAP**: Attribues feature importance values based on cooperative game theory concepts.

**Platform Capabilities:**
*   **Dynamic UI**: A Streamlit-based dashboard designed for ease of use and real-time inference.
*   **Context-Aware Filtering**: The system automatically restricts XAI method selection based on the input modality (e.g., enabling Grad-CAM only for compatible image tensors).
*   **Comparative Analysis**: A dedicated view to juxtapose results from LIME, Grad-CAM, and SHAP simultaneously, allowing for cross-verification of explanatory evidence.

---

## 2. Setup & Installation

### Requirements
*   **Environment**: Python 3.9+
*   **Version Control**: Git

### Deployment

**1. Clone the repository**
```bash
git clone <repository-url>
cd XAI
```

**2. Environment Configuration**
It is strictly recommended to use a virtual environment to avoid dependency conflicts, particularly with TensorFlow/PyTorch versioning.
```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
.\venv\Scripts\activate
```

**3. Dependency Installation**
```bash
pip install -r requirements.txt
```

---

## 3. Usage Guide

### Launching the Dashboard
Execute the wrapper script from the project root:

```bash
source venv/bin/activate
streamlit run app.py
```
*The interface typically binds to `http://localhost:8501`.*

### Workflow
1.  **Inference Mode**: Upload raw data (Audio `.wav` or Medical Imaging `.jpg/.png`). The backend logic utilizes a factory pattern to route the file to the appropriate pre-processing and model pipeline.
2.  **XAI Selection**: Choose an interpretation algorithm. The system will render the decision boundary (Real/Fake or Pathology) alongside the visual explanation.
3.  **Cross-Method Validation**: Use the "Comparison Mode" to audit a sample against all three XAI techniques. This is particularly useful for analyzing edge cases where models might diverge in their reasoning.
### Test Data
To facilitate testing, we have provided sample datasets within the repository:
*   `Audio_Test/`: Contains sample `.wav` files (Real vs. Fake) to test the deepfake detection module.
*   `Images_Test/`: Contains sample Chest X-Rays `.jpg/.png` to test the pathology detection module.

---

## 4. Generative AI Usage

This project was developed using a "Vibe Coding" methodology, utilizing advanced AI assistants, in our exemple **Claude Code** and autonomous agents to ensure code correctness and efficiency. The development process remained strictly **human-in-the-loop**, ensuring the final product aligns perfectly with our original ideas and architectural vision. The AI acted as a powerful accelerator while the human developer maintained full creative and technical control.
    
    

---

## 5. Project Architecture

```
Project_XAI/
├── app.py                  # Streamlit entry point & detailed dashboard logic
├── requirements.txt        # Frozen dependencies
├── README.md               # Main documentation
├── TECHNICAL_REPORT.md     # Detailed methodology 
├── src/
│   ├── model_loader.py     # Factory class for model instantiation
│   ├── preprocessing.py    # Signal processing (Audio) & Tensor transforms (Image)
│   └── xai_engine.py       # Wrapper classes for Grad-CAM, LIME, and SHAP
└── models/                 # Pre-trained weights directory
```
