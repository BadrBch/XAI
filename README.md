> **M2-DS Final Project - 2025**
DIA 2 

## Team Members

* Badr Bouchabchoub
* Seifeddine Ben Rhouma
* Eric Bernhart
* Serdar Charyyev
* Noé Bourdin

---

## Project Overview

This project integrates two distinct Explainable AI (XAI) systems into a single, unified interactive platform. It provides a seamless interface for processing multi-modal data (Audio and Images) to detect Deepfakes and Lung Cancer, while offering powerful interpretability through state-of-the-art XAI techniques.

### Key Features

* **Multi-Modal Support**:
* **Audio**: Deepfake detection using Mel-Spectrogram analysis with MobileNetV2.
* **Image**: Lung cancer pathology detection using TorchXRayVision (DenseNet121).


* **Unified Interface**: A modern, flat-design Streamlit dashboard for easy interaction.
* **Explainable AI (XAI)**:
* **Grad-CAM**: Visual heatmaps highlighting important regions.
* **LIME**: Local interpretable model-agnostic explanations.
* **SHAP**: Game-theoretic approach to feature importance.


* **Smart Filtering**: Automatic compatibility checks ensure only relevant XAI methods are shown for each input type.
* **Comparison Mode**: Dedicated tab to compare LIME, Grad-CAM, and SHAP results side-by-side.

---

## Setup & Installation

### Prerequisites

* Python 3.9 or higher
* Git

### Installation Steps

1. **Clone the repository**
```bash
git clone <repository-url>
cd Project_XAI

```


2. **Create a Virtual Environment** (Recommended)
```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
.\venv\Scripts\activate

```


3. **Install Dependencies**
```bash
pip install -r requirements.txt

```



---

## Usage

### Running the Application

To start the web interface, run the following command in your terminal:

```bash
# 1. Activate the virtual environment
source venv/bin/activate
# On Windows use: .\venv\Scripts\activate

# 2. Run the application
streamlit run app.py
```

The application will open automatically in your default web browser (usually at `http://localhost:8501`).

### How to Use

1. **Project Overview**: Read about the models and capabilities.
2. **Analysis Mode**:
* Upload a file (Audio `.wav` or Image `.jpg/.png`).
* The system automatically detects the file type and loads the appropriate model.
* Select an XAI method (Grad-CAM, LIME, or SHAP).
* View the prediction (Real/Fake or Pathology) and the explanation visualization.


3. **Comparison Mode**:
* Upload a file to see all three XAI methods generated simultaneously side-by-side.
* Useful for evaluating which explanation method offers the best insight for a specific sample.



---

## Generative AI Usage Statement

> **Transparency Declaration**

This project was developed with the assistance of Generative AI tools to enhance code quality, optimize refactoring, and generate documentation.

**Claude Code / Vibe Coding** | **Code Refactoring & Integration**: Used to merge the original Deepfake and Lung Cancer repositories into a modular structure (`src/` folder), ensuring clean separation of concerns.<br>

<br>**Debugging**: Assisted in resolving TensorFlow/PyTorch conflicts (GPU memory handling) and ensuring compatibility between Keras 2 and Keras 3 environments. |
| **Streamlit Assistant** | **UI Design**: Helped generate the custom CSS for the modern "Flat Design" aesthetic to improve user experience. |
| **Antigravity Agent** | **Documentation**: Assisted in structuring and writing this README and the Technical Report to ensure all project requirements were met. |

*The core logic, architectural decisions, and final validation were performed by the student team.*

---

## Project Structure

```
Project_XAI/
|-- app.py                  # Main Streamlit Dashboard application
|-- requirements.txt        # Project dependencies
|-- README.md               # Project documentation
|-- TECHNICAL_REPORT.md     # In-depth technical details
|-- src/
|   |-- model_loader.py     # Model loading logic (Factory Pattern)
|   |-- preprocessing.py    # Data preprocessing for Audio/Images
|   |-- xai_engine.py       # Implementation of Grad-CAM, LIME, SHAP
|-- models/                 # Directory for saved model weights

```