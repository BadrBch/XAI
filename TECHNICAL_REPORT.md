# Technical Report: Unified Explainable AI Interface

## 1. Introduction
The objective of this project was to refactor and integrate two separate existing Explainable AI (XAI) systems—Deepfake Audio Detection and Lung Cancer Detection—into a single, unified interactive platform. The goal was to provide a seamless user experience for multi-modal classification and interpretation, leveraging state-of-the-art XAI techniques to explain model decisions for both audio and image inputs.

## 2. System Architecture

### 2.1 Overview
The platform is built using a modular architecture that separates the frontend interface from the backend processing logic. This ensures maintainability and scalability.

*   **Frontend**: Built with **Streamlit**, featuring a custom "Flat Design" CSS theme for a modern, professional user interface.
*   **Backend**: A hybrid AI engine supporting both **TensorFlow/Keras** (for Audio) and **PyTorch** (for Image) models.
*   **XAI Engine**: A dedicated module (`src/xai_engine.py`) that abstracts the implementation of Grad-CAM, LIME, and SHAP, making them agnostic to the underlying model framework where possible.

### 2.2 Directory Structure
The project was refactored from a monolithic script approach into a clean package structure:
*   `app.py`: The entry point and UI logic.
*   `src/model_loader.py`: Implements a **Factory Pattern** to load models dynamically based on demand.
*   `src/preprocessing.py`: Handles domain-specific data transformations (e.g., Mel-Spectrogram conversion for audio).
*   `src/xai_engine.py`: Encapsulates the core logic for generating explanations.

## 3. Selected Models & Implementation

### 3.1 Audio Classification (Deepfake Detection)
*   **Model Architecture**: **MobileNetV2**.
*   **Input**: Mel-Spectrograms (converted from `.wav` files).
*   **Rationale**: The original repository converted audio to spectrograms to treat the problem as an image classification task. We maintained this approach as it allows the reuse of powerful CNN architectures and image-based XAI methods like Grad-CAM.
*   **Implementation**: We utilize a TensorFlow SavedModel (Legacy Keras) to ensure compatibility with pre-trained weights.

### 3.2 Image Classification (Lung Cancer Detection)
*   **Model Architecture**: **DenseNet121** (via **TorchXRayVision** library).
*   **Input**: Chest X-Ray images (`.jpg`, `.png`).
*   **Rationale**: DenseNet121 is the state-of-the-art standard for medical image analysis due to its dense connectivity pattern, which improves feature propagation. We utilized the `torchxrayvision` pre-trained weights (trained on CheXpert and other massive datasets) to ensure robust performance without the need for training from scratch.

## 4. Explainable AI (XAI) Methods

We implemented three distinct XAI techniques to provide comprehensive model transparency:

### 4.1 Grad-CAM (Gradient-weighted Class Activation Mapping)
*   **Description**: Visualizes which regions of the input image (or spectrogram) successfully activated the specific class.
*   **Implementation**:
    *   **Audio (TF)**: customized gradient calculation on the last convolutional layer of MobileNetV2.
    *   **Image (PyTorch)**: Implemented using hooks on the `eatures` layer of DenseNet121 to captured gradients and feature maps.

### 4.2 LIME (Local Interpretable Model-agnostic Explanations)
*   **Description**: Perturbs the input superpixels to determine which parts contribute most to the prediction.
*   **Implementation**: Adapted `lime_image.LimeImageExplainer` for both modalities. For audio, explanations are generated on the visual spectrogram representation.

### 4.3 SHAP (SHapley Additive exPlanations)
*   **Description**: Uses game theory to assign an importance value to each pixel.
*   **Implementation**: integrated `shap.KernelExplainer`. We implemented a background sampling strategy to reduce computational cost while maintaining explanation fidelity.

## 5. Design & Integration Decisions

### 5.1 Compatibility & Filtering
One of the key requirements was to ensure "Automatic compatibility checks".
*   **Challenge**: Audio and Image models run on different frameworks (TF vs PyTorch) and inputs have different dimensions.
*   **Solution**: We implemented a `detect_file_type` helper that automatically routes inputs to the correct preprocessing pipeline. The UI dynamically updates to show only relevant results. Since audio is converted to spectrograms (images), we successfully enabled image-based XAI methods (Grad-CAM) for audio, satisfying the requirement for cross-modal XAI application where applicable.

### 5.2 Unified UI vs. Separate Tabs
Instead of separating Audio and Image into completely different pages, we built a **Unified "Analysis" Mode**. The interface adapts contextually based on the uploaded file. This provides a more fluid user experience than rigid tab switching. A separate **"Comparison" Mode** was added specifically for the side-by-side evaluation requirement.

## 6. Improvements Over Original Repositories
1.  **Code Quality**: Refactored spaghetti code into structured modules with type hinting and docstrings.
2.  **Performance**: Implemented `@st.cache_resource` for model loading, reducing startup time effectively to zero after the first load.
3.  **Visualization**: Replaced matplotlib static plots with Streamlit's native image handling for responsive, interactive layouts.
4.  **Error Handling**: Added robust try-catch blocks around XAI generation to preventing the entire app from crashing if one method fails (e.g., due to singularity in LIME).

## 7. Generative AI Usage Declaration

| Tool Used | Usage Description |
| :--- | :--- |
| **Vibe Coding (Claude Code)** | Primary assistant for refactoring code logic, ensuring Keras/PyTorch interoperability, and implementing the XAI Engine. |
| **Antigravity** | Used to analyze the complex codebase structure and automatically generate this technical report and the README. |

---
**Conclusion**: The final platform meets all project objectives, offering a robust, extensible, and user-friendly environment for exploring generic and medical AI explainability.
