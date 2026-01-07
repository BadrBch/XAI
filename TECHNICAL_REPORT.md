# Technical Report: Unified XAI Platform for Multi-Modal Pathology Detection
****Course****: M2 Data Science - 2025
****Subject****: Explainable AI & Deep Learning Integration

## 1. Executive Summary

This project addresses the engineering challenge of unifying two distinct deep learning workflows, audio-based Deepfake detection and image-based lung pathology diagnosis, into a coherent, interactive framework. The primary objective was not merely to implement classification models, but to bridge the interpretability gap using Explainable AI (XAI). We successfully integrated a TensorFlow/Keras-based audio pipeline with a PyTorch-based medical imaging pipeline, unifying them under a single Streamlit dashboard. This report details the architectural decisions, model selection process, specifically the migration to domain-specific architectures for X-Ray analysis, and the implementation of three XAI methods: Grad-CAM, LIME, and SHAP.

## 2. System Architecture & Engineering

### 2.1 Hybrid Backend Design
A significant technical constraint was the need to support heterogeneous deep learning frameworks simultaneously.
****Audio Subsystem****: Relies on tensorflow (legacy Keras) for the MobileNetV2 architecture.
****Image Subsystem****: Utilizes torch and torchxrayvision for the DenseNet121 architecture.

To manage this, we implemented a ****Factory Pattern**** in src/model_loader.py. This module abstracts the initialization logic, loading models into memory only upon first request (Lazy Loading) and caching them via @st.cache_resource to minimize inference latency.

### 2.2 Modular XAI Engine
We developed a framework-agnostic XAI wrapper (src/xai_engine.py). This module standardizes the input/output interface for explanation generation, handling the tensor transformations required to switch between NumPy arrays (for LIME/SHAP) and framework-specific tensors (for Grad-CAM).

## 3. Model Selection and Methodology

### 3.1 Audio Classification (Deepfake Detection)
We adopted a transfer learning approach using ****MobileNetV2****.
****Preprocessing****: Raw .wav files are converted into Mel-Spectrograms. This transformation allows us to treat audio classification as a computer vision problem, leveraging the spatial feature extraction capabilities of CNNs.
****Performance****: The lightweight nature of MobileNetV2 ensures low-latency inference, crucial for the interactive dashboard.

### 3.2 Image Classification (Lung Cancer/Pathology)
For the medical imaging component, our trusted methodology evolved significantly during the development phase.

Initial Approach & Limitations:
Initially, we experimented with general-purpose architectures such as ****VGG16**** and ****ResNet50****, pre-trained on ImageNet. While these models performed adequately on standard object recognition, they struggled to generalize to the grayscale, texture-heavy domain of chest X-Rays. We observed high false-positive rates and, more critically, the Grad-CAM heatmaps frequently focused on irrelevant artifacts (e.g., clavicles or image borders) rather than lung opacities.

Final Implementation: TorchXRayVision:
To address these deficiencies, we migrated to the ****DenseNet121**** architecture provided by the torchxrayvision library.
****Weights****: densenet121-res224-all
****Justification****: This library is known for its effectiveness in detecting lung anomalies. It provided a powerful pre-trained model capable of detecting ****18 different pathologies**** simultaneously (including Effusion, Pneumonia, and Infiltration).
****Strategic Advantage****: We leveraged this multi-label capability to enhance the user experience. By default, we configured the system to prioritize ****"Mass"**** detection, as it is a critical indicator for lung cancer, while still allowing the user to filter for any of the other 17 conditions. This design choice focuses the initial analysis on the most relevant pathology for our use case while maintaining broadened diagnostic utility.
****Result****: This domain-specific pre-training yielded superior feature extraction. The Grad-CAM outputs became significantly more clinically relevant, correctly localizing pathological regions within the lung fields.

## 4. Explainable AI Implementation

We integrated three complementary methods to validate model decision-making.

### 4.1 Grad-CAM (Gradient-weighted Class Activation Mapping)
This method was essential for visualizing the "where" of the prediction.
****Audio****: We compute gradients of the target class with respect to the last convolutional block of MobileNetV2.
