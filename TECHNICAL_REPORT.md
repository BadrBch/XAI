# Technical Report: Unified XAI Platform for Multi-Modal Pathology Detection
****Course****: M2 Data Science - 2025
****Subject****: Explainable AI & Deep Learning Integration

## 1. Executive Summary

This project addresses the engineering challenge of unifying two distinct deep learning workflows, audio-based Deepfake detection and image-based lung pathology diagnosis, into a coherent, interactive framework. The primary objective was not merely to implement classification models, but to bridge the interpretability gap using Explainable AI (XAI). We successfully integrated a TensorFlow/Keras-based audio pipeline with a PyTorch-based medical imaging pipeline, unifying them under a single Streamlit dashboard. This report details the architectural decisions, model selection process, specifically the migration to domain-specific architectures for X-Ray analysis, and the implementation of three XAI methods: Grad-CAM, LIME, and SHAP.

## 2. System Architecture & Engineering

### 2.1 Hybrid Backend Design
A significant technical constraint was the need to support heterogeneous deep learning frameworks simultaneously.
****Audio Subsystem****: Relies on tensorflow (legacy Keras, because the newer was non compatible with our environnement) for the MobileNetV2 architecture.
****Image Subsystem****: Utilizes torch and torchxrayvision for the DenseNet121 architecture.

To manage this, we implemented a ****Factory Pattern**** in src/model_loader.py. This module abstracts the initialization logic, loading models into memory only upon first request (Lazy Loading) and caching them via @st.cache_resource to minimize inference latency.

### 2.2 Modular XAI Engine
We developed a framework-agnostic XAI wrapper (src/xai_engine.py). This module standardizes the input/output interface for explanation generation, handling the tensor transformations required to switch between NumPy arrays (for LIME/SHAP) and framework-specific tensors (for Grad-CAM).

## 3. Model Selection and Methodology

### 3.1 Audio Classification (Deepfake Detection)
We adopted a transfer learning approach using ****MobileNetV2****. The reason is its the model with the best results in the deepfake repo.
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

### 3.3 Data Sources

To ensure the validity of our experiments, we utilized distinct, high-quality sources for our datasets:

*   **X-Ray Pathology Data**:
    *   **Pathological Samples**: Sourced from [Radiology Masterclass - Lung Cancer](https://www.radiologymasterclass.co.uk/gallery/chest/lung_cancer/mass_consolidation). These images provide clear examples of mass consolidations essential for validating our "Mass" detection capabilities.
    *   **Normal Controls**: Sourced from [Radiology Masterclass - Normal Chest X-Ray](https://www.radiologymasterclass.co.uk/gallery/chest/quality/normal-chest-x-ray-male) for establishing a baseline for healthy lung tissue.

*   **Audio Deepfake Data**:
    *   **Real Audio Samples**: Obtained from the Mendeley Data repository [Deep Fake Detection Dataset](https://data.mendeley.com/datasets/5czyx2vppv/2).
    *   **Fake Audio Generation**: Synthetic voice samples were generated using a high-quality, free online Text-to-Speech (TTS) service to simulate accessible deepfake generation vectors.

## 4. Explainable AI Implementation

We integrated three complementary methods to validate model decision-making.

### 4.1 Grad-CAM (Gradient-weighted Class Activation Mapping)
This method was essential for visualizing the "where" of the prediction.
*   **Audio**: We compute gradients of the target class with respect to the last convolutional block of MobileNetV2.
*   **Image**: We registered forward and backward hooks on the `features` layer of the DenseNet121. This was necessary because `torchxrayvision` models optimize the feature map flow differently than standard torchvision models.

### 4.2 LIME (Local Interpretable Model-agnostic Explanations)
LIME was implemented using the `lime_image` module.
*   **Segmentation Strategy**: We utilized Quickshift segmentation to define superpixels.
*   **Adaptation**: For audio, LIME is applied to the spectrogram image. This provides an intuitive check: if LIME highlights background noise rather than the voice formants, we know the model is overfitting to silence/artifacts.

### 4.3 SHAP (SHapley Additive exPlanations)
We employed `shap.KernelExplainer` for its model-agnostic properties.
*   **Optimization**: Due to the high computational cost of Shapley value estimation, we implemented a background summarization strategy (using K-means clustering on a subset of data) to serve as the reference distribution. This reduced explanation generation time from minutes to seconds without a significant loss in fidelity.

## 5. Conclusion

The final platform demonstrates the viability of unifying disparate Deep Learning frameworks into a user-centric application. The transition to specialized medical models (`torchxrayvision`) was the decisive factor in achieving credible results for the lung pathology module. By combining this with a robust XAI suite, the tool serves not just as a classifier, but as an auditing mechanism for "black box" models.
