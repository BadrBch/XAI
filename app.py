"""
Unified XAI Platform - Streamlit Dashboard
A comprehensive interface for Explainable AI analysis on audio and image data.
"""

# CRITICAL: Force TensorFlow to use CPU to avoid conflict with PyTorch MPS
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU for TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Reduce TF logging

import streamlit as st
import numpy as np
import cv2
import tempfile
from PIL import Image

# Import custom modules
from src.preprocessing import preprocess_audio, preprocess_image_from_array, preprocess_image_for_xrv
from src.model_loader import load_all_models, get_model_info
from src.xai_engine import XAIEngine, is_pytorch_model

# PyTorch for device handling
import torch


# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Unified XAI Platform",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern flat design
st.markdown("""
<style>
    /* Global Reset & Typography */
    .stApp {
        background-color: #f8f9fa; /* Light grey background for better contrast */
        font-family: 'Inter', 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1a1a1a;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(120deg, #2563eb, #1e40af);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #64748b;
        margin-bottom: 2rem;
    }
    
    /* Buttons - Flat Design & Navigation */
    .stButton > button {
        border-radius: 6px; /* Slightly tighter radius */
        padding: 0.5rem 1rem; /* Smaller padding */
        font-size: 0.95rem; /* Slightly smaller text */
        font-weight: 600;
        width: 100%;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
        /* Note: Background and Color removed to respect 'type="primary"' vs 'secondary' */
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 4px -1px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }

    /* Force "Primary" (Active) Buttons to be Blue */
    button[kind="primary"] {
        background-color: #2563eb !important;
        border: 1px solid #2563eb !important;
        color: white !important;
    }
    
    button[kind="primary"]:hover {
        background-color: #1d4ed8 !important;
        border-color: #1d4ed8 !important;
    }
    
    /* Ensure Secondary (Inactive) Buttons are clean */
    button[kind="secondary"] {
        background-color: #f1f5f9;
        border: 1px solid #e2e8f0;
        color: #475569;
    }
    
    button[kind="secondary"]:hover {
        background-color: #e2e8f0;
        border-color: #cbd5e1;
        color: #1e293b;
    }
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        padding: 0.5rem 1rem;
    }
    
    /* Cards / Containers */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Sidebar Status Badges */
    .status-badge {
        display: flex;
        align-items: center;
        width: 100%;
        margin-bottom: 0.5rem;
        padding: 0.5rem 0.75rem;
        background-color: #f0fdf4; /* Light green bg */
        border: 1px solid #dcfce7;
        border-radius: 6px;
        color: #166534; /* Dark green text */
        font-size: 0.9rem;
        font-weight: 600;
    }
    
    .status-badge.error {
        background-color: #fef2f2;
        border-color: #fee2e2;
        color: #991b1b;
    }
    
    .status-icon {
        margin-right: 8px;
        font-weight: 800;
    }
    
    /* File Uploader */
    .stFileUploader {
        border: 1px dashed #cbd5e1;
        border-radius: 12px;
        padding: 1rem;
        background-color: #ffffff;
    }
    
</style>
""", unsafe_allow_html=True)


# ============================================
# LOAD MODELS (Cached)
# ============================================
@st.cache_resource
def get_models():
    return load_all_models()


# ============================================
# HELPER FUNCTIONS
# ============================================
def detect_file_type(uploaded_file) -> str:
    """Detect file type from uploaded file."""
    filename = uploaded_file.name.lower()
    if filename.endswith('.wav'):
        return 'audio'
    elif filename.endswith(('.jpg', '.jpeg', '.png')):
        return 'image'
    return 'unknown'


def get_prediction(model, preprocessed_data, file_type: str) -> tuple:
    """Get model prediction and confidence."""
    
    if file_type == 'image' and is_pytorch_model(model):
        # TorchXRayVision model - returns 18 pathology scores
        model.eval()
        with torch.no_grad():
            if isinstance(preprocessed_data, torch.Tensor):
                outputs = model(preprocessed_data)
            else:
                # Convert numpy to tensor
                device = next(model.parameters()).device
                tensor = torch.from_numpy(preprocessed_data).float().to(device)
                outputs = model(tensor)
            
            # Get all pathology scores
            scores = outputs[0].cpu().numpy()
            pathologies = model.pathologies
            
            # Create sorted list of (pathology, score) tuples
            path_scores = [(p, float(s)) for p, s in zip(pathologies, scores)]
            path_scores_sorted = sorted(path_scores, key=lambda x: x[1], reverse=True)
            
            # Determine Normal/Abnormal based on max score
            max_score = path_scores_sorted[0][1]
            if max_score > 0.5:
                class_name = 'Abnormal'
                confidence = max_score
            else:
                class_name = 'Normal'
                confidence = 1.0 - max_score
            
            return class_name, confidence, path_scores_sorted
    else:
        # TensorFlow model (audio)
        # Use direct call instead of model.predict() to avoid deadlocks
        import tensorflow as tf
        tensor_input = tf.convert_to_tensor(preprocessed_data, dtype=tf.float32)
        predictions = model(tensor_input, training=False)
        predictions = predictions.numpy()
        
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class])
        
        # Class labels for audio
        labels = {0: 'Real Audio', 1: 'Deepfake Audio'}
        class_name = labels.get(predicted_class, f'Class {predicted_class}')
        
        return class_name, confidence, predictions


def display_pathology_scores(path_scores: list, selected_pathologies: list = None) -> tuple:
    """
    Display pathology scores and prediction based on selected pathologies.
    
    Args:
        path_scores: List of (pathology_name, score) tuples, sorted by score
        selected_pathologies: List of pathology names to analyze (pre-selected by user)
    
    Returns:
        Tuple of (class_name, confidence) based on selected pathologies
    """
    scores_dict = {p: s for p, s in path_scores}
    
    # Use provided selection or default to Mass
    selected = selected_pathologies if selected_pathologies else ['Mass']
    
    # Compute prediction based on selected pathologies only
    if selected:
        selected_scores = [scores_dict[p] for p in selected if p in scores_dict]
        max_selected_score = max(selected_scores) if selected_scores else 0.0
        
        if max_selected_score > 0.5:
            class_name = 'Abnormal'
            confidence = max_selected_score
        else:
            class_name = 'Normal'
            confidence = 1.0 - max_selected_score
    else:
        class_name = 'Normal'
        confidence = 1.0
    
    # Display main prediction
    st.markdown("### Prediction (based on selected pathologies)")
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.metric("Prediction", class_name)
    with res_col2:
        st.metric("Confidence", f"{confidence:.1%}")
    
    # Display selected pathologies scores
    st.markdown("---")
    st.markdown("#### Pathology Scores")
    
    for pathology, score in path_scores:
        if pathology not in selected:
            continue
            
        percentage = score * 100
        
        # Color coding based on score
        if percentage > 50:
            status = "[High]"
        elif percentage > 25:
            status = "[Med]"
        else:
            status = "[Low]"
        
        col1, col2, col3 = st.columns([3, 5, 2])
        with col1:
            st.markdown(f"{status} **{pathology}**")
        with col2:
            st.progress(float(min(score, 1.0)))
        with col3:
            st.markdown(f"**{percentage:.1f}%**")
    
    return class_name, confidence


# ============================================
# ============================================
# SIDEBAR NAVIGATION
# ============================================

# Initialize Session State for Page Navigation
if 'page' not in st.session_state:
    st.session_state.page = "Project Overview"

# Navigation Keys (for button callbacks)
def set_page(page_name):
    st.session_state.page = page_name

# Navigation Buttons
st.sidebar.markdown("### Navigation")

# Project Overview Button
if st.sidebar.button("Project Overview", use_container_width=True, type="primary" if st.session_state.page == "Project Overview" else "secondary"):
    set_page("Project Overview")

# Analysis Button
if st.sidebar.button("Analysis", use_container_width=True, type="primary" if st.session_state.page == "Analysis" else "secondary"):
    set_page("Analysis")

# Comparison Button
if st.sidebar.button("Comparison", use_container_width=True, type="primary" if st.session_state.page == "Comparison" else "secondary"):
    set_page("Comparison")

# Get current page from session state
page = st.session_state.page


st.sidebar.markdown("---")
st.sidebar.markdown("### Model Status")
models = get_models()
model_info = get_model_info(models)

for name, info in model_info.items():
    is_loaded = info['status'] == 'loaded'
    status_class = "" if is_loaded else "error"
    status_text = "loaded" if is_loaded else "error"
    icon = "" # Icon removed as requested
    
    st.sidebar.markdown(
        f"""
        <div class="status-badge {status_class}">
            <span class="status-icon">{icon}</span>
            <span style="flex-grow: 1;">{name.title()}</span>
            <span>{status_text}</span>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================
# MAIN CONTENT AREA
# ============================================
st.markdown('<h1 class="main-header">Unified XAI Platform</h1>', unsafe_allow_html=True)



# ============================================
# PAGE: PROJECT OVERVIEW
# ============================================
if page == "Project Overview":
    st.markdown("## Project Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Capabilities")
        st.markdown("""
        - **Audio Analysis**: Detect deepfake audio using MobileNet on Mel-Spectrograms
        - **Image Analysis**: Chest X-Ray pathology detection using TorchXRayVision
        - **XAI Methods**: Grad-CAM, LIME, and SHAP explanations
        - **Comparison Mode**: Side-by-side XAI method comparison
        """)
        
        st.markdown("### Supported Formats")
        st.markdown("""
        | Type | Formats | Model |
        |------|---------|-------|
        | Audio | `.wav` | MobileNetV2 |
        | Image | `.jpg`, `.jpeg`, `.png` | TorchXRayVision |
        """)
    
    with col2:
        st.markdown("### Loaded Models")
        for name, info in model_info.items():
            with st.expander(f"{name.title()} Model"):
                model_name = "MobileNetV2" if name == "audio" else "TorchXRayVision"
                st.markdown(f"**Model:** {model_name}")
                st.markdown(f"**Status:** {info['status']}")
        
        st.markdown("### XAI Methods")
        st.markdown("""
        1. **Grad-CAM**: Gradient-weighted Class Activation Mapping
        2. **LIME**: Local Interpretable Model-agnostic Explanations
        3. **SHAP**: SHapley Additive exPlanations
        """)
        
        st.markdown("### TorchXRayVision Pathologies")
        if models.get('image') is not None and hasattr(models['image'], 'pathologies'):
            pathologies = models['image'].pathologies
            st.caption(f"Detects {len(pathologies)} conditions: {', '.join(pathologies[:6])}...")


# ============================================
# PAGE: ANALYSIS MODE
# ============================================
elif page == "Analysis":
    st.markdown("## Single Analysis Mode")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Audio or Image File",
        type=['wav', 'jpg', 'jpeg', 'png'],
        help="Supported formats: .wav for audio, .jpg/.png for chest X-ray images"
    )
    
    if uploaded_file:
        file_type = detect_file_type(uploaded_file)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Input Preview")
            
            if file_type == 'audio':
                st.audio(uploaded_file, format='audio/wav')
                st.info("Audio file detected")
            elif file_type == 'image':
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", width=300)
                st.info("Chest X-Ray detected")
            else:
                st.error("Unsupported file type")
        
        with col2:
            st.markdown("### Analysis Settings")
            
            xai_method = st.selectbox(
                "Select XAI Method:",
                ["Grad-CAM", "LIME", "SHAP"]
            )
            
            # Show pathology filter for image files
            if file_type == 'image':
                model = models.get('image')
                if model and hasattr(model, 'pathologies'):
                    all_pathologies = list(model.pathologies)
                    selected_pathologies = st.multiselect(
                        "Select pathologies to analyze:",
                        options=all_pathologies,
                        default=['Mass'],
                        help="The prediction (Normal/Abnormal) will be based only on these pathologies"
                    )
                else:
                    selected_pathologies = ['Mass']
            else:
                selected_pathologies = None
            
            if st.button("Analyze", use_container_width=True):
                # Get the appropriate model
                model = models.get(file_type)
                
                if model is None:
                    st.error(f"{file_type.title()} model not loaded!")
                else:
                    with st.spinner("Processing..."):
                        try:
                            # Preprocess based on file type
                            if file_type == 'audio':
                                print("DEBUG: Starting audio processing...")
                                # Save to temp file for processing
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                                    tmp.write(uploaded_file.getvalue())
                                    tmp_path = tmp.name
                                print(f"DEBUG: Temp file created at {tmp_path}")
                                preprocessed = preprocess_audio(tmp_path)
                                print(f"DEBUG: Audio preprocessed, shape: {preprocessed.shape}")
                                os.unlink(tmp_path)
                                vis_img = preprocessed[0]
                                
                                # Get prediction
                                print("DEBUG: Running prediction...")
                                class_name, confidence, _ = get_prediction(model, preprocessed, file_type)
                                print(f"DEBUG: Prediction done: {class_name}, {confidence}")
                                
                                # Display audio results
                                st.markdown("### Results")
                                res_col1, res_col2 = st.columns(2)
                                with res_col1:
                                    st.metric("Prediction", class_name)
                                with res_col2:
                                    st.metric("Confidence", f"{confidence:.1%}")
                                    
                            else:  # Image
                                # Use TorchXRayVision preprocessing
                                img = np.array(Image.open(uploaded_file).convert('RGB'))
                                device = model.device if hasattr(model, 'device') else torch.device('cpu')
                                preprocessed, vis_img = preprocess_image_for_xrv(img, device)
                                
                                # Get all pathology scores
                                _, _, path_scores = get_prediction(model, preprocessed, file_type)
                                
                                # Display prediction and pathology scores based on pre-selected filter
                                display_pathology_scores(path_scores, selected_pathologies)
                                
                                # Note removed as requested
                            
                            # Compute XAI explanation
                            st.markdown(f"### {xai_method} Explanation")
                            
                            if xai_method == "Grad-CAM":
                                if file_type == 'image':
                                    heatmap = XAIEngine.compute_gradcam(model, preprocessed)
                                else:
                                    heatmap = XAIEngine.compute_gradcam(model, preprocessed)
                                result_img = XAIEngine.apply_heatmap(vis_img, heatmap)
                                st.image(result_img, caption="Grad-CAM Heatmap", width="stretch")
                                
                            elif xai_method == "LIME":
                                # For LIME, we need the display image (RGB, 224x224)
                                mask, _ = XAIEngine.compute_lime(model, vis_img[np.newaxis, ...] if len(vis_img.shape) == 3 else vis_img)
                                result_img = XAIEngine.visualize_lime_mask(vis_img, mask)
                                st.image(result_img, caption="LIME Explanation", width="stretch")
                                
                            elif xai_method == "SHAP":
                                shap_heatmap = XAIEngine.compute_shap(model, vis_img[np.newaxis, ...] if len(vis_img.shape) == 3 else vis_img)
                                result_img = XAIEngine.apply_heatmap(vis_img, shap_heatmap)
                                st.image(result_img, caption="SHAP Values", width="stretch")
                            
                        except Exception as e:
                            st.error(f"Analysis failed: {str(e)}")
                            st.exception(e)


# ============================================
# PAGE: COMPARISON MODE
# ============================================
elif page == "Comparison":
    st.markdown("## XAI Method Comparison")
    st.markdown("*Compare all three XAI methods side-by-side*")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Audio or Image File",
        type=['wav', 'jpg', 'jpeg', 'png'],
        key="comparison_upload"
    )
    
    if uploaded_file:
        file_type = detect_file_type(uploaded_file)
        
        # Preview
        st.markdown("### Input Preview")
        if file_type == 'audio':
            st.audio(uploaded_file, format='audio/wav')
        elif file_type == 'image':
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=300)
        
        if st.button("Compare All Methods", use_container_width=True):
            model = models.get(file_type)
            
            if model is None:
                st.error(f"{file_type.title()} model not loaded!")
            else:
                with st.spinner("Running all XAI methods..."):
                    try:
                        # Preprocess
                        if file_type == 'audio':
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                                tmp.write(uploaded_file.getvalue())
                                tmp_path = tmp.name
                            preprocessed = preprocess_audio(tmp_path)
                            os.unlink(tmp_path)
                            vis_img = preprocessed[0]
                            # No prediction display in comparison mode
                        else:
                            # Image with TorchXRayVision
                            img = np.array(Image.open(uploaded_file).convert('RGB'))
                            device = model.device if hasattr(model, 'device') else torch.device('cpu')
                            preprocessed, vis_img = preprocess_image_for_xrv(img, device)
                            # No prediction display in comparison mode
                        
                        st.markdown("---")
                        st.markdown("### XAI Comparison")
                        
                        # Three columns for comparison
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("#### LIME")
                            with st.spinner("Computing LIME..."):
                                mask, _ = XAIEngine.compute_lime(model, vis_img[np.newaxis, ...] if len(vis_img.shape) == 3 else vis_img)
                                lime_img = XAIEngine.visualize_lime_mask(vis_img, mask)
                                st.image(lime_img, caption="LIME Result", width="stretch")
                        
                        with col2:
                            st.markdown("#### Grad-CAM")
                            with st.spinner("Computing Grad-CAM..."):
                                heatmap = XAIEngine.compute_gradcam(model, preprocessed)
                                gradcam_img = XAIEngine.apply_heatmap(vis_img, heatmap)
                                st.image(gradcam_img, caption="Grad-CAM Result", width="stretch")
                        
                        with col3:
                            st.markdown("#### SHAP")
                            with st.spinner("Computing SHAP..."):
                                shap_heatmap = XAIEngine.compute_shap(model, vis_img[np.newaxis, ...] if len(vis_img.shape) == 3 else vis_img)
                                shap_img = XAIEngine.apply_heatmap(vis_img, shap_heatmap)
                                st.image(shap_img, caption="SHAP Result", width="stretch")
                        
                    except Exception as e:
                        st.error(f"Comparison failed: {str(e)}")
                        st.exception(e)


# ============================================
# FOOTER: GENERATIVE AI DISCLOSURE
# ============================================
st.markdown("---")
with st.expander("Generative AI Statement"):
    st.markdown("""
    **Vibe Coding Methodology**
    
    This project was developed using a "Vibe Coding" methodology, utilizing advanced AI assistants like **Claude Code** and autonomous agents to ensure code correctness and efficiency. 
    
    The development process remained strictly **human-in-the-loop**, ensuring the final product aligns perfectly with our original ideas and architectural vision. The AI acted as a powerful accelerator while the human developer maintained full creative and technical control.
    """)

# Fixed footer
st.markdown("""
<div class="footer">
    Unified XAI Platform | Vibe Coding & Human-in-the-Loop Development
</div>
""", unsafe_allow_html=True)
