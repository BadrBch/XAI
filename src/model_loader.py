"""
Model Loader Module for Unified XAI Interface
Handles loading of audio and image classification models.
"""

import os
import streamlit as st
import tensorflow as tf
# We keep standard keras import for modern image models
from tensorflow import keras 

# PyTorch imports for TorchXRayVision
import torch
import torchvision
import torchxrayvision as xrv


# Get the project root directory (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define model paths as absolute paths
AUDIO_MODEL_PATH = os.path.join(PROJECT_ROOT, "Deepfake-Audio-Detection-with-XAI-main", "Streamlit", "saved_model", "model")


def _load_torchxrayvision_model():
    """
    Load TorchXRayVision DenseNet121 pre-trained on multiple chest X-ray datasets.
    
    Returns:
        TorchXRayVision DenseNet model with pathologies attribute.
    """
    print("🏗️ Loading TorchXRayVision DenseNet121 (all datasets)...")
    
    # Load the pre-trained model with weights from all datasets
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    
    # Set to evaluation mode
    model.eval()
    
    # Determine device (MPS for Mac, CUDA for GPU, CPU fallback)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using Apple Metal (MPS) acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ Using CUDA GPU acceleration")
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU (no GPU acceleration)")
    
    model = model.to(device)
    
    # Store device and pathologies for later use
    model.device = device
    
    print(f"✅ TorchXRayVision model loaded with {len(model.pathologies)} pathologies")
    print(f"   Pathologies: {model.pathologies[:5]}...")
    
    return model


@st.cache_resource
def load_all_models() -> dict:
    """
    Load all models for the XAI interface.
    Uses Streamlit caching to avoid reloading on every interaction.
    
    Returns:
        Dictionary containing loaded models:
        {
            'audio': TensorFlow SavedModel (legacy tf_keras) for deepfake detection,
            'image': TorchXRayVision DenseNet121 for chest X-ray analysis
        }
    """
    models = {}
    
    # =========================================
    # AUDIO MODEL: Load TensorFlow SavedModel
    # =========================================
    try:
        if os.path.exists(AUDIO_MODEL_PATH):
            import tf_keras
            # Use tf_keras (Keras 2) to load the legacy SavedModel
            # This preserves layers and enables Grad-CAM
            audio_model = tf_keras.models.load_model(AUDIO_MODEL_PATH)
            models['audio'] = audio_model
            print(f"✅ Audio model loaded from: {AUDIO_MODEL_PATH} (using tf_keras)")
        else:
            print(f"⚠️ Audio model not found at: {AUDIO_MODEL_PATH}")
            models['audio'] = None
    except Exception as e:
        print(f"❌ Error loading audio model: {str(e)}")
        import traceback
        traceback.print_exc()
        try:
            # Fallback to tf.saved_model.load with skip_checkpoint if tf_keras fails
            print("⚠️ Attempting fallback to tf.saved_model.load...")
            load_options = tf.saved_model.LoadOptions(
                experimental_skip_checkpoint=True
            )
            loaded_model = tf.saved_model.load(AUDIO_MODEL_PATH, options=load_options)
            
            class SavedModelWrapper:
                def __init__(self, model):
                    self._model = model
                    if hasattr(model, 'signatures') and 'serving_default' in model.signatures:
                        self.infer = model.signatures['serving_default']
                    else:
                        self.infer = model
                    self.layers = [] # No layers in raw SavedModel
                    
                def predict(self, x, verbose=0):
                    x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
                    result = self.infer(x_tensor)
                    if isinstance(result, dict):
                        return result[list(result.keys())[0]].numpy()
                    return result.numpy()
                
                def __call__(self, x):
                    return self.predict(x)

            models['audio'] = SavedModelWrapper(loaded_model)
            print("✅ Fallback successful (Grad-CAM will be limited)")
        except Exception as e2:
            print(f"❌ Fallback failed: {str(e2)}")
            models['audio'] = None
    
    # =========================================
    # IMAGE MODEL: Load TorchXRayVision
    # =========================================
    try:
        image_model = _load_torchxrayvision_model()
        models['image'] = image_model
        
    except Exception as e:
        print(f"❌ Error with image model: {str(e)}")
        import traceback
        traceback.print_exc()
        models['image'] = None
    
    return models


def get_model_info(models: dict) -> dict:
    """
    Get information about loaded models.
    """
    info = {}
    
    for name, model in models.items():
        if model is not None:
            status = 'loaded'
            input_shape = 'Unknown'
            output_shape = 'Unknown'
            
            try:
                # Check if it's a PyTorch model (TorchXRayVision)
                if hasattr(model, 'pathologies'):
                    # TorchXRayVision model
                    input_shape = '(1, 1, 224, 224)'
                    output_shape = f'({len(model.pathologies)} pathologies)'
                elif hasattr(model, 'input_shape'):
                    input_shape = str(model.input_shape)
                elif hasattr(model, 'inputs') and hasattr(model.inputs[0], 'shape'):
                    input_shape = str(model.inputs[0].shape)
                    
                if hasattr(model, 'output_shape'):
                    output_shape = str(model.output_shape)
                elif hasattr(model, 'outputs') and hasattr(model.outputs[0], 'shape'):
                    output_shape = str(model.outputs[0].shape)
            except:
                pass
                
            info[name] = {
                'status': status,
                'input_shape': input_shape,
                'output_shape': output_shape
            }
        else:
            info[name] = {
                'status': 'not loaded',
                'input_shape': 'N/A',
                'output_shape': 'N/A'
            }
    
    return info
