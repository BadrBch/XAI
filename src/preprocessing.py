"""
Preprocessing Module for Unified XAI Interface
Handles audio and image preprocessing for model inference.
"""

import numpy as np
import cv2
import os
import librosa
import librosa.display
from typing import Optional

# PyTorch imports for TorchXRayVision preprocessing
import torch
import torchvision
import torchxrayvision as xrv


def preprocess_audio(file_path: str, duration: float = 2.0, sr: int = 22050) -> np.ndarray:
    """
    Preprocess audio file for deepfake detection model.
    
    Args:
        file_path: Path to the audio file (.wav)
        duration: Duration in seconds to load (default: 2s)
        sr: Sample rate (default: 22050)
    
    Returns:
        Preprocessed audio as numpy array with shape (1, 224, 224, 3)
    
    Raises:
        FileNotFoundError: If audio file doesn't exist
        ValueError: If audio file is corrupted or unreadable
    """
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage
    import tempfile
    
    try:
        # Load audio (matching original: no specific duration)
        audio, sr_loaded = librosa.load(file_path, sr=sr)
        
        # Create spectrogram figure exactly like the original
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        
        # Generate Mel Spectrogram (matching original)
        ms = librosa.feature.melspectrogram(y=audio, sr=sr_loaded)
        log_ms = librosa.power_to_db(ms, ref=np.max)
        librosa.display.specshow(log_ms, sr=sr_loaded)
        
        # Save to temporary file (matching original behavior)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
        plt.savefig(tmp_path)
        plt.close(fig)
        
        # Load the image back (matching original: load_img with target_size)
        image_data = PILImage.open(tmp_path).convert('RGB').resize((224, 224))
        
        # Clean up temp file
        os.remove(tmp_path)
        
        # Convert to array and normalize (matching original: divide by 255)
        img_array = np.array(image_data)
        img_array_normalized = img_array / 255.0
        img_batch = np.expand_dims(img_array_normalized, axis=0).astype(np.float32)
        
        return img_batch
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Error processing audio file: {str(e)}")


def preprocess_image(file_path: str) -> np.ndarray:
    """
    Preprocess image file for classification model.
    
    Args:
        file_path: Path to the image file (.jpg, .png)
    
    Returns:
        Preprocessed image as numpy array with shape (1, 224, 224, 3)
    
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image file is corrupted or unreadable
    """
    try:
        # Load image using OpenCV
        img = cv2.imread(file_path)
        
        if img is None:
            raise ValueError(f"Failed to load image: {file_path}")
        
        # Resize to (224, 224)
        img_resized = cv2.resize(img, (224, 224))
        
        # Convert BGR to RGB (OpenCV loads as BGR)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to 0-1 range
        img_normalized = img_rgb.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Error processing image file: {str(e)}")


def preprocess_image_from_array(img_array: np.ndarray) -> np.ndarray:
    """
    Preprocess image from numpy array (for uploaded files).
    Used for legacy TensorFlow models.
    
    Args:
        img_array: Input image as numpy array (RGB format)
    
    Returns:
        Preprocessed image as numpy array with shape (1, 224, 224, 3)
    """
    try:
        # Ensure array is in correct format
        if len(img_array.shape) == 2:
            # Grayscale - convert to RGB
            img_array = np.stack([img_array] * 3, axis=-1)
        
        # Resize to (224, 224)
        img_resized = cv2.resize(img_array, (224, 224))
        
        # Normalize to 0-1 range if not already
        if img_resized.max() > 1.0:
            img_resized = img_resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_resized, axis=0).astype(np.float32)
        
        return img_batch
        
    except Exception as e:
        raise ValueError(f"Error processing image array: {str(e)}")


def preprocess_image_for_xrv(img_array: np.ndarray, device: torch.device = None) -> tuple:
    """
    Preprocess image for TorchXRayVision model.
    
    TorchXRayVision requires:
    1. Normalization to [-1024, 1024] range using xrv.datasets.normalize()
    2. Single channel (grayscale)
    3. Center crop and resize to 224x224
    4. PyTorch tensor format
    
    Args:
        img_array: Input image as numpy array (RGB or grayscale)
        device: PyTorch device to move tensor to
    
    Returns:
        Tuple of (preprocessed PyTorch tensor, display image numpy array)
        - tensor: Shape (1, 1, 224, 224) for model input
        - display_img: Shape (224, 224, 3) normalized 0-1 for visualization
    """
    try:
        # If RGB, convert to grayscale
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            # Use mean across channels (as per TorchXRayVision docs)
            img_gray = img_array.mean(axis=2)
        elif len(img_array.shape) == 2:
            img_gray = img_array
        else:
            img_gray = img_array[:, :, 0]  # Take first channel
        
        # Normalize to [-1024, 1024] range (TorchXRayVision requirement)
        # xrv.datasets.normalize expects image in 0-255 range
        if img_gray.max() <= 1.0:
            img_gray = (img_gray * 255).astype(np.float32)
        
        img_normalized = xrv.datasets.normalize(img_gray, 255)
        
        # Apply TorchXRayVision transforms
        transform = torchvision.transforms.Compose([
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224)
        ])
        
        # Add channel dimension for transforms: (H, W) -> (1, H, W)
        img_for_transform = img_normalized[None, ...]
        img_transformed = transform(img_for_transform)
        
        # Convert to PyTorch tensor
        img_tensor = torch.from_numpy(img_transformed).float()
        
        # Add batch dimension: (1, H, W) -> (1, 1, H, W)
        img_tensor = img_tensor.unsqueeze(0)
        
        # Move to device if specified
        if device is not None:
            img_tensor = img_tensor.to(device)
        
        # Create display image for visualization (RGB format, 0-1 range)
        # Resize original RGB image to 224x224
        if len(img_array.shape) == 3:
            display_img = cv2.resize(img_array, (224, 224))
        else:
            display_img = cv2.resize(np.stack([img_array] * 3, axis=-1), (224, 224))
        
        if display_img.max() > 1.0:
            display_img = display_img.astype(np.float32) / 255.0
        
        return img_tensor, display_img
        
    except Exception as e:
        raise ValueError(f"Error preprocessing image for TorchXRayVision: {str(e)}")
