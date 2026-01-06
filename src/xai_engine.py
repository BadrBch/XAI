"""
XAI Engine Module for Unified XAI Interface
Implements Grad-CAM, LIME, and SHAP explainability methods.
Supports both TensorFlow/Keras and PyTorch models.
"""

import numpy as np
import cv2
import tensorflow as tf
from lime import lime_image
import shap
from typing import Optional, Tuple

# PyTorch imports for TorchXRayVision XAI
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def is_pytorch_model(model) -> bool:
    """Check if model is a PyTorch model."""
    return isinstance(model, torch.nn.Module)


class XAIEngine:
    """
    Static methods for computing various XAI explanations.
    Supports both TensorFlow and PyTorch models.
    """
    
    @staticmethod
    def compute_gradcam(
        model: object,
        img_array: np.ndarray,
        layer_name: Optional[str] = None,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for the given image.
        Automatically detects PyTorch vs TensorFlow models.
        
        Args:
            model: Keras or PyTorch model to analyze
            img_array: Input image array 
                - TensorFlow: shape (1, 224, 224, 3)
                - PyTorch: shape (1, 1, 224, 224) tensor or will be converted
            layer_name: Target conv layer name (optional, auto-detects)
            target_class: Target class index for Grad-CAM (optional)
        
        Returns:
            Heatmap normalized to 0-1 range with shape (224, 224)
        """
        if is_pytorch_model(model):
            return XAIEngine._compute_gradcam_pytorch(model, img_array, target_class)
        else:
            return XAIEngine._compute_gradcam_tensorflow(model, img_array, layer_name)
    
    @staticmethod
    def _compute_gradcam_pytorch(
        model: torch.nn.Module,
        img_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute Grad-CAM for PyTorch models (TorchXRayVision).
        
        Args:
            model: PyTorch model (TorchXRayVision DenseNet)
            img_tensor: Input tensor with shape (1, 1, 224, 224)
            target_class: Target class index (0-17 for pathologies)
        
        Returns:
            Heatmap normalized to 0-1 range with shape (224, 224)
        """
        # Ensure input is on correct device
        device = next(model.parameters()).device
        
        if isinstance(img_tensor, np.ndarray):
            img_tensor = torch.from_numpy(img_tensor).float()
        
        img_tensor = img_tensor.to(device)
        
        # Ensure requires_grad
        img_tensor.requires_grad = True
        
        # Find target layer for DenseNet (last dense block)
        # TorchXRayVision DenseNet has features.denseblock4 as last conv block
        target_layers = [model.features.denseblock4]
        
        # Initialize GradCAM
        cam = GradCAM(model=model, target_layers=target_layers)
        
        # If no target class specified, use class with highest output
        if target_class is None:
            with torch.no_grad():
                outputs = model(img_tensor)
                target_class = outputs.argmax(dim=1).item()
        
        targets = [ClassifierOutputTarget(target_class)]
        
        # Compute CAM
        grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
        
        # Get the heatmap for the first image in batch
        heatmap = grayscale_cam[0, :]
        
        return heatmap
    
    @staticmethod
    def _compute_gradcam_tensorflow(
        model: object,
        img_array: np.ndarray,
        layer_name: Optional[str] = None
    ) -> np.ndarray:
        """
        Compute Grad-CAM for TensorFlow/Keras models (Audio model).
        
        Args:
            model: Keras model to analyze (tf.keras or tf_keras)
            img_array: Input image array with shape (1, 224, 224, 3)
            layer_name: Target conv layer name. If None, auto-detects last conv layer.
        
        Returns:
            Heatmap normalized to 0-1 range with shape (224, 224)
        """
        # Determine strict type of model to select correct Keras backend
        is_legacy_model = False
        try:
            import tf_keras
            if isinstance(model, tf_keras.Model):
                is_legacy_model = True
                keras_module = tf_keras
            else:
                keras_module = tf.keras
        except ImportError:
            keras_module = tf.keras

        # Auto-detect layer for Grad-CAM
        if layer_name is None:
            # Look for last "conv" layer
            for layer in reversed(model.layers):
                if 'conv' in layer.name.lower():
                    layer_name = layer.name
                    break
            
            if layer_name is None:
                for layer in reversed(model.layers):
                    if 'pool' in layer.name.lower() and 'global' not in layer.name.lower():
                        layer_name = layer.name
                        break
            
            if layer_name is None:
                print("Available layers:", [l.name for l in model.layers])
                raise ValueError("No suitable convolutional layer found in model for Grad-CAM")
            
            print(f"🔍 Grad-CAM using layer: {layer_name}")
        
        # Create gradient model using the correct Keras backend
        grad_model = keras_module.Model(
            inputs=model.input,
            outputs=[model.get_layer(layer_name).output, model.output]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            tensor_input = tf.convert_to_tensor(img_array, dtype=tf.float32)
            conv_outputs, predictions = grad_model(tensor_input)
            predicted_class = tf.argmax(predictions[0])
            class_output = predictions[:, predicted_class]
        
        grads = tape.gradient(class_output, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
        
        heatmap_np = heatmap.numpy()
        heatmap_resized = cv2.resize(heatmap_np, (224, 224))
        
        return heatmap_resized
    
    @staticmethod
    def compute_lime(
        model: object,
        img_array: np.ndarray,
        num_samples: int = 100
    ) -> Tuple[np.ndarray, object]:
        """
        Compute LIME explanation for the given image.
        Supports both TensorFlow and PyTorch models.
        
        Args:
            model: Model to analyze
            img_array: Input image array with shape (1, 224, 224, 3) or display image
            num_samples: Number of samples for LIME (default: 100)
        
        Returns:
            Tuple of (explanation mask, lime explanation object)
        """
        # Create LIME explainer
        explainer = lime_image.LimeImageExplainer()
        
        # Prediction wrapper based on model type
        if is_pytorch_model(model):
            device = next(model.parameters()).device
            
            def predict_fn(images):
                """PyTorch prediction wrapper for LIME."""
                model.eval()
                with torch.no_grad():
                    # Convert images to tensor
                    # LIME sends (N, H, W, 3) RGB images
                    imgs_np = np.array(images).astype(np.float32)
                    
                    # Convert to grayscale and normalize for TorchXRayVision
                    import torchxrayvision as xrv
                    
                    results = []
                    for img in imgs_np:
                        # Grayscale
                        if img.max() > 1.0:
                            img = img / 255.0
                        img_gray = img.mean(axis=2) * 255.0
                        img_norm = xrv.datasets.normalize(img_gray, 255)
                        
                        # Resize to 224x224
                        img_resized = cv2.resize(img_norm, (224, 224))
                        
                        tensor = torch.from_numpy(img_resized[None, None, ...]).float().to(device)
                        output = model(tensor)
                        results.append(output.cpu().numpy()[0])
                    
                    return np.array(results)
        else:
            def predict_fn(images):
                """TensorFlow prediction wrapper for LIME."""
                import tensorflow as tf
                images = np.array(images).astype(np.float32)
                if images.max() > 1.0:
                    images = images / 255.0
                # Use direct call instead of predict() to avoid deadlocks
                tensor_input = tf.convert_to_tensor(images, dtype=tf.float32)
                return model(tensor_input, training=False).numpy()
        
        # Get the image without batch dimension
        if len(img_array.shape) == 4:
            img = img_array[0]
        else:
            img = img_array
        
        # Scale to 0-255 for LIME (it expects uint8-like input)
        img_scaled = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        
        # Generate LIME explanation
        explanation = explainer.explain_instance(
            img_scaled,
            predict_fn,
            top_labels=5 if is_pytorch_model(model) else 2,
            hide_color=0,
            num_samples=num_samples
        )
        
        # Get the explanation mask for the top predicted class
        if len(img_array.shape) == 4:
            preds = predict_fn([img_scaled])
        else:
            preds = predict_fn([img_scaled])
        
        predicted_class = np.argmax(preds[0])
        temp, mask = explanation.get_image_and_mask(
            predicted_class,
            positive_only=True,
            num_features=10,
            hide_rest=False
        )
        
        return mask, explanation
    
    @staticmethod
    def compute_shap(
        model: object,
        img_array: np.ndarray,
        nsamples: int = 50
    ) -> np.ndarray:
        """
        Compute SHAP values for the given image.
        Supports both TensorFlow and PyTorch models.
        
        Args:
            model: Model to analyze
            img_array: Input image array with shape (1, 224, 224, 3)
            nsamples: Maximum number of samples (default: 50 for speed)
        
        Returns:
            SHAP values array
        """
        if is_pytorch_model(model):
            return XAIEngine._compute_shap_pytorch(model, img_array, nsamples)
        else:
            return XAIEngine._compute_shap_tensorflow(model, img_array, nsamples)
    
    @staticmethod
    def _compute_shap_pytorch(
        model: torch.nn.Module,
        img_array: np.ndarray,
        nsamples: int = 50
    ) -> np.ndarray:
        """
        Compute SHAP values for PyTorch models using KernelExplainer.
        """
        device = next(model.parameters()).device
        
        # Flatten input for KernelExplainer
        flat_img = img_array.reshape(1, -1)
        flat_background = np.zeros((1, flat_img.shape[1]), dtype=np.float32)
        
        def predict_check_reshape(data):
            """Prediction wrapper for SHAP."""
            import torchxrayvision as xrv
            
            model.eval()
            with torch.no_grad():
                results = []
                for sample in data:
                    # Reshape back to image
                    img = sample.reshape(224, 224, 3)
                    
                    # Convert to grayscale and normalize
                    img_gray = img.mean(axis=2) * 255.0
                    img_norm = xrv.datasets.normalize(img_gray, 255)
                    img_resized = cv2.resize(img_norm, (224, 224))
                    
                    tensor = torch.from_numpy(img_resized[None, None, ...]).float().to(device)
                    output = model(tensor)
                    results.append(output.cpu().numpy()[0])
                
                return np.array(results)
        
        explainer = shap.KernelExplainer(predict_check_reshape, flat_background)
        
        shap_values_flat = explainer.shap_values(
            flat_img,
            nsamples=min(nsamples, 50), 
            silent=True
        )
        
        # Reshape SHAP values back to image shape
        if isinstance(shap_values_flat, list):
            # Multi-output: use first class or average
            shap_values = shap_values_flat[0].reshape(-1, 224, 224, 3)
        else:
            shap_values = shap_values_flat.reshape(-1, 224, 224, 3)
        
        # Aggregate across color channels for visualization
        shap_heatmap = np.abs(shap_values[0]).sum(axis=-1)
        
        # Normalize to 0-1
        shap_heatmap = (shap_heatmap - shap_heatmap.min()) / (shap_heatmap.max() - shap_heatmap.min() + 1e-8)
        
        return shap_heatmap
    
    @staticmethod
    def _compute_shap_tensorflow(
        model: object,
        img_array: np.ndarray,
        nsamples: int = 50
    ) -> np.ndarray:
        """
        Compute SHAP values for TensorFlow models.
        """
        background = np.zeros((1, 224, 224, 3), dtype=np.float32)
        
        use_kernel_explainer = False
        try:
            import tf_keras
            if isinstance(model, tf_keras.Model):
                use_kernel_explainer = True
        except ImportError:
            pass
            
        if use_kernel_explainer:
            flat_img = img_array.reshape(1, -1)
            flat_background = np.zeros((1, flat_img.shape[1]), dtype=np.float32)
            
            def predict_check_reshape(data):
                import tensorflow as tf
                if len(data.shape) == 2:
                    reshaped_data = data.reshape(-1, 224, 224, 3)
                else:
                    reshaped_data = data
                # Use direct call instead of predict() to avoid deadlocks
                tensor_input = tf.convert_to_tensor(reshaped_data, dtype=tf.float32)
                return model(tensor_input, training=False).numpy()

            explainer = shap.KernelExplainer(predict_check_reshape, flat_background)
            
            shap_values_flat = explainer.shap_values(
                flat_img,
                nsamples=min(nsamples, 50), 
                silent=True
            )
            
            if isinstance(shap_values_flat, list):
                shap_values = [sv.reshape(-1, 224, 224, 3) for sv in shap_values_flat]
            else:
                shap_values = shap_values_flat.reshape(-1, 224, 224, 3)
                
        else:
            explainer = shap.GradientExplainer(model, background)
            shap_values = explainer.shap_values(
                img_array,
                nsamples=min(nsamples, 50)
            )
        
        if isinstance(shap_values, list):
            import tensorflow as tf
            tensor_input = tf.convert_to_tensor(img_array, dtype=tf.float32)
            preds = model(tensor_input, training=False).numpy()
            predicted_class = np.argmax(preds[0])
            shap_values = shap_values[predicted_class]
        
        shap_heatmap = np.abs(shap_values[0]).sum(axis=-1)
        shap_heatmap = (shap_heatmap - shap_heatmap.min()) / (shap_heatmap.max() - shap_heatmap.min() + 1e-8)
        
        return shap_heatmap
    
    @staticmethod
    def apply_heatmap(
        original_img: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Apply heatmap overlay on the original image.
        
        Args:
            original_img: Original image (H, W, 3) in range 0-1 or 0-255
            heatmap: Heatmap array (H', W') in range 0-1
            alpha: Blending factor (default: 0.4)
            colormap: OpenCV colormap (default: COLORMAP_JET)
        
        Returns:
            Blended image with heatmap overlay (H, W, 3) in range 0-255
        """
        h, w = original_img.shape[:2]
        
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        if original_img.max() <= 1.0:
            original_img = (original_img * 255).astype(np.uint8)
        else:
            original_img = original_img.astype(np.uint8)
        
        blended = cv2.addWeighted(original_img, 1 - alpha, heatmap_colored, alpha, 0)
        
        return blended
    
    @staticmethod
    def visualize_lime_mask(
        original_img: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Visualize LIME explanation mask on original image.
        
        Args:
            original_img: Original image (H, W, 3)
            mask: LIME mask array
            alpha: Blending factor
        
        Returns:
            Visualization image
        """
        if original_img.max() <= 1.0:
            img = (original_img * 255).astype(np.uint8)
        else:
            img = original_img.astype(np.uint8)
        
        mask_colored = np.zeros_like(img)
        mask_colored[mask > 0] = [0, 255, 0]
        
        blended = cv2.addWeighted(img, 1 - alpha, mask_colored, alpha, 0)
        
        return blended
