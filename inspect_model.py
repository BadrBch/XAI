
import os
import tensorflow as tf

# Try to import tf_keras (standalone keras 2)
try:
    import tf_keras
    keras_lib = tf_keras
    print("Using tf_keras (Legacy Support)")
except ImportError:
    print("tf_keras not found, falling back to tensorflow.keras (might fail for SavedModel in Keras 3)")
    keras_lib = tf.keras

PROJECT_ROOT = os.getcwd()
AUDIO_MODEL_PATH = os.path.join(PROJECT_ROOT, "Deepfake-Audio-Detection-with-XAI-main", "Streamlit", "saved_model", "model")

try:
    print(f"Loading model from {AUDIO_MODEL_PATH}...")
    model = keras_lib.models.load_model(AUDIO_MODEL_PATH)
    model.summary()
    
    # Check layer names to infer architecture
    layer_names = [l.name for l in model.layers]
    print("\nFirst 10 layers:", layer_names[:10])
    
    if any("mobilenet" in name.lower() for name in layer_names) or any("depthwise" in name.lower() for name in layer_names):
        print("\nLikely Architecture: MobileNet")
    elif any("vgg" in name.lower() for name in layer_names) or any("block1_conv1" in name.lower() for name in layer_names):
        print("\nLikely Architecture: VGG")
    elif any("resnet" in name.lower() for name in layer_names):
        print("\nLikely Architecture: ResNet")
    else:
        print("\nLikely Architecture: Custom CNN")
        
except Exception as e:
    print(f"Error loading model: {e}")
