# 🎯 MEGA-PROMPT: Unified Explainable AI Interface
## Pour Vibe Coding - Objectif 20/20

---

# PARTIE 1: COMPRÉHENSION TECHNIQUE DES XAI

## 1.1 LIME (Local Interpretable Model-agnostic Explanations)

### Principe Fondamental
LIME explique les prédictions d'un modèle "boîte noire" en créant un modèle simple (linéaire) autour d'une prédiction spécifique.

### Comment ça fonctionne (étape par étape)
1. **Perturbation**: Génère des versions modifiées de l'entrée (pour images: cache certaines régions/superpixels)
2. **Prédiction**: Obtient les prédictions du modèle sur ces versions perturbées
3. **Pondération**: Attribue des poids basés sur la similarité avec l'entrée originale
4. **Approximation locale**: Entraîne un modèle interprétable (régression linéaire) sur ces données
5. **Visualisation**: Affiche quelles régions contribuent positivement/négativement

### Pour les IMAGES (Spectrogrammes & X-rays)
```python
from lime import lime_image
from skimage.segmentation import mark_boundaries

# Créer l'explainer
explainer = lime_image.LimeImageExplainer()

# Générer l'explication
explanation = explainer.explain_instance(
    image,                           # Image numpy array (H, W, C)
    model.predict,                   # Fonction de prédiction (retourne probas)
    top_labels=2,                    # Nombre de classes à expliquer
    hide_color=0,                    # Couleur pour masquer (0=gris)
    num_samples=1000                 # Plus = meilleure explication mais plus lent
)

# Obtenir le masque pour la classe prédite
temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[0],       # Classe à visualiser
    positive_only=True,              # True = que les régions positives
    num_features=5,                  # Nombre de superpixels à afficher
    hide_rest=False                  # False = montrer l'image complète
)

# Visualiser
plt.imshow(mark_boundaries(temp / 255.0, mask))
```

### ⚠️ Pièges à éviter avec LIME
- **Images grayscale**: LIME attend du RGB (3 canaux). Convertir avec `np.stack([img]*3, axis=-1)`
- **Format du modèle**: `predict` doit retourner des probabilités, pas des classes
- **Normalisation**: S'assurer que les images sont dans la même plage que l'entraînement
- **Lenteur**: LIME est lent (~1-2 min par image). Réduire `num_samples` pour la démo

---

## 1.2 Grad-CAM (Gradient-weighted Class Activation Mapping)

### Principe Fondamental
Grad-CAM utilise les gradients qui circulent dans la dernière couche convolutionnelle pour produire une heatmap montrant quelles régions de l'image sont importantes pour une classe donnée.

### Comment ça fonctionne (étape par étape)
1. **Forward pass**: Passe l'image dans le modèle, récupère les feature maps de la dernière couche conv
2. **Backward pass**: Calcule les gradients de la classe cible par rapport à ces feature maps
3. **Pondération**: Fait la moyenne globale des gradients (global average pooling)
4. **Combinaison**: Multiplie chaque feature map par son poids et fait la somme
5. **ReLU**: Applique ReLU pour ne garder que les influences positives
6. **Upsample**: Redimensionne la heatmap à la taille de l'image originale

### Formule mathématique
```
α_k = (1/Z) Σ_i Σ_j (∂y^c / ∂A^k_ij)    # Poids = moyenne des gradients
L_Grad-CAM = ReLU(Σ_k α_k · A^k)         # Heatmap = somme pondérée + ReLU
```

### Implémentation TensorFlow/Keras
```python
import tensorflow as tf
import numpy as np
import cv2

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Créer un modèle qui retourne les activations de la couche conv + les prédictions
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    # Calculer le gradient de la classe prédite par rapport aux feature maps
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    
    # Gradient de la sortie par rapport aux feature maps
    grads = tape.gradient(class_channel, conv_outputs)
    
    # Moyenne des gradients sur chaque feature map (global average pooling)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Pondérer les feature maps par les gradients moyens
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normaliser entre 0 et 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_gradcam(img, heatmap, alpha=0.4):
    # Redimensionner la heatmap à la taille de l'image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convertir en colormap
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superposer
    superimposed = cv2.addWeighted(img, 1-alpha, heatmap, alpha, 0)
    return superimposed
```

### Noms des couches convolutionnelles par modèle
| Modèle | Dernière couche conv |
|--------|---------------------|
| VGG16 | `block5_conv3` |
| VGG19 | `block5_conv4` |
| ResNet50 | `conv5_block3_out` |
| MobileNet | `conv_pw_13_relu` |
| MobileNetV2 | `out_relu` |
| InceptionV3 | `mixed10` |
| DenseNet121 | `relu` |
| AlexNet (custom) | Dernière couche avec "conv" dans le nom |

### ⚠️ Pièges à éviter avec Grad-CAM
- **Trouver la bonne couche**: Utiliser `model.summary()` pour voir les noms
- **Input shape**: L'image doit avoir la bonne forme (batch dimension incluse)
- **Modèles Functional vs Sequential**: L'accès aux couches diffère légèrement
- **Grayscale**: Convertir la heatmap en RGB avant overlay

---

## 1.3 SHAP (SHapley Additive exPlanations)

### Principe Fondamental
SHAP utilise la théorie des jeux (valeurs de Shapley) pour attribuer à chaque feature sa contribution à la prédiction. C'est mathématiquement la seule méthode qui garantit certaines propriétés désirables (local accuracy, missingness, consistency).

### Comment ça fonctionne
1. Considère toutes les coalitions possibles de features
2. Pour chaque coalition, calcule la prédiction avec et sans la feature
3. La valeur Shapley = moyenne pondérée des contributions marginales
4. Pour les images, utilise des superpixels comme "features"

### Implémentation avec shap library
```python
import shap
import numpy as np

# Pour les images - utiliser GradientExplainer (rapide) ou DeepExplainer
def shap_explain_image(model, image, background_data):
    """
    model: modèle Keras/TF
    image: image à expliquer (1, H, W, C)
    background_data: échantillon de données d'entraînement (N, H, W, C)
    """
    # GradientExplainer est plus rapide que KernelExplainer pour les CNNs
    explainer = shap.GradientExplainer(model, background_data)
    
    # Calculer les valeurs SHAP
    shap_values = explainer.shap_values(image)
    
    return shap_values

# Visualisation
def visualize_shap(shap_values, image, class_idx=0):
    """
    shap_values: output de shap_explain_image
    image: image originale
    class_idx: index de la classe à visualiser
    """
    # Pour classification binaire ou multi-classe
    if isinstance(shap_values, list):
        values = shap_values[class_idx]
    else:
        values = shap_values
    
    # Visualiser
    shap.image_plot(values, image, show=False)
    return plt.gcf()
```

### Alternative: SHAP avec superpixels (plus interprétable)
```python
from skimage.segmentation import slic

def shap_superpixel_explain(model, image, num_segments=50):
    # Segmenter l'image en superpixels
    segments = slic(image, n_segments=num_segments, compactness=10)
    
    # Créer une fonction de masquage
    def mask_image(zs, segments, image):
        out = np.zeros((zs.shape[0], *image.shape))
        for i in range(zs.shape[0]):
            for j in range(zs.shape[1]):
                if zs[i, j] == 1:
                    out[i][segments == j] = image[segments == j]
        return out
    
    # Fonction de prédiction pour SHAP
    def f(zs):
        masked_images = mask_image(zs, segments, image)
        return model.predict(masked_images)
    
    # Créer l'explainer et calculer
    num_features = len(np.unique(segments))
    explainer = shap.KernelExplainer(f, np.zeros((1, num_features)))
    shap_values = explainer.shap_values(np.ones((1, num_features)), nsamples=200)
    
    return shap_values, segments
```

### ⚠️ Pièges à éviter avec SHAP
- **Données de background**: Crucial! Utiliser un échantillon représentatif (100-200 images)
- **Très lent**: KernelExplainer peut prendre plusieurs minutes par image
- **Mémoire**: GradientExplainer est plus efficace pour les grands modèles
- **Interprétation**: Pour images, les valeurs brutes sont difficiles à interpréter → utiliser superpixels

---

# PARTIE 2: ARCHITECTURE DU PROJET

## 2.1 Structure de fichiers recommandée

```
unified_xai_platform/
├── app.py                      # Point d'entrée Streamlit
├── requirements.txt            # Dépendances
├── README.md                   # Documentation
│
├── models/                     # Modèles et poids
│   ├── audio/
│   │   ├── vgg16_audio.h5
│   │   ├── mobilenet_audio.h5
│   │   ├── resnet_audio.h5
│   │   └── custom_cnn_audio.h5
│   └── image/
│       ├── alexnet_xray.h5
│       └── densenet_xray.h5
│
├── src/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── audio_processor.py   # Audio → Spectrogramme
│   │   └── image_processor.py   # X-ray preprocessing
│   │
│   ├── classifiers/
│   │   ├── __init__.py
│   │   ├── audio_classifier.py  # Modèles audio
│   │   └── image_classifier.py  # Modèles X-ray
│   │
│   └── explainers/
│       ├── __init__.py
│       ├── lime_explainer.py    # LIME pour images
│       ├── gradcam_explainer.py # Grad-CAM
│       └── shap_explainer.py    # SHAP
│
├── data/                        # Données de démo
│   ├── sample_audio/
│   └── sample_xrays/
│
└── assets/                      # Images UI, logos
```

## 2.2 Mapping Input Type → Modèles → XAI

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INPUT                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Detect Type    │
                    │  .wav → Audio   │
                    │  .jpg/.png → Img│
                    └────────┬────────┘
                             │
           ┌─────────────────┴─────────────────┐
           │                                   │
           ▼                                   ▼
    ┌──────────────┐                   ┌──────────────┐
    │    AUDIO     │                   │    IMAGE     │
    │  (.wav)      │                   │  (X-ray)     │
    └──────┬───────┘                   └──────┬───────┘
           │                                   │
           ▼                                   ▼
    ┌──────────────┐                   ┌──────────────┐
    │ Audio→Spect  │                   │  Preprocess  │
    │ (Mel Spectro)│                   │  (Resize,    │
    │              │                   │   Normalize) │
    └──────┬───────┘                   └──────┬───────┘
           │                                   │
           ▼                                   ▼
    ┌──────────────┐                   ┌──────────────┐
    │   MODÈLES    │                   │   MODÈLES    │
    │ • VGG16      │                   │ • AlexNet    │
    │ • MobileNet  │                   │ • DenseNet   │
    │ • ResNet     │                   │              │
    │ • Custom CNN │                   │              │
    └──────┬───────┘                   └──────┬───────┘
           │                                   │
           ▼                                   ▼
    ┌──────────────┐                   ┌──────────────┐
    │     XAI      │                   │     XAI      │
    │ ✓ LIME       │                   │ ✓ LIME       │
    │ ✓ Grad-CAM   │                   │ ✓ Grad-CAM   │
    │ ✓ SHAP       │                   │ ✓ SHAP       │
    └──────────────┘                   └──────────────┘
```

## 2.3 Logique de filtrage XAI automatique

```python
# Configuration de compatibilité XAI
XAI_COMPATIBILITY = {
    "audio": {
        "lime": True,      # Fonctionne sur spectrogrammes
        "gradcam": True,   # Fonctionne sur CNNs
        "shap": True,      # Fonctionne sur tous modèles
    },
    "image": {
        "lime": True,
        "gradcam": True,
        "shap": True,
    }
}

# Couches Grad-CAM par modèle
GRADCAM_LAYERS = {
    # Audio models
    "vgg16_audio": "block5_conv3",
    "mobilenet_audio": "conv_pw_13_relu",
    "resnet_audio": "conv5_block3_out",
    "custom_cnn_audio": "conv2d_4",  # À ajuster selon l'architecture
    
    # Image models
    "alexnet_xray": "conv2d_5",      # À ajuster
    "densenet_xray": "relu",
}
```

---

# PARTIE 3: CODE CRITIQUE (NON VIBE-CODABLE)

## 3.1 Audio → Spectrogramme (Preprocessing Crucial)

```python
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

def audio_to_spectrogram(audio_path, target_size=(224, 224)):
    """
    Convertit un fichier audio en spectrogramme Mel.
    
    CRITIQUE: Cette fonction détermine la qualité de la classification!
    
    Paramètres:
    - audio_path: chemin vers le fichier .wav
    - target_size: taille de sortie pour le modèle (224x224 pour VGG/ResNet)
    
    Retourne:
    - numpy array de shape (H, W, 3) normalisé entre 0 et 1
    """
    # Charger l'audio
    y, sr = librosa.load(audio_path, sr=22050, duration=2.0)
    
    # Paramètres du spectrogramme Mel
    # Ces valeurs sont cruciales et doivent correspondre à l'entraînement!
    n_mels = 128
    n_fft = 2048
    hop_length = 512
    
    # Calculer le spectrogramme Mel
    mel_spec = librosa.feature.melspectrogram(
        y=y, 
        sr=sr, 
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    # Convertir en dB (log scale) - IMPORTANT!
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Créer l'image du spectrogramme
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.specshow(
        mel_spec_db,
        sr=sr,
        hop_length=hop_length,
        x_axis='time',
        y_axis='mel',
        ax=ax,
        cmap='magma'  # Colormap utilisée lors de l'entraînement
    )
    ax.axis('off')
    plt.tight_layout(pad=0)
    
    # Convertir figure en array
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    plt.close(fig)
    
    # Convertir en numpy array normalisé
    spec_array = np.array(img) / 255.0
    
    return spec_array

def preprocess_for_model(spectrogram, model_name):
    """
    Applique le preprocessing spécifique au modèle.
    
    CRUCIAL: Chaque modèle pré-entraîné attend un format différent!
    """
    from tensorflow.keras.applications import (
        vgg16, mobilenet, resnet50, inception_v3
    )
    
    # Ajouter la dimension batch
    img = np.expand_dims(spectrogram, axis=0)
    
    # Convertir en float32 et scale 0-255
    img = (img * 255).astype(np.float32)
    
    # Appliquer le preprocessing du modèle
    if 'vgg' in model_name.lower():
        img = vgg16.preprocess_input(img)
    elif 'mobilenet' in model_name.lower():
        img = mobilenet.preprocess_input(img)
    elif 'resnet' in model_name.lower():
        img = resnet50.preprocess_input(img)
    elif 'inception' in model_name.lower():
        img = inception_v3.preprocess_input(img)
    else:
        # Pour les modèles custom, normaliser entre -1 et 1 ou 0 et 1
        img = img / 255.0
    
    return img
```

## 3.2 X-ray Preprocessing

```python
import cv2
import numpy as np
from tensorflow.keras.applications import densenet, vgg16

def preprocess_xray(image_path, target_size=(224, 224)):
    """
    Prétraitement des radiographies pulmonaires.
    
    IMPORTANT: Les X-rays sont souvent en niveaux de gris mais les modèles
    pré-entraînés attendent du RGB!
    """
    # Charger l'image
    img = cv2.imread(image_path)
    
    # Vérifier si grayscale
    if len(img.shape) == 2:
        # Convertir grayscale → RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        # Convertir BGR → RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Redimensionner
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)
    
    # Normaliser entre 0 et 1
    img = img.astype(np.float32) / 255.0
    
    return img

def preprocess_xray_for_model(img_array, model_name):
    """
    Preprocessing spécifique au modèle pour X-rays.
    """
    # Ajouter dimension batch
    img = np.expand_dims(img_array, axis=0)
    
    # Scale à 0-255 pour le preprocessing des modèles pré-entraînés
    img = (img * 255).astype(np.float32)
    
    if 'densenet' in model_name.lower():
        img = densenet.preprocess_input(img)
    elif 'alexnet' in model_name.lower():
        # AlexNet custom - normalisation standard ImageNet
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img / 255.0
        img = (img - mean) / std
    else:
        img = img / 255.0
    
    return img
```

## 3.3 Grad-CAM Universel

```python
import tensorflow as tf
import numpy as np
import cv2

class GradCAMExplainer:
    """
    Explainer Grad-CAM universel pour tout modèle Keras/TF.
    """
    
    def __init__(self, model, layer_name=None):
        """
        Args:
            model: Modèle Keras
            layer_name: Nom de la couche conv. Si None, trouve automatiquement.
        """
        self.model = model
        self.layer_name = layer_name or self._find_last_conv_layer()
        
    def _find_last_conv_layer(self):
        """Trouve automatiquement la dernière couche convolutionnelle."""
        for layer in reversed(self.model.layers):
            if 'conv' in layer.name.lower():
                return layer.name
        raise ValueError("Aucune couche convolutionnelle trouvée!")
    
    def explain(self, img_array, class_idx=None):
        """
        Génère la heatmap Grad-CAM.
        
        Args:
            img_array: Image preprocessée avec batch dimension (1, H, W, C)
            class_idx: Index de la classe à expliquer. None = classe prédite.
            
        Returns:
            heatmap: numpy array (H, W) normalisé entre 0 et 1
        """
        # Créer le modèle gradient
        grad_model = tf.keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[
                self.model.get_layer(self.layer_name).output,
                self.model.output
            ]
        )
        
        # Forward + Backward pass
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            
            class_output = predictions[:, class_idx]
        
        # Calculer les gradients
        grads = tape.gradient(class_output, conv_outputs)
        
        # Global Average Pooling des gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Pondérer les feature maps
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # ReLU + Normalisation
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
        
        return heatmap.numpy()
    
    def overlay(self, original_img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        Superpose la heatmap sur l'image originale.
        
        Args:
            original_img: Image originale (H, W, 3) normalisée 0-1
            heatmap: Heatmap Grad-CAM (h, w)
            alpha: Transparence (0-1)
            
        Returns:
            Image avec heatmap superposée
        """
        # Redimensionner la heatmap
        heatmap_resized = cv2.resize(
            heatmap, 
            (original_img.shape[1], original_img.shape[0])
        )
        
        # Convertir en colormap
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), 
            colormap
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Superposer
        original_uint8 = np.uint8(255 * original_img)
        superimposed = cv2.addWeighted(
            original_uint8, 1 - alpha,
            heatmap_colored, alpha,
            0
        )
        
        return superimposed
```

## 3.4 LIME pour Images (Spectrogrammes & X-rays)

```python
from lime import lime_image
from skimage.segmentation import mark_boundaries
import numpy as np

class LIMEExplainer:
    """
    Explainer LIME pour classification d'images.
    """
    
    def __init__(self, model, preprocess_fn=None):
        """
        Args:
            model: Modèle Keras avec méthode predict
            preprocess_fn: Fonction de preprocessing (optionnel)
        """
        self.model = model
        self.preprocess_fn = preprocess_fn
        self.explainer = lime_image.LimeImageExplainer()
    
    def _predict_fn(self, images):
        """Wrapper pour la prédiction compatible LIME."""
        if self.preprocess_fn:
            images = np.array([self.preprocess_fn(img) for img in images])
        return self.model.predict(images)
    
    def explain(self, image, num_samples=1000, num_features=10):
        """
        Génère l'explication LIME.
        
        Args:
            image: Image numpy (H, W, C) normalisée 0-1
            num_samples: Nombre de perturbations (plus = meilleur mais lent)
            num_features: Nombre de superpixels à afficher
            
        Returns:
            explanation: Objet LimeImageExplanation
        """
        # LIME attend des images en 0-255
        image_uint8 = np.uint8(image * 255) if image.max() <= 1 else image
        
        explanation = self.explainer.explain_instance(
            image_uint8,
            self._predict_fn,
            top_labels=2,
            hide_color=0,
            num_samples=num_samples
        )
        
        return explanation
    
    def visualize(self, explanation, image, positive_only=True, 
                  num_features=5, hide_rest=False):
        """
        Visualise l'explication LIME.
        
        Returns:
            Image avec boundaries des superpixels importants
        """
        top_label = explanation.top_labels[0]
        
        temp, mask = explanation.get_image_and_mask(
            top_label,
            positive_only=positive_only,
            num_features=num_features,
            hide_rest=hide_rest
        )
        
        # Normaliser si nécessaire
        if temp.max() > 1:
            temp = temp / 255.0
        
        # Ajouter les boundaries
        result = mark_boundaries(temp, mask, color=(1, 1, 0))
        
        return np.uint8(result * 255)
```

## 3.5 SHAP pour Images

```python
import shap
import numpy as np
import tensorflow as tf

class SHAPExplainer:
    """
    Explainer SHAP pour CNNs.
    
    ATTENTION: SHAP est TRÈS LENT pour les images!
    Utiliser GradientExplainer pour plus de rapidité.
    """
    
    def __init__(self, model, background_data):
        """
        Args:
            model: Modèle Keras
            background_data: Échantillon de données (N, H, W, C) 
                           pour calculer les valeurs de référence.
                           Utiliser 50-100 images représentatives.
        """
        self.model = model
        self.background = background_data
        
        # GradientExplainer est plus rapide que KernelExplainer pour les CNNs
        self.explainer = shap.GradientExplainer(
            model,
            background_data
        )
    
    def explain(self, image):
        """
        Calcule les valeurs SHAP.
        
        Args:
            image: Image avec batch dimension (1, H, W, C)
            
        Returns:
            shap_values: Valeurs SHAP par pixel et par classe
        """
        shap_values = self.explainer.shap_values(image)
        return shap_values
    
    def visualize(self, shap_values, image, class_idx=0):
        """
        Visualise les valeurs SHAP.
        
        Args:
            shap_values: Output de explain()
            image: Image originale
            class_idx: Classe à visualiser
            
        Returns:
            Figure matplotlib
        """
        # Pour multi-classe, shap_values est une liste
        if isinstance(shap_values, list):
            values = shap_values[class_idx]
        else:
            values = shap_values
        
        # Créer la visualisation
        shap.image_plot(
            values,
            image,
            show=False
        )
        
        return plt.gcf()
```

---

# PARTIE 4: MEGA-PROMPT POUR VIBE CODING

Copie ce prompt dans ton outil de vibe coding (Cursor, Windsurf, etc.):

---

## 🚀 MEGA-PROMPT STREAMLIT XAI PLATFORM

```
Tu dois créer une application Streamlit pour une plateforme unifiée d'Explainable AI (XAI) qui analyse à la fois des fichiers audio et des images médicales.

## CONTEXTE DU PROJET
C'est un projet académique de M2 qui fusionne deux systèmes existants:
1. Détection de deepfakes audio (VGG16, MobileNet, ResNet, Custom CNN sur spectrogrammes)
2. Détection de cancer pulmonaire sur X-rays (AlexNet, DenseNet)

## FONCTIONNALITÉS OBLIGATOIRES

### Page 1: Classification Simple
- Upload drag-and-drop de fichiers (.wav pour audio, .jpg/.png pour X-ray)
- Détection automatique du type d'input
- Sélection du modèle compatible (filtrage automatique selon le type)
- Sélection de la technique XAI (LIME, Grad-CAM, SHAP) - filtrer les incompatibles
- Affichage: résultat de classification + visualisation XAI

### Page 2: Comparaison XAI
- Même input que Page 1
- Affichage côte-à-côte de plusieurs techniques XAI
- Grille comparative: LIME | Grad-CAM | SHAP
- Indication claire de quelle méthode est utilisée

## STRUCTURE DE L'APPLICATION

```python
# app.py - Structure principale
import streamlit as st

# Configuration de la page
st.set_page_config(
    page_title="Unified XAI Platform",
    page_icon="🔬",
    layout="wide"
)

# Sidebar pour la navigation
page = st.sidebar.selectbox(
    "Navigation",
    ["🎯 Classification", "📊 Comparaison XAI", "ℹ️ À propos"]
)

# Logique de routing
if page == "🎯 Classification":
    show_classification_page()
elif page == "📊 Comparaison XAI":
    show_comparison_page()
else:
    show_about_page()
```

## CONFIGURATION DES MODÈLES ET XAI

```python
# Configuration à utiliser
MODELS = {
    "audio": {
        "VGG16": {"path": "models/audio/vgg16_audio.h5", "gradcam_layer": "block5_conv3"},
        "MobileNet": {"path": "models/audio/mobilenet_audio.h5", "gradcam_layer": "conv_pw_13_relu"},
        "ResNet": {"path": "models/audio/resnet_audio.h5", "gradcam_layer": "conv5_block3_out"},
        "Custom CNN": {"path": "models/audio/custom_cnn_audio.h5", "gradcam_layer": "conv2d_4"}
    },
    "image": {
        "AlexNet": {"path": "models/image/alexnet_xray.h5", "gradcam_layer": "conv2d_5"},
        "DenseNet": {"path": "models/image/densenet_xray.h5", "gradcam_layer": "relu"}
    }
}

XAI_METHODS = {
    "LIME": {"audio": True, "image": True, "description": "Perturbe l'image pour identifier les régions importantes"},
    "Grad-CAM": {"audio": True, "image": True, "description": "Utilise les gradients pour générer une heatmap"},
    "SHAP": {"audio": True, "image": True, "description": "Calcule les valeurs Shapley pour chaque région"}
}

CLASSES = {
    "audio": ["Real", "Fake"],
    "image": ["Benign", "Malignant"]
}
```

## UI POUR LA PAGE CLASSIFICATION

```python
def show_classification_page():
    st.title("🎯 Classification avec Explication")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📁 Upload")
        uploaded_file = st.file_uploader(
            "Glissez un fichier audio (.wav) ou image (.jpg, .png)",
            type=["wav", "jpg", "jpeg", "png"]
        )
        
        if uploaded_file:
            # Détecter le type
            file_type = detect_file_type(uploaded_file)
            st.info(f"Type détecté: {'🎵 Audio' if file_type == 'audio' else '🖼️ Image'}")
            
            # Afficher les modèles compatibles
            available_models = list(MODELS[file_type].keys())
            selected_model = st.selectbox("Modèle", available_models)
            
            # Afficher les méthodes XAI compatibles
            available_xai = [k for k, v in XAI_METHODS.items() if v[file_type]]
            selected_xai = st.selectbox("Méthode XAI", available_xai)
            
            # Bouton d'analyse
            if st.button("🔍 Analyser", type="primary"):
                with st.spinner("Analyse en cours..."):
                    result = analyze(uploaded_file, file_type, selected_model, selected_xai)
                    st.session_state.result = result
    
    with col2:
        if "result" in st.session_state:
            display_results(st.session_state.result)
```

## UI POUR LA PAGE COMPARAISON

```python
def show_comparison_page():
    st.title("📊 Comparaison des Méthodes XAI")
    
    # Upload
    uploaded_file = st.file_uploader("Upload", type=["wav", "jpg", "jpeg", "png"])
    
    if uploaded_file:
        file_type = detect_file_type(uploaded_file)
        selected_model = st.selectbox("Modèle", list(MODELS[file_type].keys()))
        
        # Sélection multiple des méthodes XAI
        available_xai = [k for k, v in XAI_METHODS.items() if v[file_type]]
        selected_xais = st.multiselect(
            "Méthodes XAI à comparer",
            available_xai,
            default=available_xai
        )
        
        if st.button("🔍 Comparer", type="primary"):
            # Afficher en grille
            cols = st.columns(len(selected_xais))
            
            for i, xai_method in enumerate(selected_xais):
                with cols[i]:
                    st.subheader(xai_method)
                    with st.spinner(f"Calcul {xai_method}..."):
                        result = analyze(uploaded_file, file_type, selected_model, xai_method)
                        st.image(result["xai_image"], caption=f"{xai_method} Explanation")
                        st.metric("Confiance", f"{result['confidence']:.2%}")
```

## FONCTIONS UTILITAIRES CRITIQUES

```python
def detect_file_type(uploaded_file):
    """Détecte si c'est audio ou image."""
    ext = uploaded_file.name.split(".")[-1].lower()
    if ext == "wav":
        return "audio"
    return "image"

def analyze(uploaded_file, file_type, model_name, xai_method):
    """
    Pipeline d'analyse complet.
    
    IMPORTANT: Cette fonction orchestre tout le processus!
    """
    # 1. Preprocessing
    if file_type == "audio":
        # Sauvegarder temporairement et convertir en spectrogramme
        temp_path = save_temp_file(uploaded_file)
        processed_input = audio_to_spectrogram(temp_path)
        display_image = processed_input.copy()
    else:
        # Charger et prétraiter l'image
        processed_input = preprocess_xray(uploaded_file)
        display_image = processed_input.copy()
    
    # 2. Classification
    model = load_model(MODELS[file_type][model_name]["path"])
    preprocessed = preprocess_for_model(processed_input, model_name)
    prediction = model.predict(preprocessed)
    class_idx = np.argmax(prediction[0])
    confidence = prediction[0][class_idx]
    class_name = CLASSES[file_type][class_idx]
    
    # 3. XAI
    gradcam_layer = MODELS[file_type][model_name]["gradcam_layer"]
    
    if xai_method == "LIME":
        explainer = LIMEExplainer(model)
        explanation = explainer.explain(processed_input)
        xai_image = explainer.visualize(explanation, processed_input)
    
    elif xai_method == "Grad-CAM":
        explainer = GradCAMExplainer(model, gradcam_layer)
        heatmap = explainer.explain(preprocessed, class_idx)
        xai_image = explainer.overlay(display_image, heatmap)
    
    elif xai_method == "SHAP":
        # Charger les données de background (précompilées)
        background = load_background_data(file_type)
        explainer = SHAPExplainer(model, background)
        shap_values = explainer.explain(preprocessed)
        xai_image = create_shap_visualization(shap_values, display_image, class_idx)
    
    return {
        "class": class_name,
        "confidence": confidence,
        "xai_image": xai_image,
        "original_image": display_image
    }
```

## STYLE ET UX

```python
# CSS custom pour une meilleure apparence
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
    .stSelectbox {
        margin-bottom: 1rem;
    }
    .result-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)
```

## REQUIREMENTS.TXT

```
streamlit>=1.28.0
tensorflow>=2.10.0
numpy>=1.21.0
opencv-python>=4.5.0
librosa>=0.9.0
lime>=0.2.0
shap>=0.41.0
scikit-image>=0.19.0
matplotlib>=3.5.0
Pillow>=9.0.0
```

## POINTS CRITIQUES À NE PAS OUBLIER

1. **Preprocessing cohérent**: Les paramètres du spectrogramme (n_mels, hop_length, etc.) DOIVENT correspondre à l'entraînement
2. **Couches Grad-CAM**: Utiliser les bons noms de couches pour chaque modèle
3. **Format des images**: LIME attend 0-255, les modèles attendent 0-1 ou preprocessed
4. **SHAP est LENT**: Prévoir un indicateur de chargement et limiter à 200 samples
5. **Cache Streamlit**: Utiliser @st.cache_data et @st.cache_resource pour les modèles
6. **Gestion d'erreurs**: Try/except autour des opérations critiques

## BONUS POUR 20/20

- Ajouter une technique XAI supplémentaire (Integrated Gradients, Attention Maps)
- Permettre le zoom sur les heatmaps
- Export PDF des résultats
- Dark mode
- Internationalisation (FR/EN)
```

---

# PARTIE 5: RÉFÉRENCES À CITER

## Sources à mentionner dans le rapport

```markdown
## Références

### Datasets
1. Fake-or-Real (FoR) Dataset - York University
   https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset

2. CheXpert Dataset - Stanford ML Group
   Irvin, J., et al. (2019). CheXpert: A Large Chest Radiograph Dataset with 
   Uncertainty Labels and Expert Comparison. AAAI Conference on AI.
   https://stanfordmlgroup.github.io/competitions/chexpert/

### Techniques XAI
3. LIME
   Ribeiro, M.T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?": 
   Explaining the Predictions of Any Classifier. KDD '16.

4. Grad-CAM
   Selvaraju, R.R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks 
   via Gradient-based Localization. ICCV 2017.

5. SHAP
   Lundberg, S.M., & Lee, S.I. (2017). A Unified Approach to Interpreting Model 
   Predictions. NeurIPS 2017.

### Repos GitHub de référence
6. Deepfake-Audio-Detection-with-XAI
   https://github.com/Guri10/Deepfake-Audio-Detection-with-XAI

7. LungCancerDetection
   https://github.com/schaudhuri16/LungCancerDetection
   (Repo conceptuel - implémentation propre requise)

### Papier associé au projet audio
8. Godse, P., et al. (2023). Deepfake audio detection and justification with 
   Explainable Artificial Intelligence (XAI). ResearchGate.
   DOI: 10.21203/rs.3.rs-3444277/v1
```

---

# PARTIE 6: CHECKLIST AVANT RENDU

```markdown
## Checklist Projet XAI - Objectif 20/20

### Fonctionnalités Obligatoires
- [ ] Interface Streamlit/Gradio fonctionnelle
- [ ] Upload audio (.wav) avec conversion spectrogramme
- [ ] Upload image (X-ray) avec preprocessing
- [ ] Au moins 2 modèles par type (audio: VGG16, MobileNet / image: AlexNet, DenseNet)
- [ ] LIME implémenté et fonctionnel
- [ ] Grad-CAM implémenté et fonctionnel
- [ ] SHAP implémenté et fonctionnel
- [ ] Filtrage automatique des méthodes XAI selon l'input
- [ ] Page de comparaison côte-à-côte

### Documentation
- [ ] README.md complet (installation, usage, démo)
- [ ] Noms des membres et groupe TD
- [ ] Rapport technique (choix de design, intégration, améliorations)
- [ ] Statement sur l'utilisation d'IA générative

### Code Quality
- [ ] Code structuré et modulaire
- [ ] Commentaires pertinents
- [ ] Gestion des erreurs
- [ ] Requirements.txt à jour

### Bonus (pour 20/20)
- [ ] Technique XAI supplémentaire
- [ ] Interactivité (zoom, toggle layers)
- [ ] Tests unitaires
- [ ] Docker support
```

---

**Ce document contient tout ce dont tu as besoin pour réaliser le projet avec succès. La partie vibe-codable est dans le mega-prompt de la Partie 4. Les parties critiques (preprocessing, Grad-CAM, LIME, SHAP) sont dans la Partie 3 et ne doivent PAS être vibe-codées mais comprises et adaptées.**

Bonne chance pour le 20/20! 🚀