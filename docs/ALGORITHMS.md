# Algorithms Implemented in Groundnut AI

This document provides a comprehensive overview of all algorithms implemented in the project.

## 1. Data Augmentation

### Mosaic & Mixup Augmentation
**Location**: `ml/augmentation/configs/mosaic_mixup.json`, `ml/augmentation/src/transforms.py`

**Purpose**: Forces the model to learn overlapping leaf edges, occlusion, and varied light exposures.

**Implementation**:
- **Mosaic**: Combines 4 images into a mosaic grid by dividing the image into quadrants and applying photometric variations (brightness, contrast, hue, saturation) to each quadrant
- **Mixup**: Blends an image with a slightly transformed version using lambda sampled from Beta(alpha, alpha) distribution

**Usage**:
```python
from ml.augmentation.src.pipeline import AugmentationPipeline
from ml.augmentation.src.transforms import op_mosaic, op_mixup

# Load mosaic_mixup config
pipeline = AugmentationPipeline(config, seed=42)
augmented_image = pipeline.apply(image)
```

## 2. Synthesis

### CycleGAN (Cycle-Consistent Generative Adversarial Network)
**Location**: `ml/synthesis/cyclegan.py`

**Purpose**: Converts normal crop images to weed-infested scenarios without needing paired training data.

**Implementation**:
- **Generator G**: Transforms domain A (normal crops) to domain B (weed-infested)
- **Generator F**: Transforms domain B to domain A (inverse mapping)
- **Discriminator D_A**: Classifies real vs fake images in domain A
- **Discriminator D_B**: Classifies real vs fake images in domain B
- **Cycle Consistency Loss**: Ensures F(G(A)) ≈ A and G(F(B)) ≈ B
- **Identity Loss**: Preserves color and structure when translating similar images

**Architecture**:
- U-Net style generator with 9 residual blocks
- PatchGAN discriminator for realistic texture generation

**Training**:
```bash
python ml/synthesis/cyclegan.py --domain_a path/to/normal_crops --domain_b path/to/weed_infested --epochs 200
```

## 3. Preprocessing

### Gaussian Filtering + ExG Thresholding
**Location**: `ml/preprocessing/vegetation_isolation.py`

**Purpose**: Instantly scrubs sensor noise and isolates vegetation from bare dirt.

**Implementation**:
- **Gaussian Filtering**: Removes sensor noise while preserving edges
  - Kernel size: 5x5 (configurable)
  - Sigma: 1.0 (configurable)
- **Excess Green (ExG) Index**: Separates vegetation from soil
  - Formula: ExG = 2*G - R - B
  - Thresholding: Otsu's method for automatic threshold selection

**Usage**:
```python
from ml.preprocessing.vegetation_isolation import preprocess_image

result = preprocess_image(
    image,
    gaussian_kernel=5,
    gaussian_sigma=1.0,
    use_otsu=True
)

# Access results
filtered_image = result["filtered"]
exg_map = result["exg"]
vegetation_mask = result["vegetation_mask"]
```

## 4. Semantic & Instance Segmentation

### YOLOv8-Seg / YOLO11-Seg
**Location**: `ml/segmentation/train_yoloseg.py`

**Purpose**: Combines both semantic and instance segmentation into a single, high-speed, pixel-precise mask engine.

**Implementation**:
- **YOLOv8-Seg**: YOLOv8 with segmentation head
- **YOLO11-Seg**: Latest YOLO11 with improved segmentation
- Supports both semantic (pixel-level classification) and instance (object-level masks) segmentation
- Real-time performance on edge devices

**Training**:
```bash
python ml/segmentation/train_yoloseg.py --model yolov8n-seg.pt --epochs 100 --batch 16
python ml/segmentation/train_yoloseg.py --model yolo11n-seg.pt --epochs 100 --batch 16
```

**Inference**:
```python
from ultralytics import YOLO

model = YOLO("path/to/yolov8n-seg.pt")
results = model("image.jpg")

# Access segmentation masks
for result in results:
    masks = result.masks  # Segmentation masks
    boxes = result.boxes  # Bounding boxes
```

## 5. Indexing

### NDVI (Normalized Difference Vegetation Index)
**Location**: `ml/indexing/vegetation_indices.py`

**Purpose**: Quantifies vegetation health and density using multispectral imagery.

**Formula**: NDVI = (NIR - Red) / (NIR + Red)

**Range**: -1 to 1
- Values close to 1: Healthy, dense vegetation
- Values close to 0: Bare soil or sparse vegetation
- Negative values: Water, snow, or non-vegetative features

**Usage**:
```python
from ml.indexing.vegetation_indices import calculate_ndvi

ndvi = calculate_ndvi(nir_band, red_band)
```

### ExG (Excess Green Index)
**Location**: `ml/indexing/vegetation_indices.py`

**Purpose**: Vegetation detection for standard RGB cameras.

**Formula**: ExG = 2*G - R - B

**Usage**:
```python
from ml.indexing.vegetation_indices import calculate_exg

exg = calculate_exg(rgb_image)
```

**Additional Indices**:
- **ExGR**: ExG - 1.4*(R - B) - Improved separation of vegetation from soil

## 6. Classification

### Single Module Models
**Location**: `ml/models/registry.py`, `ml/classification/train_classifier.py`

**Implemented Models**:
1. **CNN**: Custom convolutional neural network
2. **ResNet50**: Residual network with 50 layers
3. **VGG16**: Visual Geometry Group 16-layer network
4. **MobileNetV2**: Lightweight mobile architecture
5. **EfficientNet-B0**: Compound scaling for efficiency

**Training**:
```bash
python ml/classification/train_classifier.py --model resnet50 --epochs 30
python ml/classification/train_classifier.py --all-models
```

### Hybrid Module Models
**Location**: `ml/models/hybrid.py`

**Implemented Hybrid Models**:
1. **ResNet50 + MobileNet**: Combines powerful features with efficiency
2. **ResNet50 + EfficientNet**: Residual learning with compound scaling
3. **ResNet50 + EfficientNet + MobileNet**: Three-way ensemble leveraging all strengths

**Architecture**:
- Feature extraction from multiple backbones
- Feature fusion via concatenation
- Shared classification head with dropout

**Training**:
```bash
python ml/classification/train_classifier.py --model resnet50_mobilenet --epochs 30
python ml/classification/train_classifier.py --model resnet50_efficientnet --epochs 30
python ml/classification/train_classifier.py --model resnet50_efficientnet_mobilenet --epochs 30
```

## 7. Detection

### YOLOv8
**Location**: `ml/detection/train_yolo.py`

**Purpose**: Real-time, high-accuracy bounding box detection on agricultural edge-computing devices.

**Model**: YOLOv8n (nano variant for edge deployment)

**Performance**:
- Test mAP@50: 0.8651
- Test mAP@50-95: 0.6957
- F1 Score: 0.8400
- Model Size: 6.25 MB
- Inference Speed: ~88.6 FPS

**Training**:
```bash
python ml/detection/train_yolo.py --epochs 100 --batch 16 --device 0
```

**Augmentation**:
- Mosaic: 1.0 probability
- Mixup: 0.0 probability (configurable)
- HSV augmentation
- Geometric transforms (rotation, scale, flip)

## 8. Herbicide Recommendation

### Fuzzy Logic Rule Engine
**Location**: `herbicide/database/recommendation_engine.py`

**Purpose**: Translates multi-variable field data—such as detected weed species, density, and growth stage—into precise chemical dosages without relying on a black-box AI model.

**Implementation**:
- Knowledge base with validated herbicide records
- Multi-criteria filtering:
  - Crop compatibility (groundnut)
  - Weed species/category matching
  - Growth stage alignment
  - Regional/seasonal considerations
  - Confidence ranking (official regulatory > research bulletin > extension)
- Returns only validated recommendations (validation_status: "passed")

**Usage**:
```python
from herbicide.database.recommendation_engine import HerbicideRecommendationEngine, RecommendationInput

engine = HerbicideRecommendationEngine(knowledge_base_path)
input_data = RecommendationInput(
    crop="groundnut",
    weed_species="specific_weed",
    weed_density="high",
    crop_growth_stage="post_emergence"
)
recommendation = engine.generate_recommendation(input_data)
```

## 9. Feature Extraction

### ResNet Residual Backbones
**Location**: `ml/gate/features.py`, `ml/models/registry.py`

**Purpose**: Automatically extracts hierarchical plant features like vein structure and leaf texture while preventing gradient vanishing.

**Implementation**:
- **ResNet50**: 50-layer residual network with skip connections
- **ImageNet Pretrained**: Leverages transfer learning from large-scale dataset
- **Hierarchical Features**: Extracts features at multiple scales (low-level edges to high-level semantics)

**Features Extracted**:
- HSV color statistics (green vegetation, soil/tan, vivid colors)
- Texture (RMS gradient detail)
- Colorfulness and brightness
- ImageNet-based plant-likeness signals

**Usage**:
```python
from ml.gate.features import ImageNetFeaturizer, extract_features

featurizer = ImageNetFeaturizer(name="resnet50")
features = extract_features(rgb_image, detections, image_size, featurizer)
```

## 10. Explainable AI (XAI)

### Grad-CAM (Gradient-weighted Class Activation Mapping)
**Location**: `ml/xai/gradcam.py`

**Purpose**: Generates an intuitive, real-time visual heatmap over the exact leaf structures that triggered the weed classification.

**Implementation**:
- Registers forward/backward hooks on the last convolutional layer
- Computes gradients of target class score w.r.t. feature maps
- Weights feature maps by mean gradient
- Applies ReLU and upscales to input resolution
- Returns gradient-computed heatmap (falls back to activation map if gradient path fails)

**Usage**:
```python
from ml.xai.gradcam import GradCAMExplainer

explainer = GradCAMExplainer(yolo_model)
heatmap, method = explainer.generate_cam(image_path, class_id=1)

# Visualize
import cv2
overlay = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
result = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
```

## Configuration Files

All algorithm configurations are stored in `configs/` directory:
- `mosaic_mixup.json`: Mosaic & Mixup augmentation config
- `cnn.yaml`: CNN classifier config
- `vgg16.yaml`: VGG16 classifier config
- `resnet50.yaml`: ResNet50 classifier config
- `mobilenet.yaml`: MobileNet classifier config
- `efficientnet.yaml`: EfficientNet classifier config
- `resnet50_mobilenet.yaml`: ResNet50+MobileNet hybrid config
- `resnet50_efficientnet.yaml`: ResNet50+EfficientNet hybrid config
- `resnet50_efficientnet_mobilenet.yaml`: Three-way hybrid config

## References

1. **Mosaic & Mixup**: YOLOv4, YOLOv5 training strategies
2. **CycleGAN**: Zhu et al., "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks", ICCV 2017
3. **ExG Index**: Woebbecke et al., "Color indices for weed identification under various soil, residue, and lighting conditions", 1995
4. **NDVI**: Rouse et al., "Monitoring the vernal advancement and retrogradation (greenwave effect) of natural vegetation", 1974
5. **YOLOv8**: Ultralytics YOLO implementation
6. **YOLO11**: Latest YOLO architecture
7. **ResNet**: He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
8. **Grad-CAM**: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", ICCV 2017
