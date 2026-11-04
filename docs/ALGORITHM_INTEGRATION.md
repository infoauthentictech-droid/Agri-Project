# Algorithm Integration Guide

This document explains how all algorithms in the Groundnut AI project are connected and work together in the complete pipeline.

## Complete Algorithm Pipeline

```
User Input Image
    ↓
┌─────────────────────────────────────────┐
│ 1. Plant Gate (Feature Extraction)      │
│    - ResNet backbone features           │
│    - HSV color statistics               │
│    - Texture analysis                   │
│    - Reject non-plant images           │
└─────────────────────────────────────────┘
    ↓ (if plant detected)
┌─────────────────────────────────────────┐
│ 2. Preprocessing                       │
│    - Gaussian Filtering (noise removal) │
│    - ExG Thresholding (vegetation iso)  │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 3. Detection (YOLOv8)                  │
│    - Bounding box detection             │
│    - Groundnut vs Weed classification   │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 4. Segmentation (YOLOv8-Seg)           │
│    - Pixel-precise masks                │
│    - Semantic + Instance segmentation   │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 5. Classification (Hybrid Models)       │
│    - Single: CNN, ResNet50, etc.       │
│    - Hybrid: ResNet50+MobileNet, etc.  │
│    - Detailed species classification    │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 6. Indexing (NDVI/ExG)                 │
│    - Vegetation health assessment       │
│    - Density mapping                    │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 7. XAI (Grad-CAM)                      │
│    - Visual explanations                │
│    - Heatmap generation                 │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 8. Herbicide Recommendation            │
│    - Fuzzy Logic Rule Engine            │
│    - Multi-criteria filtering          │
│    - Validated recommendations          │
└─────────────────────────────────────────┘
```

## Service Integration

### Backend Services Architecture

```
backend/app/
├── services/
│   ├── classifier_service.py      # Classification (Single + Hybrid)
│   ├── segmentation_service.py     # YOLO-Seg segmentation
│   ├── xai_service.py             # Grad-CAM explanations
│   ├── recommendation_service.py   # Herbicide recommendations
│   └── model_service.py           # Model loading/management
├── utils/
│   ├── plant_gate.py              # Plant detection gate
│   ├── density.py                 # Weed density calculation
│   └── security.py                # Input validation
└── api/
    └── endpoints.py              # API endpoints integrating all services
```

### 1. Plant Gate Integration

**File**: `backend/app/utils/plant_gate.py`

**Purpose**: First line of defense - rejects non-plant images before processing.

**Algorithm Used**:
- ResNet50 backbone for feature extraction
- HSV color statistics
- Texture analysis (RMS gradient)
- Random Forest classifier

**Integration Point**:
```python
# In endpoints.py - before analysis
from backend.app.utils.plant_gate import is_plant_image

if not is_plant_image(image_data):
    return {"error": "not_plant_image", "message": "Please upload a plant image"}
```

### 2. Preprocessing Integration

**File**: `ml/preprocessing/vegetation_isolation.py`

**Purpose**: Clean input data before detection/segmentation.

**Algorithm Used**:
- Gaussian Filtering (kernel=5, sigma=1.0)
- ExG Thresholding (Otsu's method)

**Integration Point**:
```python
# In analysis pipeline
from ml.preprocessing.vegetation_isolation import preprocess_image

result = preprocess_image(image)
filtered_image = result["filtered"]
vegetation_mask = result["vegetation_mask"]
```

### 3. Detection Integration

**File**: `backend/app/services/model_service.py`

**Purpose**: Detect groundnut vs weed in images.

**Algorithm Used**: YOLOv8n

**Integration Point**:
```python
# In endpoints.py
from backend.app.services.model_service import ModelService

model_service = ModelService()
detections = model_service.detect(image_data)
```

### 4. Segmentation Integration

**File**: `backend/app/services/segmentation_service.py` (Updated to use YOLO-Seg)

**Purpose**: Generate pixel-precise masks for detected objects.

**Algorithm Used**: YOLOv8-Seg / YOLO11-Seg

**Integration Point**:
```python
# In endpoints.py
from backend.app.services.segmentation_service import SegmentationService

seg_service = SegmentationService()
segmentation = seg_service.segment(image_data)
```

### 5. Classification Integration

**File**: `backend/app/services/classifier_service.py`

**Purpose**: Detailed classification of detected patches.

**Algorithms Used**:
- Single: CNN, ResNet50, VGG16, MobileNet, EfficientNet
- Hybrid: ResNet50+MobileNet, ResNet50+EfficientNet, ResNet50+EfficientNet+MobileNet

**Integration Point**:
```python
# In endpoints.py
from backend.app.services.classifier_service import ClassifierService

classifier = ClassifierService()
# Use single model
result = classifier.classify(image_data, model_name="resnet50")
# Use hybrid model
result = classifier.classify(image_data, model_name="resnet50_efficientnet")
```

### 6. Indexing Integration

**File**: `ml/indexing/vegetation_indices.py`

**Purpose**: Assess vegetation health and density.

**Algorithms Used**:
- NDVI (for multispectral)
- ExG/ExGR (for RGB)

**Integration Point**:
```python
# In analysis pipeline
from ml.indexing.vegetation_indices import calculate_exg, classify_vegetation_health

exg = calculate_exg(image)
classification, health_stats = classify_vegetation_health(exg, "exg")
```

### 7. XAI Integration

**File**: `backend/app/services/xai_service.py`

**Purpose**: Provide visual explanations for predictions.

**Algorithm Used**: Grad-CAM

**Integration Point**:
```python
# In endpoints.py
from backend.app.services.xai_service import XAIService

xai_service = XAIService()
explanation = xai_service.explain(image_data, detection_result)
```

### 8. Herbicide Recommendation Integration

**File**: `backend/app/services/recommendation_service.py`

**Purpose**: Generate herbicide recommendations based on field data.

**Algorithm Used**: Fuzzy Logic Rule Engine

**Integration Point**:
```python
# In endpoints.py
from backend.app.services.recommendation_service import RecommendationService

rec_service = RecommendationService()
recommendation = rec_service.get_recommendation(
    crop="groundnut",
    weed_species="detected_weed",
    weed_density="high",
    growth_stage="post_emergence"
)
```

## Data Flow Between Algorithms

### Complete Analysis Request Flow

```python
# 1. User uploads image
image_data = request.json["image"]

# 2. Plant Gate - Check if image contains plants
if not plant_gate.is_plant(image_data):
    return error_response("not_plant_image")

# 3. Preprocessing - Clean image
preprocessed = preprocessing_service.preprocess(image_data)

# 4. Detection - Find groundnut and weed objects
detections = detection_service.detect(preprocessed)

# 5. Segmentation - Get pixel-precise masks
segmentation = segmentation_service.segment(preprocessed)

# 6. Classification - Detailed species classification
# Extract patches from detections
patches = extract_patches(preprocessed, detections)
classifications = []
for patch in patches:
    # Use hybrid model for best accuracy
    result = classifier_service.classify(patch, model_name="resnet50_efficientnet")
    classifications.append(result)

# 7. Indexing - Assess vegetation health
exg_index = indexing_service.calculate_exg(preprocessed)
health_stats = indexing_service.classify_vegetation_health(exg_index)

# 8. XAI - Generate explanations
explanations = []
for i, detection in enumerate(detections):
    explanation = xai_service.explain(preprocessed, detection)
    explanations.append(explanation)

# 9. Herbicide Recommendation - Based on all above data
recommendation = recommendation_service.get_recommendation(
    crop="groundnut",
    weed_species=classifications[0]["class_name"],
    weed_density=calculate_density(detections),
    growth_stage=determine_growth_stage(health_stats)
)

# 10. Return complete response
return {
    "detections": detections,
    "segmentation": segmentation,
    "classifications": classifications,
    "vegetation_health": health_stats,
    "explanations": explanations,
    "recommendation": recommendation
}
```

## Model Path Configuration

All model paths are configured in the backend services:

### Classification Models
```python
# backend/app/services/classifier_service.py
experiment_root = project_root / "experiments" / "classification"
# Models: cnn, resnet50, mobilenet, efficientnet, resnet50_mobilenet, etc.
```

### Detection Model
```python
# backend/app/services/model_service.py
model_path = project_root / "experiments" / "yolo" / "agri_yolov8n" / "weights" / "best.pt"
```

### Segmentation Model
```python
# backend/app/services/segmentation_service.py
model_path = project_root / "experiments" / "yoloseg" / "agri_yolov8n_seg" / "weights" / "best.pt"
```

### Plant Gate Model
```python
# backend/app/utils/plant_gate.py
model_path = project_root / "backend" / "app" / "utils" / "plant_gate_model.joblib"
```

## API Endpoints Integration

All algorithms are exposed through the FastAPI backend:

```
POST /analyze
  ├─ Plant Gate check
  ├─ Preprocessing
  ├─ Detection (YOLO)
  ├─ Segmentation (YOLO-Seg)
  ├─ Classification (Hybrid)
  ├─ Indexing (ExG/NDVI)
  ├─ XAI (Grad-CAM)
  └─ Herbicide Recommendation

GET /models
  └─ List available classification models

POST /classify
  └─ Classification only endpoint

POST /segment
  └─ Segmentation only endpoint

POST /recommend
  └─ Herbicide recommendation only endpoint

POST /explain
  └─ XAI explanation only endpoint
```

## Training Workflow Integration

### Step 1: Prepare Datasets
```bash
# Prepare detection dataset
python ml/data/prepare_detection_dataset.py

# Prepare classification dataset
python ml/data/prepare_classification_dataset.py

# Prepare segmentation dataset (for YOLO-Seg)
python ml/data/prepare_segmentation_dataset.py
```

### Step 2: Train Models
```bash
# Train all models
.\train_all_models.ps1

# Or train individually
# See docs/TRAINING_GUIDE.md
```

### Step 3: Evaluate Models
```bash
# Comprehensive evaluation
python ml/evaluation/evaluate_all_models.py
```

### Step 4: Deploy Models
```bash
# Models are automatically loaded by backend services
# No manual deployment needed - just ensure:
# - experiments/classification/ contains trained classifiers
# - experiments/yolo/agri_yolov8n/weights/best.pt exists
# - experiments/yoloseg/agri_yolov8n_seg/weights/best.pt exists
# - backend/app/utils/plant_gate_model.joblib exists
```

## Algorithm Dependencies

### Required Packages
```
torch
torchvision
ultralytics
opencv-python
pillow
numpy
scikit-learn
fastapi
pyyaml
```

### Algorithm-Specific Dependencies
- **CycleGAN**: `torch`, `torchvision` (for GAN training)
- **YOLOv8/YOLO-Seg**: `ultralytics`
- **Grad-CAM**: `torch`, `opencv-python`
- **Plant Gate**: `scikit-learn` (Random Forest)
- **NDVI**: `numpy`, `opencv-python`

## Performance Optimization

### GPU Utilization
All GPU-accelerated models automatically use CUDA if available:
- Classification: `device=0` in training scripts
- Detection: `device=0` in YOLO
- Segmentation: `device=0` in YOLO-Seg
- Plant Gate: Uses ResNet50 on GPU

### Memory Management
- Hybrid models use smaller batch sizes (8-16)
- Single models use larger batch sizes (32)
- Detection/Segmentation use 640x640 resolution

### Inference Optimization
- Models loaded once and cached
- Batch processing for multiple images
- Lazy loading for on-demand models

## Troubleshooting Integration Issues

### Model Not Found
```bash
# Check if model files exist
ls experiments/classification/
ls experiments/yolo/agri_yolov8n/weights/
ls experiments/yoloseg/agri_yolov8n_seg/weights/
ls backend/app/utils/plant_gate_model.joblib
```

### Service Not Loading Model
```python
# Check backend logs for loading errors
# Common issues:
# - Wrong model path
# - Corrupted checkpoint file
# - Missing dependencies
```

### Algorithm Not Producing Results
```bash
# Test individual algorithms
python -c "from ml.preprocessing.vegetation_isolation import preprocess_image; print('OK')"
python -c "from ml.indexing.vegetation_indices import calculate_exg; print('OK')"
python -c "from ml.models.hybrid import build_hybrid_model; print('OK')"
```

## Complete Training Commands Summary

### Quick Start (All Models)
```powershell
cd "D:\Flutter\PHD PROJECT\Agri Project"
.\train_all_models.ps1
```

### Individual Training
```powershell
# Classification
.venv-gpu\Scripts\python.exe ml/classification/train_classifier.py --model resnet50 --epochs 30 --device 0

# Hybrid Classification
.venv-gpu\Scripts\python.exe ml/classification/train_classifier.py --model resnet50_efficientnet --epochs 30 --device 0

# Detection
.venv-gpu\Scripts\python.exe ml/detection/train_yolo.py --epochs 100 --device 0

# Segmentation
.venv-gpu\Scripts\python.exe ml/segmentation/train_yoloseg.py --model yolov8n-seg.pt --epochs 100 --device 0

# Plant Gate
.venv-gpu\Scripts\python.exe ml/gate/train_plant_gate.py --device 0
```

### Evaluation
```powershell
.venv-gpu\Scripts\python.exe ml/evaluation/evaluate_all_models.py
```

## Verification Checklist

After training, verify all algorithms are connected:

- [ ] Plant gate model loads successfully
- [ ] Classification models discovered and loadable
- [ ] Detection model loads and runs inference
- [ ] Segmentation model loads and runs inference
- [ ] Preprocessing functions work correctly
- [ ] Indexing functions calculate indices correctly
- [ ] XAI service generates explanations
- [ ] Herbicide recommendations generate correctly
- [ ] API endpoints return integrated results
- [ ] Frontend displays all algorithm outputs

## Next Steps

1. **Train all models** using `train_all_models.ps1`
2. **Verify integration** by testing API endpoints
3. **Test complete pipeline** with sample images
4. **Monitor performance** and optimize as needed
5. **Deploy** following deployment guide
