# Canonical Pipeline Documentation

This document defines the canonical (official) serving paths for each task in AgriVision to prevent workflow confusion and ensure consistent behavior.

## Overview

AgriVision has multiple models and workflows. This document identifies the canonical production pipeline for each task.

## Canonical Serving Paths

### 1. Image Classification (Crop vs Weed)

**Task**: Classify whether an image contains a crop or weed

**Canonical Model**: `mobilenet` (MobileNetV2)
- **Path**: `experiments/classification/mobilenet/best_model.pth`
- **Architecture**: MobileNetV2
- **Input Size**: 128x128
- **Default Threshold**: N/A (classification uses softmax)
- **Status**: Validated (test accuracy: 96.59%)

**Alternative Model**: `efficientnet` (EfficientNet-B0)
- **Path**: `experiments/classification/efficientnet/best_model.pth`
- **Status**: Validated (test accuracy: 98.01%)
- **Use Case**: Higher accuracy required, can tolerate slower inference

**API Endpoint**: `POST /api/v1/classify`
**Service**: `backend/app/services/classifier_service.py`
**Schema**: `backend/app/schemas/analysis.py` (ClassifyRequest, ClassifyResponse)

**Preprocessing**:
1. Decode base64 image
2. Resize to model input size (128x128)
3. Normalize with ImageNet mean/std
4. Convert to tensor (C, H, W)

**Inference**: Single forward pass through classifier

**Postprocessing**:
1. Apply softmax to logits
2. Get top-k classes
3. Return class name, confidence, and probabilities

---

### 2. Object Detection (Crop and Weed Bounding Boxes)

**Task**: Detect crop and weed instances in full field images

**Canonical Model**: `yolov8n` (YOLOv8n)
- **Path**: `experiments/yolo/agri_yolov8n/weights/best.pt`
- **Architecture**: YOLOv8n
- **Input Size**: 640x640
- **Default Confidence Threshold**: 0.25
- **Default IoU Threshold**: 0.45
- **Status**: Validated

**API Endpoint**: `POST /api/v1/detect`
**Service**: `backend/app/services/model_service.py`
**Schema**: `backend/app/schemas/detection.py` (DetectionRequest, DetectionResponse)

**Preprocessing**:
1. Decode base64 image
2. Resize to input size (640x640) with letterboxing
3. No normalization (YOLO handles internally)

**Inference**: YOLO forward pass with NMS

**Postprocessing**:
1. Filter by confidence threshold
2. Apply NMS with IoU threshold
3. Convert normalized coordinates to absolute
4. Return detections with class names and bounding boxes

---

### 3. Semantic Segmentation (Pixel-Level Plant Masks)

**Task**: Generate pixel-precise masks for crop and weed regions

**Canonical Model**: `yolov8n-seg` (YOLOv8n-Seg)
- **Path**: `experiments/yoloseg/agri_yolov8n_seg/weights/best.pt`
- **Architecture**: YOLOv8n-Seg
- **Input Size**: 640x640 (configurable: 64-1024)
- **Default Confidence Threshold**: 0.25
- **Default IoU Threshold**: 0.7
- **Status**: Validated

**API Endpoint**: `POST /api/v1/segment`
**Service**: `backend/app/services/segmentation_service.py`
**Schema**: `backend/app/schemas/segmentation.py` (SegmentationRequest, SegmentationResponse)

**Preprocessing**:
1. Decode base64 image
2. Resize to target size (default 640x640)
3. No normalization (YOLO handles internally)

**Inference**: YOLO-Seg forward pass with mask generation

**Postprocessing**:
1. Resize masks to original image dimensions
2. Build semantic mask (0=background, 1=crop, 2=weed)
3. Build RGB visualization mask
4. Calculate coverage percentages
5. Encode masks as base64 PNG

---

### 4. Plant Gate (Plant vs Non-Plant Filter)

**Task**: Filter out non-plant images before expensive detection

**Canonical Model**: `plant_gate` (Random Forest)
- **Path**: `backend/app/utils/plant_gate_model.joblib`
- **Architecture**: RandomForestClassifier
- **Input Size**: 256x256 (longest side)
- **Features**: ImageNet features (ResNet50 backbone)
- **Default Threshold**: 0.30 (selected on validation set)
- **Status**: Validated (with proper train/val/test split)

**Service**: `backend/app/utils/plant_gate.py`
**Note**: Applied before detection in the pipeline

**Preprocessing**:
1. Resize image (longest side = 256)
2. Extract ImageNet features using ResNet50
3. Create feature vector

**Inference**: Random Forest prediction

**Postprocessing**:
1. Apply threshold to probability
2. Return plant (pass) or non-plant (reject)

---

### 5. Herbicide Recommendation

**Task**: Recommend herbicide based on detected weed and crop

**Knowledge Base**: `herbicide/processed/herbicide_dataset_normalized.json`
**Engine**: `herbicide/database/recommendation_engine.py`
**Service**: `backend/app/services/recommendation_service.py`
**Safety Checks**: `backend/app/utils/recommendation_safety.py`

**API Endpoint**: `POST /api/v1/recommend`
**Schema**: `backend/app/schemas/recommendation.py` (RecommendationRequest, RecommendationResponse)

**Input**:
- Crop type
- Weed species (from detection)
- Weed category
- Crop growth stage
- Season
- Region
- Weed density
- Application category (pre-sowing, pre-emergence, post-emergence)
- Detection confidence (for safety checks)

**Safety Checks**:
1. Species identification confidence
2. Crop compatibility
3. Regional registration
4. Required fields present
5. Dosage and application information available
6. Expert review flag for uncertain cases

**Output**:
- Active ingredient
- Application information (dose, method, timing)
- Warnings
- Safety status (safe/uncertain/unsafe/needs_expert_review)

---

## Complete Inference Pipeline

### Image Analysis Flow

```
1. User uploads image
   ↓
2. Input validation (security checks)
   ↓
3. Plant gate (optional, can be skipped)
   - If non-plant: reject with error
   - If plant: continue
   ↓
4. Detection (YOLO)
   - Returns bounding boxes and classes
   ↓
5. Classification (on top detection crop)
   - Returns class name and confidence
   ↓
6. Segmentation (optional, user-triggered)
   - Returns pixel masks
   ↓
7. Herbicide recommendation (if weed detected)
   - Returns safe recommendation with safety checks
   ↓
8. XAI explanation (optional, user-triggered)
   - Returns Grad-CAM heatmap
```

### Density Estimation

**Module**: `ml/density_estimator.py`
**Status**: Experimental - requires independent validation before production use

**Purpose**: Estimate weed density from detection results

**Method**:
1. Calculate bounding box areas
2. Sum areas for weed class
3. Divide by total image area
4. Categorize as low/medium/high density

**Note**: This is currently marked as experimental and should not be used for agronomic decisions without field validation.

---

## Model Registry

All models are registered in `backend/app/utils/model_registry.py`

**Registry Contains**:
- Model name and task
- Checkpoint path
- Architecture
- Preprocessing parameters
- Class mapping
- Evaluation status
- Test metrics
- Version information

**Use Cases**:
- Model discovery in classifier service
- Validation of model paths
- Tracking model versions
- Preventing model-task confusion

---

## Dataset Manifests

All datasets are tracked in `ml/data/dataset_manifest.py`

**Manifest Contains**:
- Image IDs and original filenames
- Source and capture session
- Split assignment (train/val/test)
- Labels and annotations
- File hashes for duplicate detection
- Annotation provenance

**Use Cases**:
- Leakage detection (session spanning splits)
- Duplicate detection
- Dataset lineage tracking
- Reproducible splits

---

## Training Pipeline

### Classification Training

**Script**: `ml/training/train_classifier.py`
**Canonical Configuration**:
- Data: `datasets/processed/classification_groundnut_weed/`
- Models: mobilenet, efficientnet, resnet50, vgg16, cnn
- Image size: 128
- Epochs: 30
- Batch size: 32
- Validation split: 20%
- Test split: Separate from validation

**Output**: `experiments/classification/<model_name>/best_model.pth`

### Detection Training

**Script**: `ml/detection/train_yolo.py`
**Canonical Configuration**:
- Data: `datasets/processed/groundnut_weed_detection/data.yaml`
- Model: yolov8n.pt
- Image size: 640
- Epochs: 100
- Batch size: 16
- Patience: 20 (early stopping)
- Augmentation: Enabled

**Output**: `experiments/yolo/agri_yolov8n/weights/best.pt`

### Segmentation Training

**Script**: `ml/segmentation/train_yoloseg.py`
**Canonical Configuration**:
- Data: `datasets/processed/groundnut_weed_segmentation/data.yaml`
- Model: yolov8n-seg.pt
- Image size: 640
- Epochs: 100
- Batch size: 16
- Patience: 20

**Output**: `experiments/yoloseg/agri_yolov8n_seg/weights/best.pt`

### Plant Gate Training

**Script**: `ml/gate/train_plant_gate.py`
**Canonical Configuration**:
- Data: Combined from detection and classification datasets
- Feature extractor: ResNet50 ImageNet features
- Model: RandomForest (400 trees, max_depth=18)
- Test split: 20%
- Validation split: 20% of remaining
- Threshold selection: On validation set
- Final evaluation: On test set only

**Output**: `backend/app/utils/plant_gate_model.joblib`

---

## Configuration Files

### Backend Configuration

**Main Config**: `backend/app/main.py`
- CORS settings
- API key validation
- Body size limits
- Middleware (GZip, security headers)
- Service initialization

**Environment Variables**:
- `GROUNDNUT_PRELOAD_MODELS`: Preload models on startup (default: 1)
- `GROUNDNUT_DISABLE_DOCS`: Disable API docs (default: 0)
- `GROUNDNUT_CORS_ORIGINS`: Allowed CORS origins
- `GROUNDNUT_CORS_ORIGIN_REGEX`: CORS origin regex pattern

### Dataset Configuration

**Detection Data**: `datasets/processed/groundnut_weed_detection/data.yaml`
- Path: Relative (uses `.` for portability)
- Splits: train, val, test
- Classes: crop (0), weed (1)

**Segmentation Data**: `datasets/processed/groundnut_weed_segmentation/data.yaml`
- Path: Relative
- Splits: train/images, val/images, test/images
- Classes: crop (0), weed (1)

---

## Security and Validation

### Input Validation

**Module**: `backend/app/utils/input_validation.py`

**Checks**:
- Base64 encoding validity
- File size limits (10 MB max)
- Image dimensions (32-4096 pixels)
- Format validation (JPEG, PNG, WEBP)
- Decompression bomb detection
- Truncated file detection
- Color space validation

### Safety Checks

**Module**: `backend/app/utils/recommendation_safety.py`

**Herbicide Safety**:
- Species identification confidence
- Crop compatibility
- Regional registration
- Required fields validation
- Dosage/application information validation
- Expert review flagging

---

## Testing

### Unit Tests

**Location**: `tests/`
- `test_ai_services.py`: Model loading and inference
- `test_ml_regression.py`: Regression tests for model outputs
- `test_input_validation.py`: Input validation tests

### Integration Tests

**Required** (not yet implemented):
- Flutter → Backend → Model → Response → UI
- Complete user journey testing

### ML Regression Tests

**Location**: `tests/test_ml_regression.py`

**Tests**:
- Model output consistency (class labels, confidence ranges)
- Detection count validation
- Mask shape validation
- Model registry consistency
- Dataset manifest consistency
- Herbicide safety checks

---

## Deployment

### Docker Configuration

**File**: `Dockerfile` (in project root)
- Base image: Python 3.10+
- Dependencies: `requirements.txt`
- Model artifacts: Mounted at runtime (not baked in)
- Environment: Production settings

### Environment Setup

**Virtual Environment**: `.venv-gpu/`
- Python 3.10+
- GPU support (CUDA)
- All ML dependencies

**Activation**:
```bash
.venv-gpu/Scripts/Activate.ps1  # Windows
source .venv-gpu/bin/activate  # Linux/Mac
```

---

## Troubleshooting

### Model Loading Issues

**Symptom**: Model not found
**Solution**: 
1. Check model registry: `backend/app/utils/model_registry.json`
2. Verify checkpoint paths exist
3. Run `python -c "from backend.app.utils.model_registry import get_registry; print(get_registry().validate_model_paths())"`

### Dataset Path Issues

**Symptom**: Dataset not found
**Solution**:
1. Check data.yaml uses relative paths (`path: .`)
2. Verify dataset structure matches expected format
3. Run dataset manifest audit

### Safety Check Failures

**Symptom**: Recommendation marked unsafe
**Solution**:
1. Check safety metadata in response
2. Verify crop compatibility in knowledge base
3. Ensure all required fields are provided
4. If uncertain, review with agronomist

---

## Version History

### v1.0 (2024-10-26)
- Fixed segmentation endpoint contract
- Fixed plant-gate training with proper train/val/test split
- Added central model registry
- Added dataset manifest system
- Added herbicide safety checks
- Enhanced input validation
- Added ML regression tests
- Documented canonical pipelines

---

## Contact

For questions about canonical pipelines or to propose changes, consult the development team or update this document with rationale.
