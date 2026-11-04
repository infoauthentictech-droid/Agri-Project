# Training Guide for Groundnut AI Algorithms

This guide provides step-by-step instructions for training all algorithms in the project.

## Prerequisites

1. **Activate GPU Virtual Environment**
   ```powershell
   cd "D:\Flutter\PHD PROJECT\Agri Project"
   .venv-gpu\Scripts\Activate.ps1
   ```

2. **Verify GPU Availability**
   ```powershell
   .venv-gpu\Scripts\python.exe -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
   ```

## Training All Models (Automated)

### Option 1: Train All Models at Once
```powershell
cd "D:\Flutter\PHD PROJECT\Agri Project"
.\train_all_models.ps1
```

This will train:
1. Single classification models (CNN, ResNet50, MobileNet, EfficientNet)
2. Hybrid classification models (ResNet50+MobileNet, ResNet50+EfficientNet, ResNet50+EfficientNet+MobileNet)
3. YOLO detection model
4. YOLO-Seg segmentation model
5. Plant gate model
6. Comprehensive evaluation

## Training Individual Algorithms

### 1. Single Classification Models

#### CNN
```powershell
cd "D:\Flutter\PHD PROJECT\Agri Project"
.venv-gpu\Scripts\python.exe ml/classification/train_classifier.py --model cnn --epochs 30 --batch 32 --device 0
```

#### ResNet50
```powershell
.venv-gpu\Scripts\python.exe ml/classification/train_classifier.py --model resnet50 --epochs 30 --batch 32 --device 0
```

#### MobileNetV2
```powershell
.venv-gpu\Scripts\python.exe ml/classification/train_classifier.py --model mobilenet --epochs 30 --batch 32 --device 0
```

#### EfficientNet-B0
```powershell
.venv-gpu\Scripts\python.exe ml/classification/train_classifier.py --model efficientnet --epochs 30 --batch 32 --device 0
```

#### VGG16 (Optional - requires more memory)
```powershell
.venv-gpu\Scripts\python.exe ml/classification/train_classifier.py --model vgg16 --epochs 30 --batch 16 --device 0
```

### 2. Hybrid Classification Models

#### ResNet50 + MobileNet
```powershell
.venv-gpu\Scripts\python.exe ml/classification/train_classifier.py --model resnet50_mobilenet --epochs 30 --batch 16 --device 0
```

#### ResNet50 + EfficientNet
```powershell
.venv-gpu\Scripts\python.exe ml/classification/train_classifier.py --model resnet50_efficientnet --epochs 30 --batch 16 --device 0
```

#### ResNet50 + EfficientNet + MobileNet
```powershell
.venv-gpu\Scripts\python.exe ml/classification/train_classifier.py --model resnet50_efficientnet_mobilenet --epochs 30 --batch 8 --device 0
```

### 3. YOLO Detection Model

```powershell
.venv-gpu\Scripts\python.exe ml/detection/train_yolo.py --epochs 100 --batch 16 --device 0 --imgsz 640
```

**Parameters:**
- `--epochs`: Number of training epochs (default: 100)
- `--batch`: Batch size (default: 16)
- `--device`: GPU device ID (default: 0)
- `--imgsz`: Image size (default: 640)
- `--model`: YOLO model variant (default: yolov8n.pt)

### 4. YOLO-Seg Segmentation Model

```powershell
.venv-gpu\Scripts\python.exe ml/segmentation/train_yoloseg.py --model yolov8n-seg.pt --epochs 100 --batch 16 --device 0
```

**Parameters:**
- `--model`: YOLO-Seg model (yolov8n-seg.pt, yolo11n-seg.pt)
- `--epochs`: Number of training epochs (default: 100)
- `--batch`: Batch size (default: 16)
- `--device`: GPU device ID (default: 0)

### 5. Plant Gate Model

```powershell
.venv-gpu\Scripts\python.exe ml/gate/train_plant_gate.py --device 0
```

### 6. CycleGAN Synthesis (Optional)

First, prepare your datasets:
- Domain A: Normal crop images
- Domain B: Weed-infested images

```powershell
.venv-gpu\Scripts\python.exe ml/synthesis/cyclegan.py --domain_a path/to/normal_crops --domain_b path/to/weed_infested --epochs 200 --batch 1 --device 0
```

## Evaluation

### Evaluate All Models
```powershell
.venv-gpu\Scripts\python.exe ml/evaluation/evaluate_all_models.py
```

### Evaluate Individual Models

#### YOLO Detection
```powershell
.venv-gpu\Scripts\python.exe -c "from ultralytics import YOLO; model = YOLO('experiments/yolo/agri_yolov8n/weights/best.pt'); metrics = model.val(data='datasets/processed/groundnut_weed_detection/data.yaml', split='test', device=0); print(f'mAP50: {metrics.box.map50:.4f}')"
```

#### Classification
```powershell
.venv-gpu\Scripts\python.exe ml/evaluation/evaluate_classifier.py --model resnet50
```

## Training Parameters Reference

### Classification Models
- **Image Size**: 128x128 (single), 224x224 (hybrid)
- **Epochs**: 30
- **Batch Size**: 32 (single), 16 (hybrid), 8 (triple hybrid)
- **Learning Rate**: 0.001
- **Optimizer**: AdamW
- **Augmentation**: Mosaic & Mixup

### Detection Models
- **Image Size**: 640x640
- **Epochs**: 100
- **Batch Size**: 16
- **Learning Rate**: 0.01
- **Optimizer**: AdamW
- **Augmentation**: Mosaic (1.0), Mixup (0.0)

### Segmentation Models
- **Image Size**: 640x640
- **Epochs**: 100
- **Batch Size**: 16
- **Learning Rate**: 0.01
- **Optimizer**: AdamW

## Output Locations

### Classification Models
```
experiments/classification/
├── cnn/
│   ├── best_model.pth
│   ├── training_history.json
│   └── results.json
├── resnet50/
├── mobilenet/
├── efficientnet/
├── resnet50_mobilenet/
├── resnet50_efficientnet/
└── resnet50_efficientnet_mobilenet/
```

### Detection Model
```
experiments/yolo/agri_yolov8n/
├── weights/
│   ├── best.pt
│   └── last.pt
└── training_summary.json
```

### Segmentation Model
```
experiments/yoloseg/agri_yolov8n_seg/
├── weights/
│   ├── best.pt
│   └── last.pt
└── training_summary.json
```

### Plant Gate Model
```
backend/app/utils/
├── plant_gate_model.joblib
└── plant_gate_model.json
```

### Evaluation Results
```
experiments/evaluation/
├── yolo_evaluation.json
├── classification_evaluation.json
├── plant_gate_evaluation.json
├── comprehensive_evaluation_report.json
└── *_confusion_matrix.png
```

## Troubleshooting

### CUDA Out of Memory
Reduce batch size:
```powershell
# Classification
--batch 16  # or 8 for hybrid models

# Detection/Segmentation
--batch 8
```

### Training Too Slow
- Increase batch size if GPU memory allows
- Reduce image size for detection/segmentation: `--imgsz 512`
- Use smaller model variants (e.g., yolov8n.pt instead of yolov8s.pt)

### Poor Performance
- Increase training epochs: `--epochs 50` or `--epochs 100`
- Adjust learning rate: `--lr 0.0005`
- Check data quality and dataset balance

## Next Steps After Training

1. **Update Backend Configuration**
   - Copy trained models to production directory
   - Update model paths in backend configuration

2. **Test Integration**
   - Start backend server
   - Test each algorithm endpoint
   - Verify all algorithms work together

3. **Deploy**
   - Follow deployment guide in `docs/DEPLOYMENT.md`
