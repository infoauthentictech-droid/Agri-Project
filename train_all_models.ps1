# Training Script for All Algorithms in Groundnut AI
# This script trains all models using GPU
#
# IMPORTANT: You MUST activate the virtual environment first:
#   cd "D:\Flutter\PHD PROJECT\Agri Project"
#   .\.venv-gpu\Scripts\Activate.ps1
#
# Then run this script:
#   .\train_all_models.ps1

$ErrorActionPreference = "Stop"

# Set project directory
$PROJECT_DIR = "D:\Flutter\PHD PROJECT\Agri Project"
# Use python from activated environment (not full path)
$PYTHON = "python"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Groundnut AI - Training All Algorithms (GPU Mode)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Change to project directory
Set-Location $PROJECT_DIR

# Verify GPU availability
Write-Host "Verifying GPU availability..." -ForegroundColor Yellow
try {
    $gpuCheck = & $PYTHON -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
    Write-Host $gpuCheck -ForegroundColor Green
    if ($gpuCheck -match "CUDA Available: False") {
        Write-Host "WARNING: CUDA not available! Training will attempt to use GPU but may fail." -ForegroundColor Red
        Write-Host "Press Ctrl+C to cancel or Enter to continue..." -ForegroundColor Yellow
        Read-Host
    }
} catch {
    Write-Host "WARNING: GPU check failed, but training will attempt to use GPU anyway" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Starting training with GPU (device 0)..." -ForegroundColor Green
Write-Host ""

# ============================================================
# 1. Train Single Classification Models
# ============================================================
Write-Host "============================================================" -ForegroundColor Yellow
Write-Host "Step 1: Training Single Classification Models" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow

$single_models = @("cnn", "resnet50", "mobilenet", "efficientnet", "vgg16")

foreach ($model in $single_models) {
    Write-Host "`nTraining $model..." -ForegroundColor Green
    & $PYTHON ml/classification/train_classifier.py --model $model --epochs 30 --batch 32 --device 0
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error training $model" -ForegroundColor Red
    } else {
        Write-Host "$model training completed successfully" -ForegroundColor Green
    }
}

# ============================================================
# 2. Train Hybrid Classification Models
# ============================================================
Write-Host "`n============================================================" -ForegroundColor Yellow
Write-Host "Step 2: Training Hybrid Classification Models" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow

$hybrid_models = @("resnet50_mobilenet", "resnet50_efficientnet", "resnet50_efficientnet_mobilenet")

foreach ($model in $hybrid_models) {
    Write-Host "`nTraining $model..." -ForegroundColor Green
    & $PYTHON ml/classification/train_classifier.py --model $model --epochs 30 --batch 16 --device 0
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error training $model" -ForegroundColor Red
    } else {
        Write-Host "$model training completed successfully" -ForegroundColor Green
    }
}

# ============================================================
# 3. Train YOLO Detection Model
# ============================================================
Write-Host "`n============================================================" -ForegroundColor Yellow
Write-Host "Step 3: Training YOLO Detection Model" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow

Write-Host "`nTraining YOLOv8 detection..." -ForegroundColor Green
& $PYTHON ml/detection/train_yolo.py --epochs 100 --batch 16 --device 0
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error training YOLO detection" -ForegroundColor Red
} else {
    Write-Host "YOLO detection training completed successfully" -ForegroundColor Green
}

# ============================================================
# 4. Train YOLO-Seg Segmentation Model
# ============================================================
Write-Host "`n============================================================" -ForegroundColor Yellow
Write-Host "Step 4: Training YOLO-Seg Segmentation Model" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow

Write-Host "`nTraining YOLOv8-Seg..." -ForegroundColor Green
& $PYTHON ml/segmentation/train_yoloseg.py --model yolov8n-seg.pt --epochs 100 --batch 16 --device 0
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error training YOLO-Seg" -ForegroundColor Red
} else {
    Write-Host "YOLO-Seg training completed successfully" -ForegroundColor Green
}

# ============================================================
# 5. Train Plant Gate Model
# ============================================================
Write-Host "`n============================================================" -ForegroundColor Yellow
Write-Host "Step 5: Training Plant Gate Model" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow

Write-Host "`nTraining plant gate..." -ForegroundColor Green
& $PYTHON ml/gate/train_plant_gate.py --device 0
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error training plant gate" -ForegroundColor Red
} else {
    Write-Host "Plant gate training completed successfully" -ForegroundColor Green
}

# ============================================================
# 6. Prepare CycleGAN Dataset
# ============================================================
Write-Host "`n============================================================" -ForegroundColor Yellow
Write-Host "Step 6: Preparing CycleGAN Dataset" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow
Write-Host ""

$CYCLEGAN_DATASET_SOURCE = "datasets/processed/groundnut_weed_detection"
$CYCLEGAN_DATASET_OUTPUT = "datasets/cyclegan"

if (Test-Path $CYCLEGAN_DATASET_SOURCE) {
    Write-Host "Preparing CycleGAN dataset from detection dataset..." -ForegroundColor Green
    & $PYTHON ml/synthesis/prepare_cyclegan_dataset.py --source $CYCLEGAN_DATASET_SOURCE --output $CYCLEGAN_DATASET_OUTPUT --synthetic --synthetic-samples 100
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error preparing CycleGAN dataset" -ForegroundColor Red
    } else {
        Write-Host "CycleGAN dataset prepared successfully" -ForegroundColor Green
    }
} else {
    Write-Host "Warning: Detection dataset not found at $CYCLEGAN_DATASET_SOURCE" -ForegroundColor Yellow
    Write-Host "Skipping CycleGAN dataset preparation" -ForegroundColor Yellow
}

# ============================================================
# 7. Train CycleGAN Synthesis Model (Optional)
# ============================================================
Write-Host "`n============================================================" -ForegroundColor Yellow
Write-Host "Step 7: Training CycleGAN Synthesis Model (Optional)" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow
Write-Host ""

$CYCLEGAN_DOMAIN_A = "$CYCLEGAN_DATASET_OUTPUT/normal_crops/train"
$CYCLEGAN_DOMAIN_B = "$CYCLEGAN_DATASET_OUTPUT/weed_infested/train"

if ((Test-Path $CYCLEGAN_DOMAIN_A) -and (Test-Path $CYCLEGAN_DOMAIN_B)) {
    Write-Host "Training CycleGAN..." -ForegroundColor Green
    & $PYTHON ml/synthesis/cyclegan.py --domain_a $CYCLEGAN_DOMAIN_A --domain_b $CYCLEGAN_DOMAIN_B --epochs 200 --batch 1 --device 0
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error training CycleGAN" -ForegroundColor Red
    } else {
        Write-Host "CycleGAN training completed successfully" -ForegroundColor Green
    }
} else {
    Write-Host "Skipping CycleGAN training (datasets not ready)" -ForegroundColor Yellow
    Write-Host "To train CycleGAN manually:" -ForegroundColor Yellow
    Write-Host "  1. Prepare dataset: python ml/synthesis/prepare_cyclegan_dataset.py --source path/to/dataset --output datasets/cyclegan --synthetic" -ForegroundColor Cyan
    Write-Host "  2. Train: python ml/synthesis/cyclegan.py --domain_a datasets/cyclegan/normal_crops/train --domain_b datasets/cyclegan/weed_infested/train --epochs 200 --batch 1 --device 0" -ForegroundColor Cyan
}
Write-Host ""

# ============================================================
# 8. Train Plant Gate Model
# ============================================================
Write-Host "`n============================================================" -ForegroundColor Yellow
Write-Host "Step 8: Training Plant Gate Model" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow

Write-Host "`nTraining plant gate..." -ForegroundColor Green
& $PYTHON ml/gate/train_plant_gate.py --device 0
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error training plant gate" -ForegroundColor Red
} else {
    Write-Host "Plant gate training completed successfully" -ForegroundColor Green
}

# ============================================================
# 9. Evaluate All Models
# ============================================================
Write-Host "`n============================================================" -ForegroundColor Yellow
Write-Host "Step 9: Evaluating All Models" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow

Write-Host "`nRunning comprehensive evaluation..." -ForegroundColor Green
& $PYTHON ml/evaluation/evaluate_all_models.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error during evaluation" -ForegroundColor Red
} else {
    Write-Host "Evaluation completed successfully" -ForegroundColor Green
}

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "All Training Completed!" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Results saved to:"
Write-Host "  - Classification: experiments/classification/"
Write-Host "  - Detection: experiments/yolo/agri_yolov8n/"
Write-Host "  - Segmentation: experiments/yoloseg/agri_yolov8n_seg/"
Write-Host "  - CycleGAN: experiments/cyclegan/"
Write-Host "  - CycleGAN Dataset: datasets/cyclegan/"
Write-Host "  - Plant Gate: backend/app/utils/plant_gate_model.joblib"
Write-Host "  - Evaluation: experiments/evaluation/"
Write-Host ""
