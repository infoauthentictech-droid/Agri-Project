# Training Script for Remaining Algorithms Only
# This script trains only the algorithms that failed or were skipped

$ErrorActionPreference = "Stop"

# Set project directory
$PROJECT_DIR = "D:\Flutter\PHD PROJECT\Agri Project"
$PYTHON = "python"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Training Remaining Algorithms (GPU Mode)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Change to project directory
Set-Location $PROJECT_DIR

# Verify GPU availability
Write-Host "Verifying GPU availability..." -ForegroundColor Yellow
try {
    $gpuCheck = & $PYTHON -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
    Write-Host $gpuCheck -ForegroundColor Green
} catch {
    Write-Host "WARNING: GPU check failed" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Starting training with GPU (device 0)..." -ForegroundColor Green
Write-Host ""

# ============================================================
# 1. Prepare Segmentation Dataset from Detection Dataset
# ============================================================
Write-Host "`n============================================================" -ForegroundColor Yellow
Write-Host "Step 1: Preparing Segmentation Dataset" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow

$SEGMENTATION_DATASET = "datasets/processed/groundnut_weed_segmentation"
$DETECTION_DATASET = "datasets/processed/groundnut_weed_detection"

if (-not (Test-Path $SEGMENTATION_DATASET)) {
    Write-Host "Converting detection dataset to segmentation format..." -ForegroundColor Green
    & $PYTHON ml/data/prepare_segmentation_from_detection.py --detection $DETECTION_DATASET --output $SEGMENTATION_DATASET
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error converting to segmentation dataset" -ForegroundColor Red
    } else {
        Write-Host "Segmentation dataset created at $SEGMENTATION_DATASET" -ForegroundColor Green
    }
} else {
    Write-Host "Segmentation dataset already exists" -ForegroundColor Cyan
}

# ============================================================
# 2. Train YOLO-Seg Segmentation Model
# ============================================================
Write-Host "`n============================================================" -ForegroundColor Yellow
Write-Host "Step 2: Training YOLO-Seg Segmentation Model" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow

Write-Host "`nTraining YOLOv8-Seg..." -ForegroundColor Green
& $PYTHON ml/segmentation/train_yoloseg.py --model yolov8n-seg.pt --epochs 100 --batch 16 --device 0
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error training YOLO-Seg" -ForegroundColor Red
} else {
    Write-Host "YOLO-Seg training completed successfully" -ForegroundColor Green
}

# ============================================================
# 3. Train Hybrid Classification Models
# ============================================================
Write-Host "`n============================================================" -ForegroundColor Yellow
Write-Host "Step 3: Training Hybrid Classification Models" -ForegroundColor Yellow
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
# 4. Train Plant Gate Model (FIXED - no --device argument)
# ============================================================
Write-Host "`n============================================================" -ForegroundColor Yellow
Write-Host "Step 4: Training Plant Gate Model" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow

Write-Host "`nTraining plant gate..." -ForegroundColor Green
# Plant gate doesn't use --device argument (it's a RandomForest)
& $PYTHON ml/gate/train_plant_gate.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error training plant gate" -ForegroundColor Red
} else {
    Write-Host "Plant gate training completed successfully" -ForegroundColor Green
}

# ============================================================
# 5. Prepare CycleGAN Dataset (FIXED import issue)
# ============================================================
Write-Host "`n============================================================" -ForegroundColor Yellow
Write-Host "Step 5: Preparing CycleGAN Dataset" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow

$CYCLEGAN_DATASET_SOURCE = "datasets/processed/groundnut_weed_detection"
$CYCLEGAN_DATASET_OUTPUT = "datasets/cyclegan"

if (Test-Path $CYCLEGAN_DATASET_SOURCE) {
    Write-Host "Preparing CycleGAN dataset from detection dataset..." -ForegroundColor Green
    # Run the script directly instead of through -c
    $env:PYTHONPATH = $PROJECT_DIR
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
# 6. Train CycleGAN Synthesis Model (Optional)
# ============================================================
Write-Host "`n============================================================" -ForegroundColor Yellow
Write-Host "Step 6: Training CycleGAN Synthesis Model (Optional)" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow

$CYCLEGAN_DOMAIN_A = "$CYCLEGAN_DATASET_OUTPUT/normal_crops/train"
$CYCLEGAN_DOMAIN_B = "$CYCLEGAN_DATASET_OUTPUT/weed_infested/train"

if ((Test-Path $CYCLEGAN_DOMAIN_A) -and (Test-Path $CYCLEGAN_DOMAIN_B)) {
    Write-Host "Training CycleGAN..." -ForegroundColor Green
    $env:PYTHONPATH = $PROJECT_DIR
    & $PYTHON ml/synthesis/cyclegan.py --domain_a $CYCLEGAN_DOMAIN_A --domain_b $CYCLEGAN_DOMAIN_B --epochs 200 --batch 1 --device 0
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error training CycleGAN" -ForegroundColor Red
    } else {
        Write-Host "CycleGAN training completed successfully" -ForegroundColor Green
    }
} else {
    Write-Host "Skipping CycleGAN training (datasets not ready)" -ForegroundColor Yellow
}

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "Remaining Training Completed!" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Results saved to:"
Write-Host "  - Segmentation: experiments/yoloseg/agri_yolov8n_seg/"
Write-Host "  - Hybrid Classification: experiments/classification/"
Write-Host "  - Plant Gate: backend/app/utils/plant_gate_model.joblib"
Write-Host "  - CycleGAN: experiments/cyclegan/"
Write-Host "  - CycleGAN Dataset: datasets/cyclegan/"
Write-Host ""
