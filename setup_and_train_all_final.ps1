# ============================================================================
# AgriVision/Groundnut AI - Complete Fixed Setup and Training Script
# ============================================================================
# ALL PROBLEMS FIXED:
# - Data leakage prevention
# - Correct image-label pairing
# - Augmentation with label transformation
# - Duplicate detection
# - Stratified splits
# - Comprehensive validation
# - Fixed classification training (no duplicate main)
# - Fixed detection training (YOLO validation)
# - Fixed segmentation training (mask vs box metrics)
# - Fixed plant gate training (data leakage prevention)
# - Fixed hybrid comparison (dataset consistency)
# - GPU verification
# - Non-destructive operations
# ============================================================================

cd "D:\Flutter\PHD PROJECT\Agri Project"

$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "COMPLETE FIXED DATASET PREPARATION AND TRAINING" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "ALL PROBLEMS FIXED" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan

Write-Host ""
Write-Host "STEP 1: GPU Availability Check" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

.venv-gpu\Scripts\python.exe ml\data\check_gpu.py

Write-Host ""
Write-Host "STEP 2: Run Robust Dataset Preparation" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

.venv-gpu\Scripts\python.exe ml\data\robust_dataset_preparation_v2.py `
    --primary-source "datasets/my_own_dataset" `
    --secondary-source "datasets/public_dataset" `
    --output-dir "datasets" `
    --augmentation-factor 5 `
    --train-ratio 0.7 `
    --val-ratio 0.15 `
    --random-seed 42

Write-Host ""
Write-Host "STEP 3: Verify Dataset Manifest" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

.venv-gpu\Scripts\python.exe ml\data\verify_manifest.py

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "STEP 4: Training Commands (ALL SCRIPTS FIXED)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

Write-Host ""
Write-Host "CLASSIFICATION TRAINING (FIXED):" -ForegroundColor Yellow
Write-Host ""
Write-Host "# Train all models at once"
Write-Host ".venv-gpu\Scripts\python.exe ml\classification\train_classifier.py --all-models --epochs 30 --dataset-type hybrid --device 0"
Write-Host ""
Write-Host "# Or train individually"
Write-Host ".venv-gpu\Scripts\python.exe ml\classification\train_classifier.py --model resnet50 --epochs 30 --dataset-type hybrid --device 0"
Write-Host ".venv-gpu\Scripts\python.exe ml\classification\train_classifier.py --model resnet50_mobilenet --epochs 30 --dataset-type hybrid --device 0"
Write-Host ".venv-gpu\Scripts\python.exe ml\classification\train_classifier.py --model resnet50_efficientnet --epochs 30 --dataset-type hybrid --device 0"
Write-Host ""

Write-Host "DETECTION TRAINING (FIXED):" -ForegroundColor Yellow
Write-Host ".venv-gpu\Scripts\python.exe ml\detection\train_yolo.py --epochs 100 --batch 16 --device 0 --dataset-type hybrid"
Write-Host ""

Write-Host "SEGMENTATION TRAINING (FIXED - mask vs box metrics):" -ForegroundColor Yellow
Write-Host ".venv-gpu\Scripts\python.exe ml\segmentation\train_yoloseg.py --epochs 100 --batch 16 --device 0 --dataset-type hybrid"
Write-Host ""

Write-Host "PLANT GATE TRAINING (FIXED - data leakage prevention):" -ForegroundColor Yellow
Write-Host ".venv-gpu\Scripts\python.exe ml\gate\train_plant_gate.py --negatives-dir datasets/negatives_real"
Write-Host ""

Write-Host "HYBRID COMPARISON (FIXED - dataset consistency):" -ForegroundColor Yellow
Write-Host ".venv-gpu\Scripts\python.exe ml\experiments\compare_hybrid_models.py --dataset-type hybrid"
Write-Host ""

Write-Host "RUN TESTS:" -ForegroundColor Yellow
Write-Host ".venv-gpu\Scripts\python.exe -m pytest tests\test_ai_services.py tests\test_ml_regression.py -v"
Write-Host ""

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "ALL FIXES COMPLETED" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "FIXED SCRIPTS:" -ForegroundColor White
Write-Host "  ✓ ml/classification/train_classifier.py" -ForegroundColor Green
Write-Host "  ✓ ml/data/robust_dataset_preparation_v2.py" -ForegroundColor Green
Write-Host "  ✓ ml/detection/train_yolo.py" -ForegroundColor Green
Write-Host "  ✓ ml/segmentation/train_yoloseg.py" -ForegroundColor Green
Write-Host "  ✓ ml/gate/train_plant_gate.py" -ForegroundColor Green
Write-Host "  ✓ ml/experiments/compare_hybrid_models.py" -ForegroundColor Green
Write-Host ""
Write-Host "KEY IMPROVEMENTS:" -ForegroundColor White
Write-Host "  ✓ Data leakage prevention (augmented grouped with originals)" -ForegroundColor Green
Write-Host "  ✓ Image-label pairing validation (YOLO format checks)" -ForegroundColor Green
Write-Host "  ✓ Augmentation with label transformation (coordinate adjustment)" -ForegroundColor Green
Write-Host "  ✓ Duplicate detection (SHA256 hashing, reported in manifest)" -ForegroundColor Green
Write-Host "  ✓ Stratified splits with class balance (group-level tracking)" -ForegroundColor Green
Write-Host "  ✓ Comprehensive validation (fail-fast on errors)" -ForegroundColor Green
Write-Host "  ✓ Classification: No duplicate main, proper validation, device handling" -ForegroundColor Green
Write-Host "  ✓ Detection: YAML validation, annotation checks, epoch tracking" -ForegroundColor Green
Write-Host "  ✓ Segmentation: Mask vs box metrics distinguished, annotation validation" -ForegroundColor Green
Write-Host "  ✓ Plant Gate: Data leakage prevention, duplicate detection, threshold validation" -ForegroundColor Green
Write-Host "  ✓ Hybrid Comparison: Dataset consistency, preprocessing, statistical validation" -ForegroundColor Green
Write-Host "  ✓ GPU verification before training" -ForegroundColor Green
Write-Host "  ✓ Non-destructive operations (no automatic deletion)" -ForegroundColor Green
Write-Host "  ✓ Reproducible transformations (fixed random seed)" -ForegroundColor Green
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "RUN THIS SCRIPT TO:" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "1. Prepare datasets with all fixes" -ForegroundColor White
Write-Host "2. Verify GPU availability" -ForegroundColor White
Write-Host "3. Review dataset manifest" -ForegroundColor White
Write-Host "4. Run training commands (all scripts fixed)" -ForegroundColor White
Write-Host "5. Compare hybrid vs single models" -ForegroundColor White
Write-Host "6. Run tests" -ForegroundColor White
Write-Host "============================================================" -ForegroundColor Cyan
