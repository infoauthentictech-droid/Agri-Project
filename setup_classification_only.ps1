# ============================================================================
# AgriVision/Groundnut AI - Simplified Setup for Classification Only
# ============================================================================
# This script prepares datasets for classification training only.
# Detection and segmentation are skipped because:
# - Your mixed dataset has no detection labels
# - Public dataset has PNG masks (not YOLO format)
# - Classification data is complete and ready
# ============================================================================

cd "D:\Flutter\PHD PROJECT\Agri Project"

$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "SIMPLIFIED DATASET PREPARATION - CLASSIFICATION ONLY" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

Write-Host ""
Write-Host "STEP 1: GPU Availability Check" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

.venv-gpu\Scripts\python.exe ml\data\check_gpu.py

Write-Host ""
Write-Host "STEP 2: Prepare Classification Dataset" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

.venv-gpu\Scripts\python.exe ml\data\prepare_classification_dataset.py `
    --primary-source "datasets/my_own_dataset" `
    --output-dir "datasets" `
    --augmentation-factor 2 `
    --train-ratio 0.7 `
    --val-ratio 0.15 `
    --random-seed 42

Write-Host ""
Write-Host "STEP 3: Verify Dataset Manifest" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

.venv-gpu\Scripts\python.exe ml\data\verify_manifest.py

Write-Host ""
Write-Host "STEP 4: Training Commands (CLASSIFICATION ONLY)" -ForegroundColor Cyan
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

Write-Host "HYBRID COMPARISON (FIXED - dataset consistency):" -ForegroundColor Yellow
Write-Host ".venv-gpu\Scripts\python.exe ml\experiments\compare_hybrid_models.py --dataset-type hybrid"
Write-Host ""

Write-Host "RUN TESTS:" -ForegroundColor Yellow
Write-Host ".venv-gpu\Scripts\python.exe -m pytest tests\test_ai_services.py tests\test_ml_regression.py -v"
Write-Host ""

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "NOTES:" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "- Detection/Segmentation skipped (no YOLO labels available)" -ForegroundColor White
Write-Host "- Classification dataset prepared with augmentation factor 2" -ForegroundColor White
Write-Host "- Data leakage prevention enabled (group-aware splits)" -ForegroundColor White
Write-Host "- All training scripts fixed and ready" -ForegroundColor White
Write-Host "============================================================" -ForegroundColor Cyan
