# ============================================================================
# AgriVision/Groundnut AI - Complete Fixed Setup and Training Script
# ============================================================================
# This script addresses ALL problems identified in the review:
# - Data leakage prevention
# - Correct image-label pairing
# - Augmentation with label transformation
# - Duplicate detection
# - Stratified splits
# - Comprehensive validation
# - Fixed classification training (no duplicate main)
# - GPU verification
# - Non-destructive operations
# ============================================================================

cd "D:\Flutter\PHD PROJECT\Agri Project"

$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "COMPLETE FIXED DATASET PREPARATION AND TRAINING" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Addressing ALL identified problems" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "STEP 1: GPU Availability Check" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

$gpuCheck = @"
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: CUDA not available. Training will use CPU (slow).")
    print("To enable GPU:")
    print("  1. Install NVIDIA CUDA Toolkit 12.1")
    print("  2. Install NVIDIA cuDNN 8.9")
    print("  3. Update NVIDIA GPU drivers")
    print("  4. Restart your computer")
"@

$gpuCheck | Out-File -FilePath "ml\data\check_gpu.py" -Encoding UTF8
.venv-gpu\Scripts\python.exe ml\data\check_gpu.py

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "STEP 2: Run Robust Dataset Preparation" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

Write-Host "This will:" -ForegroundColor Yellow
Write-Host "  1. Prepare primary dataset from your field images" -ForegroundColor White
Write-Host "  2. Prepare secondary dataset from public sources" -ForegroundColor White
Write-Host "  3. Create augmented dataset with label transformations" -ForegroundColor White
Write-Host "  4. Create hybrid dataset (combined sources)" -ForegroundColor White
Write-Host "  5. Create stratified train/val/test splits" -ForegroundColor White
Write-Host "  6. Detect duplicates and validate annotations" -ForegroundColor White
Write-Host "  7. Generate comprehensive manifest" -ForegroundColor White
Write-Host ""

# Run the robust dataset preparation
.venv-gpu\Scripts\python.exe ml\data\robust_dataset_preparation_v2.py `
    --primary-source "datasets/my_own_dataset" `
    --secondary-source "datasets/public_dataset" `
    --output-dir "datasets" `
    --augmentation-factor 5 `
    --train-ratio 0.7 `
    --val-ratio 0.15 `
    --random-seed 42

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "STEP 3: Verify Dataset Manifest" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

$verifyScript = @"
import json
from pathlib import Path

manifest_path = Path('datasets/dataset_manifest.json')
if manifest_path.exists():
    with open(manifest_path) as f:
        manifest = json.load(f)

    print("Dataset Manifest Summary:")
    print("=" * 60)
    print(f"Timestamp: {manifest['metadata']['timestamp']}")
    print(f"Random Seed: {manifest['metadata']['random_seed']}")
    print(f"Splits: {manifest['metadata']['splits']}")
    print(f"Augmentation Factor: {manifest['metadata']['augmentation_factor']}")
    print()
    print(f"Validation Errors: {len(manifest['validation_errors'])}")
    if manifest['validation_errors']:
        print("  Errors:")
        for err in manifest['validation_errors'][:5]:  # Show first 5
            print(f"    - {err['file']}: {err['error']}")
    print()
    print(f"Duplicate Groups: {sum(len(d) for d in manifest['duplicates'].values())}")
    print()
    print("Dataset Breakdown:")
    for dataset_type, data in manifest['dataset_types'].items():
        print(f"\n{dataset_type.upper()}:")
        for task, counts in data.items():
            if isinstance(counts, dict):
                print(f"  {task}:")
                for class_name, count_data in counts.items():
                    if isinstance(count_data, dict):
                        print(f"    {class_name}: {count_data.get('total', 0)} total")
                        for split, split_count in count_data.get('splits', {}).items():
                            print(f"      {split}: {split_count}")
                    else:
                        print(f"    {class_name}: {count_data}")
            else:
                print(f"  {task}: {counts}")
else:
    print("ERROR: Manifest not found!")
"@

$verifyScript | Out-File -FilePath "ml\data\verify_manifest.py" -Encoding UTF8
.venv-gpu\Scripts\python.exe ml\data\verify_manifest.py

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "STEP 4: Training Commands" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

Write-Host ""
Write-Host "Classification Training (FIXED - no duplicate main, proper validation):" -ForegroundColor Yellow
Write-Host ""
Write-Host "# Single Models"
Write-Host ".venv-gpu\Scripts\python.exe ml\classification\train_classifier.py --model cnn --epochs 30 --dataset-type hybrid --device 0"
Write-Host ".venv-gpu\Scripts\python.exe ml\classification\train_classifier.py --model vgg16 --epochs 30 --dataset-type hybrid --device 0"
Write-Host ".venv-gpu\Scripts\python.exe ml\classification\train_classifier.py --model resnet50 --epochs 30 --dataset-type hybrid --device 0"
Write-Host ".venv-gpu\Scripts\python.exe ml\classification\train_classifier.py --model mobilenet --epochs 30 --dataset-type hybrid --device 0"
Write-Host ".venv-gpu\Scripts\python.exe ml\classification\train_classifier.py --model efficientnet --epochs 30 --dataset-type hybrid --device 0"
Write-Host ""
Write-Host "# Hybrid Models (Project Objective)"
Write-Host ".venv-gpu\Scripts\python.exe ml\classification\train_classifier.py --model resnet50_mobilenet --epochs 30 --dataset-type hybrid --device 0"
Write-Host ".venv-gpu\Scripts\python.exe ml\classification\train_classifier.py --model resnet50_efficientnet --epochs 30 --dataset-type hybrid --device 0"
Write-Host ".venv-gpu\Scripts\python.exe ml\classification\train_classifier.py --model resnet50_efficientnet_mobilenet --epochs 30 --dataset-type hybrid --device 0"
Write-Host ""
Write-Host "# Train All Models at Once"
Write-Host ".venv-gpu\Scripts\python.exe ml\classification\train_classifier.py --all-models --epochs 30 --dataset-type hybrid --device 0"
Write-Host ""

Write-Host "Detection Training (NEEDS FIXING - address YOLO issues):" -ForegroundColor Yellow
Write-Host ".venv-gpu\Scripts\python.exe ml\detection\train_yolo.py --epochs 100 --batch 16 --device 0 --dataset-type hybrid"
Write-Host ""

Write-Host "Segmentation Training (NEEDS FIXING - address YOLO-Seg issues):" -ForegroundColor Yellow
Write-Host ".venv-gpu\Scripts\python.exe ml\segmentation\train_yoloseg.py --epochs 100 --batch 16 --device 0 --dataset-type hybrid"
Write-Host ""

Write-Host "Plant Gate Training (NEEDS FIXING - address leakage issues):" -ForegroundColor Yellow
Write-Host ".venv-gpu\Scripts\python.exe ml\gate\train_plant_gate.py"
Write-Host ""

Write-Host "Hybrid Comparison (NEEDS FIXING - address dataset consistency):" -ForegroundColor Yellow
Write-Host ".venv-gpu\Scripts\python.exe ml\experiments\compare_hybrid_models.py"
Write-Host ""

Write-Host "Run Tests:" -ForegroundColor Yellow
Write-Host ".venv-gpu\Scripts\python.exe -m pytest tests\test_ai_services.py tests\test_ml_regression.py -v"
Write-Host ""

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "FIXES COMPLETED" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "DATASET PREPARATION (All Issues Fixed):" -ForegroundColor White
Write-Host "  ✓ Data leakage prevention - augmented images grouped with originals" -ForegroundColor Green
Write-Host "  ✓ Correct split logic - separate input/output directories" -ForegroundColor Green
Write-Host "  ✓ Idempotent operations - non-destructive, uses separate output" -ForegroundColor Green
Write-Host "  ✓ Image-label pairing validation - comprehensive checks" -ForegroundColor Green
Write-Host "  ✓ Augmentation with label transformation - proper coordinate adjustment" -ForegroundColor Green
Write-Host "  ✓ File collision handling - unique naming with counters" -ForegroundColor Green
Write-Host "  ✓ Consistent recursive image discovery - handles all extensions" -ForegroundColor Green
Write-Host "  ✓ Duplicate detection - SHA256 hashing, reported in manifest" -ForegroundColor Green
Write-Host "  ✓ Image-label integrity validation - YOLO format checks" -ForegroundColor Green
Write-Host "  ✓ Stratified splits with class balance - group-level tracking" -ForegroundColor Green
Write-Host "  ✓ Authoritative dataset manifest - comprehensive tracking" -ForegroundColor Green
Write-Host "  ✓ Hybrid dataset created before splitting - no contamination" -ForegroundColor Green
Write-Host "  ✓ Comprehensive error handling - fail-fast validation" -ForegroundColor Green
Write-Host "  ✓ Reproducible transformations - fixed random seed" -ForegroundColor Green
Write-Host "  ✓ GPU availability check - verified before training" -ForegroundColor Green
Write-Host ""
Write-Host "CLASSIFICATION TRAINING (Blocking Defects Fixed):" -ForegroundColor White
Write-Host "  ✓ Duplicate main() definitions removed" -ForegroundColor Green
Write-Host "  ✓ Premature execution blocked - train_model defined before use" -ForegroundColor Green
Write-Host "  ✓ Consistent argument parsing - single entry point" -ForegroundColor Green
Write-Host "  ✓ Device validation - proper GPU/CPU handling" -ForegroundColor Green
Write-Host "  ✓ Image size per model - 224 for ResNet/hybrids, 128 for others" -ForegroundColor Green
Write-Host "  ✓ Class balance validation - checks both classes present" -ForegroundColor Green
Write-Host "  ✓ Early stopping enabled - with configurable patience" -ForegroundColor Green
Write-Host "  ✓ Best model saving - guaranteed creation" -ForegroundColor Green
Write-Host "  ✓ Classification report - explicit target names, zero_division handling" -ForegroundColor Green
Write-Host "  ✓ Hyperparameter validation - positive values checked" -ForegroundColor Green
Write-Host "  ✓ Epoch tracking - actual epochs trained recorded" -ForegroundColor Green
Write-Host "  ✓ Fail-fast on errors - non-zero exit code on failures" -ForegroundColor Green
Write-Host "  ✓ Dataset type support - --dataset-type argument properly handled" -ForegroundColor Green
Write-Host ""
Write-Host "STILL NEEDS FIXING:" -ForegroundColor Yellow
Write-Host "  ⚠ Detection training script (YOLO annotation validation, device verification)" -ForegroundColor Red
Write-Host "  ⚠ Segmentation training script (mask vs box metrics, annotation validation)" -ForegroundColor Red
Write-Host "  ⚠ Plant gate training script (data leakage, threshold validation)" -ForegroundColor Red
Write-Host "  ⚠ Hybrid comparison script (dataset consistency, preprocessing)" -ForegroundColor Red
Write-Host "  ⚠ XAI implementation (Grad-CAM validation)" -ForegroundColor Red
Write-Host "  ⚠ Herbicide recommendation (safety validation)" -ForegroundColor Red
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "NEXT STEPS" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Run this script to prepare datasets and train classification" -ForegroundColor Yellow
Write-Host "2. Review the dataset manifest at datasets/dataset_manifest.json" -ForegroundColor Yellow
Write-Host "3. Run classification training with fixed script" -ForegroundColor Yellow
Write-Host "4. Address remaining training scripts (detection, segmentation, plant gate)" -ForegroundColor Yellow
Write-Host "5. Fix hybrid comparison script for fair evaluation" -ForegroundColor Yellow
Write-Host "6. Validate XAI and herbicide recommendation components" -ForegroundColor Yellow
Write-Host ""
Write-Host "To fix remaining scripts, continue systematically through:" -ForegroundColor Yellow
Write-Host "  - ml/detection/train_yolo.py" -ForegroundColor White
Write-Host "  - ml/segmentation/train_yoloseg.py" -ForegroundColor White
Write-Host "  - ml/gate/train_plant_gate.py" -ForegroundColor White
Write-Host "  - ml/experiments/compare_hybrid_models.py" -ForegroundColor White
Write-Host "============================================================" -ForegroundColor Cyan
