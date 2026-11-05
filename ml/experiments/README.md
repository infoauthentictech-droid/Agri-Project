# Model Improvement Experiments

This directory contains scripts for the recommended model improvement experiments as outlined in the project review.

## Overview

These experiments are designed to be run in order after establishing a clean baseline with the fixed code:

1. **Experiment 1**: Confidence Threshold Sweep
2. **Experiment 2**: Investigate Missed Weeds
3. **Experiment 3**: Improve Training Data
4. **Experiment 4**: Test Image Resolution
5. **Experiment 5**: Validate EfficientNet Independently

## Prerequisites

Ensure you have:
- ✅ Fixed the code issues (completed)
- ✅ Retrained the plant-gate model with proper train/val/test split
- ✅ Verified the backend can load models
- Virtual environment activated: `.venv-gpu/Scripts/Activate.ps1`

## Running the Experiments

### Experiment 1: Confidence Threshold Sweep

Evaluates YOLO detection at different confidence thresholds (0.10, 0.15, 0.20, 0.25, 0.30) to find the optimal balance between precision and recall.

```bash
# Using virtual environment
.venv-gpu/Scripts/python.exe ml/experiments/experiment_1_threshold_sweep.py

# With custom model
.venv-gpu/Scripts/python.exe ml/experiments/experiment_1_threshold_sweep.py --model experiments/yolo/agri_yolov8n/weights/best.pt

# Test on test set instead of validation
.venv-gpu/Scripts/python.exe ml/experiments/experiment_1_threshold_sweep.py --split test
```

**Output**: `experiments/evaluation/threshold_sweep/threshold_sweep_val.json`

**Key Metrics**: mAP50, mAP50-95, Precision, Recall, F1 score at each threshold

---

### Experiment 2: Investigate Missed Weeds

Analyzes false-negative examples, especially small, distant, overlapping, or partially obscured weeds. Generates statistics and visualizations.

```bash
# Run analysis
.venv-gpu/Scripts/python.exe ml/experiments/experiment_2_missed_weeds.py

# On test set
.venv-gpu/Scripts/python.exe ml/experiments/experiment_2_missed_weeds.py --split test

# With custom confidence threshold
.venv-gpu/Scripts/python.exe ml/experiments/experiment_2_missed_weeds.py --conf 0.20
```

**Output**: `experiments/evaluation/missed_weeds_analysis/missed_weeds_val.json`

**Key Metrics**:
- Total ground truth weeds vs detected weeds
- Recall by size category (small, medium, large)
- Examples of missed weeds with metadata

---

### Experiment 3: Improve Training Data

Provides guidance and tools for analyzing current training data distribution and applying augmentations.

```bash
# Analyze current data distribution
.venv-gpu/Scripts/python.exe ml/experiments/experiment_3_data_augmentation.py --analyze

# Apply augmentation to training data
.venv-gpu/Scripts/python.exe ml/experiments/experiment_3_data_augmentation.py --augment --output datasets/augmented

# With custom augmentation factor
.venv-gpu/Scripts/python.exe ml/experiments/experiment_3_data_augmentation.py --augment --augment-factor 3
```

**Output**:
- Analysis: `experiments/evaluation/data_augmentation/data_analysis.json`
- Augmented data: `datasets/augmented/`

**Key Features**:
- Analyzes class balance and weed size distribution
- Suggests augmentation strategies based on data characteristics
- Applies geometric, color, and noise augmentations

**Note**: Collecting new field images with sparse, young, or partially hidden weeds is a manual process. This script helps with augmentation of existing data.

---

### Experiment 4: Test Image Resolution

Compares different input resolutions (640, 832, 960) to find the optimal balance between small-weed recall and inference latency.

```bash
# Run resolution comparison
.venv-gpu/Scripts/python.exe ml/experiments/experiment_4_resolution_comparison.py

# With custom resolutions
.venv-gpu/Scripts/python.exe ml/experiments/experiment_4_resolution_comparison.py --resolutions 640 832 960 1024

# Measure single-image latency
.venv-gpu/Scripts/python.exe ml/experiments/experiment_4_resolution_comparison.py --measure-latency
```

**Output**: `experiments/evaluation/resolution_comparison/resolution_comparison_val.json`

**Key Metrics**:
- mAP50, mAP50-95, Precision, Recall, F1 at each resolution
- Inference time for full evaluation
- Single-image latency statistics (avg, min, max, std)
- Efficiency metric (F1 / time)

---

### Experiment 5: Validate EfficientNet Independently

Tests the EfficientNet model to verify it generalizes well and checks for potential source-image overlap between training and test sets.

```bash
# Run validation
.venv-gpu/Scripts/python.exe ml/experiments/experiment_5_efficientnet_validation.py

# With specific model
.venv-gpu/Scripts/python.exe ml/experiments/experiment_5_efficientnet_validation.py --model efficientnet

# Check for overfitting with training results
.venv-gpu/Scripts/python.exe ml/experiments/experiment_5_efficientnet_validation.py --train-results experiments/classification/efficientnet/results.json
```

**Output**: `experiments/evaluation/efficientnet_validation/efficientnet_validation_efficientnet.json`

**Key Metrics**:
- Overall accuracy
- Per-class precision, recall, F1, support
- Confusion matrix
- Calibration curve data
- Overfitting warnings (if training results provided)

---

## Expected Timeline

- **Experiment 1**: 10-15 minutes (per split)
- **Experiment 2**: 5-10 minutes (analyzes 100 images)
- **Experiment 3**: 5 minutes (analysis) + 10-20 minutes (augmentation)
- **Experiment 4**: 15-30 minutes (per resolution)
- **Experiment 5**: 5-10 minutes

## Interpreting Results

### Experiment 1: Threshold Selection
- Look for the threshold with the highest F1 score
- Consider your application requirements:
  - Higher precision = fewer false positives (but may miss weeds)
  - Higher recall = catch more weeds (but may have false positives)

### Experiment 2: Missed Weeds
- Small weeds with low recall? → Increase resolution or use small object augmentation
- Overall low recall? → Lower confidence threshold or improve training data
- Check missed examples for patterns (lighting, angle, occlusion)

### Experiment 3: Data Augmentation
- If weeds are underrepresented → Use class weights or oversample
- If average weed size is small → Use higher resolution or scale augmentation
- Apply suggested augmentations to improve generalization

### Experiment 4: Resolution
- Higher resolution = better small-weed recall but slower inference
- Choose based on your deployment constraints (mobile vs server)
- Look at efficiency metric (F1/time) for balanced choice

### Experiment 5: EfficientNet Validation
- Large train-test gap (>5%) → Possible overfitting or source-image overlap
- Test on new field captures to verify generalization
- Per-class gaps indicate which class needs more diverse data

## Next Steps After Experiments

Based on experiment results, you may need to:

1. **Retrain YOLO** with:
   - Optimized confidence threshold
   - Higher resolution (if beneficial)
   - Augmented training data
   - Better class weights

2. **Collect more data**:
   - Focus on missed weed types/sizes
   - Diverse lighting conditions
   - Different camera angles
   - Young/sparse/partially hidden weeds

3. **Fine-tune EfficientNet**:
   - If overfitting detected
   - If poor generalization on new data
   - Use data augmentation

4. **Deploy with confidence**:
   - Use validated threshold
   - Monitor performance in production
   - Collect feedback for further improvement

## Troubleshooting

### GPU Memory Issues
- Reduce batch size in training scripts
- Use smaller image resolution
- Close other GPU applications

### Model Not Found
- Check model path is correct
- Ensure model was trained/saved properly
- Use `--model` flag to specify path

### Data Not Found
- Verify dataset paths in data.yaml
- Check directory structure matches expected format
- Ensure data is in `datasets/processed/`

## Contact

For questions or issues, refer to the main project documentation or contact the development team.
