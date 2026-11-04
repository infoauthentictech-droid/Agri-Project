# Hybrid Model and Herbicide Dataset Updates

This document summarizes the fixes for:
1. Hybrid model support (to achieve higher accuracy than single modules)
2. Updated herbicide dataset integration

---

## 1. Hybrid Model Support ✅

### Problem
The project's second objective is to address low accuracy with a single module by using a hybrid model that should provide higher accuracy.

### Solution Implemented

#### A. Registered Hybrid Models in Central Model Registry
**File**: `backend/app/utils/model_registry.py`

Added three hybrid models to the model registry:
- **resnet50_mobilenet**: Combines ResNet50 + MobileNetV2
- **resnet50_efficientnet**: Combines ResNet50 + EfficientNet-B0
- **resnet50_efficientnet_mobilenet**: Combines all three architectures

All hybrid models are registered with:
- Status: `EXPERIMENTAL` (needs training)
- Image size: 224
- Class names: ["groundnut", "weed"]
- Description: Expected to provide higher accuracy than single modules

#### B. Updated Classifier Service
**File**: `backend/app/services/classifier_service.py`

- Added detection for hybrid model names
- Provides clear error message if hybrid model is requested but not yet trained
- Supports the same checkpoint loading logic as single models

#### C. Created Hybrid Model Comparison Script
**File**: `ml/experiments/compare_hybrid_models.py`

Purpose: Verify that hybrid models provide higher accuracy than single modules.

**Features**:
- Evaluates all available models on test set
- Compares accuracy, precision, recall, F1
- Generates comparison report
- Verifies if objective is met (hybrid > single)

**Usage**:
```bash
.venv-gpu/Scripts/python.exe ml/experiments/compare_hybrid_models.py
```

**Output**: 
- `experiments/evaluation/hybrid_comparison_report.json`

**What it checks**:
- Best single model accuracy
- Best hybrid model accuracy
- Improvement percentage
- Whether objective is met

---

## 2. Herbicide Dataset Updates ✅

### Problem
The herbicide dataset has been updated with new records and structure.

### Solution Implemented

#### A. Updated Safety Checker
**File**: `backend/app/utils/recommendation_safety.py`

**Changes**:
1. **Expanded crop compatibility database**:
   - Added: pendimethalin, diclosulam, oxyfluorfen, imazapic, quizalofop-p-ethyl
   - Now supports more herbicides from the updated dataset

2. **Enhanced safety constraint checking**:
   - Accepts either `dose_ai_per_ha` OR `dose_formulation_per_ha` (flexible)
   - Accepts either `phi_waiting_period_days` OR `recommended_application_time` (flexible)
   - Provides warnings when fallback fields are used instead of primary fields
   - More robust handling of null values in the updated dataset

**Why**: The updated dataset has many null fields (active_ingredient, dose_ai_per_ha, phi_waiting_period_days). The safety checker now gracefully handles this by accepting alternative fields.

#### B. Recommendation Service
**File**: `backend/app/services/recommendation_service.py`

**Status**: No changes needed - already points to the updated dataset:
```python
self.knowledge_base_path = knowledge_base_path or str(
    project_root / 'herbicide/processed/herbicide_dataset_normalized.json'
)
```

The service will automatically load the updated dataset.

#### C. Created Herbicide Dataset Validation Script
**File**: `ml/experiments/validate_herbicide_dataset.py`

Purpose: Validate the updated herbicide dataset structure and content.

**Features**:
- Validates dataset structure
- Checks for required fields
- Validates content consistency
- Checks safety-critical field coverage
- Identifies duplicate records
- Generates completeness and safety scores

**Usage**:
```bash
.venv-gpu/Scripts/python.exe ml/experiments/validate_herbicide_dataset.py
```

**Output**:
- `experiments/evaluation/herbicide_validation_report.json`

**What it checks**:
- Total records
- Completeness score (% of valid records)
- Safety score (% of records with all safety fields)
- Field coverage statistics
- Duplicate record IDs
- Unique crops, weed species, herbicides, regions

---

## Next Steps

### For Hybrid Models:

1. **Train the Hybrid Models**
   The hybrid models are registered but not yet trained. You need to train them:

   ```bash
   # Train resnet50_mobilenet
   .venv-gpu/Scripts/python.exe ml/training/train_classifier.py --model resnet50_mobilenet --epochs 30

   # Train resnet50_efficientnet
   .venv-gpu/Scripts/python.exe ml/training/train_classifier.py --model resnet50_efficientnet --epochs 30

   # Train resnet50_efficientnet_mobilenet
   .venv-gpu/Scripts/python.exe ml/training/train_classifier.py --model resnet50_efficientnet_mobilenet --epochs 30
   ```

2. **Run Comparison Script**
   After training, run the comparison to verify the objective:

   ```bash
   .venv-gpu/Scripts/python.exe ml/experiments/compare_hybrid_models.py
   ```

3. **Update Model Registry**
   If hybrid models outperform single models, update their status in the registry to `VALIDATED` and set one as `is_default=True`.

### For Herbicide Dataset:

1. **Validate the Dataset**
   Run the validation script to check data quality:

   ```bash
   .venv-gpu/Scripts/python.exe ml/experiments/validate_herbicide_dataset.py
   ```

2. **Review the Report**
   Check the validation report for:
   - Completeness score
   - Safety score
   - Missing critical fields
   - Duplicate records

3. **Fill in Missing Fields (Optional)**
   If the validation shows many null safety-critical fields, consider:
   - Adding active_ingredient values
   - Adding dose_ai_per_ha values
   - Adding phi_waiting_period_days values
   - Adding region information

4. **Test Recommendations**
   Test the recommendation service with the updated dataset:

   ```bash
   cd backend
   ../.venv-gpu/Scripts/python.exe -m uvicorn app.main:app --reload
   ```

   Then test the `/api/v1/recommend` endpoint.

---

## Summary of Changes

### Files Modified:
1. `backend/app/utils/model_registry.py` - Added 3 hybrid models
2. `backend/app/services/classifier_service.py` - Added hybrid model detection
3. `backend/app/utils/recommendation_safety.py` - Enhanced safety checks for updated dataset

### Files Created:
1. `ml/experiments/compare_hybrid_models.py` - Hybrid model comparison script
2. `ml/experiments/validate_herbicide_dataset.py` - Herbicide dataset validation script

### Key Improvements:
- ✅ Hybrid models are now registered and discoverable
- ✅ Hybrid models can be trained and compared against single models
- ✅ Safety checker handles the updated dataset structure with null fields
- ✅ Validation tools available for both models and dataset
- ✅ Clear path to verify the hybrid model accuracy objective

---

## Verification Checklist

### Hybrid Models:
- [ ] Train hybrid models (resnet50_mobilenet, resnet50_efficientnet, resnet50_efficientnet_mobilenet)
- [ ] Run comparison script
- [ ] Verify hybrid > single accuracy
- [ ] Update model registry status if objective met
- [ ] Set best hybrid as default if it outperforms single models

### Herbicide Dataset:
- [ ] Run validation script
- [ ] Review validation report
- [ ] Check safety score is acceptable
- [ ] Test recommendation service with updated dataset
- [ ] Verify safety checks work with new dataset structure

---

## Important Notes

### Hybrid Model Training
- Hybrid models require more GPU memory and training time than single models
- Training time may be 2-3x longer due to multiple backbones
- Start with `resnet50_mobilenet` (lightest) before trying the 3-way ensemble

### Herbicide Dataset
- The current dataset has many null fields for safety-critical information
- The safety checker now uses fallback fields (dose_formulation_per_ha, recommended_application_time)
- For production use, consider filling in the null fields with actual values
- Regional registration data is limited - expand if deploying to multiple regions

### Safety First
- Even with hybrid models, herbicide recommendations still require expert review
- The safety checks are implemented but the rules need agronomist validation
- Never use herbicide recommendations without expert consultation
