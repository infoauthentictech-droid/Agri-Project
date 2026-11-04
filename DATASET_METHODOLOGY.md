# Dataset Methodology

This document describes the four-tier dataset methodology used in the AgriVision/Groundnut AI project.

## Overview

The project uses a four-tier dataset approach to ensure robust AI model training:

1. **Primary Dataset** - Original field images
2. **Primary Augmented Dataset** - Augmented versions of primary dataset
3. **Secondary Dataset** - External public datasets
4. **Hybrid Dataset** - Combined dataset (primary + augmented + secondary)

---

## 1. Primary Dataset

### Description
The Primary Dataset consists of original, real-world images collected directly from the field for your specific project.

### What it is
- Raw photos you or your team take using smartphones, drones, or digital cameras in actual groundnut fields
- Unedited photos of healthy groundnut plants and various local weeds growing alongside them under natural sunlight

### Example
Photos taken with a smartphone in your local groundnut field showing:
- Healthy groundnut plants at different growth stages
- Various local weed species (nutsedge, pigweed, crabgrass, etc.)
- Natural lighting conditions (morning, noon, evening)
- Different soil types and field conditions

### Directory Structure
```
datasets/primary/
├── classification/
│   ├── train/
│   │   ├── crop/
│   │   └── weed/
│   ├── val/
│   │   ├── crop/
│   │   └── weed/
│   └── test/
│       ├── crop/
│       └── weed/
├── detection/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
└── segmentation/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

### Preparation Script
```bash
python ml/data/prepare_primary_dataset.py --task classification --input_dir /path/to/raw/images
python ml/data/prepare_primary_dataset.py --task detection --input_dir /path/to/raw/images
python ml/data/prepare_primary_dataset.py --task segmentation --input_dir /path/to/raw/images
```

### Characteristics
- **Source**: Your own field data
- **Authenticity**: High (real local conditions)
- **Variety**: Limited to your specific location and time
- **Size**: Typically 500-2000 images
- **Quality**: Variable (depends on capture conditions)

---

## 2. Primary Augmented Dataset

### Description
The Augmented Primary Dataset takes the original photos and artificially multiplies them using software tweaks to train the AI better.

### What it is
- The primary dataset modified by rotating, flipping, zooming, or changing the brightness of the original images
- Teaches the AI to recognize the plants from different angles or under different weather conditions without you having to take more field photos

### Example
Taking one photo of a groundnut plant from your primary dataset and creating 5 new versions:
- Flipped horizontally
- Rotated 90 degrees
- Darkened exposure
- Slightly blurred
- Increased contrast

### Augmentation Types
- **Geometric**: Horizontal flip, vertical flip, rotation (90°, 180°, 270°)
- **Photometric**: Brightness adjustment, contrast adjustment, saturation adjustment
- **Noise**: Gaussian blur, random noise
- **Color**: Color jitter (hue, saturation, brightness)

### Directory Structure
```
datasets/primary_augmented/
├── classification/
│   ├── train/
│   │   ├── crop/
│   │   └── weed/
│   ├── val/
│   │   ├── crop/
│   │   └── weed/
│   └── test/
│       ├── crop/
│       └── weed/
├── detection/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
└── segmentation/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

### Preparation Script
```bash
python ml/data/prepare_augmented_dataset.py --task classification --source_dir datasets/primary/classification --augmentation_factor 5
python ml/data/prepare_augmented_dataset.py --task detection --source_dir datasets/primary/detection --augmentation_factor 5
python ml/data/prepare_augmented_dataset.py --task segmentation --source_dir datasets/primary/segmentation --augmentation_factor 5
```

### Characteristics
- **Source**: Derived from primary dataset
- **Authenticity**: Medium (artificial variations)
- **Variety**: High (many synthetic variations)
- **Size**: Primary size × augmentation factor (e.g., 1000 × 5 = 5000)
- **Quality**: Consistent (controlled augmentations)

---

## 3. Secondary Dataset

### Description
The Secondary Dataset relies entirely on pre-existing data collected by other people, institutions, or researchers.

### What it is
- Publicly available image libraries, open-source agricultural datasets, or academic databases downloaded from the internet
- Thousands of labeled images of weeds and crops from various regions and conditions

### Example Sources
- **WeedMap**: Public repository of weed images
- **Kaggle**: Agricultural datasets with labeled images
- **Academic repositories**: Research datasets from universities
- **Open-source projects**: Community-contributed datasets

### Examples of External Datasets
- PlantVillage dataset (disease identification)
- Weed species identification datasets
- Crop classification datasets
- Agricultural robot vision datasets

### Directory Structure
```
datasets/secondary/
├── classification/
│   ├── train/
│   │   ├── crop/
│   │   └── weed/
│   ├── val/
│   │   ├── crop/
│   │   └── weed/
│   └── test/
│       ├── crop/
│       └── weed/
├── detection/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
└── segmentation/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

### Preparation Script
```bash
python ml/data/prepare_secondary_dataset.py --task classification --input_dir /path/to/external/dataset
python ml/data/prepare_secondary_dataset.py --task detection --input_dir /path/to/external/dataset
python ml/data/prepare_secondary_dataset.py --task segmentation --input_dir /path/to/external/dataset
```

### Characteristics
- **Source**: External public datasets
- **Authenticity**: Variable (depends on source)
- **Variety**: Very high (global diversity)
- **Size**: Large (thousands to millions of images)
- **Quality**: High (curated by researchers)

---

## 4. Hybrid Dataset

### Description
The Hybrid Dataset is a combined master dataset that blends multiple sources together to create a more robust AI model.

### What it is
- A mixture of your own unique field data (Primary/Augmented) and external public data (Secondary)
- Gives your AI the "local knowledge" of your specific groundnut fields while benefiting from the massive variety of global datasets

### Example
Combining:
- 1,000 photos you took yourself in your local area (Primary)
- 5,000 augmented versions of your photos (Primary Augmented)
- 4,000 downloaded public images of groundnuts and weeds (Secondary)
- **Total**: 10,000-image training set

### Combination Ratios
Default ratios (configurable):
- **Primary**: 30%
- **Primary Augmented**: 30%
- **Secondary**: 40%

### Directory Structure
```
datasets/hybrid/
├── classification/
│   ├── train/
│   │   ├── crop/
│   │   └── weed/
│   ├── val/
│   │   ├── crop/
│   │   └── weed/
│   └── test/
│       ├── crop/
│       └── weed/
├── detection/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
└── segmentation/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

### Preparation Script
```bash
python ml/data/create_hybrid_dataset.py --task classification --primary_ratio 0.3 --augmented_ratio 0.3 --secondary_ratio 0.4
python ml/data/create_hybrid_dataset.py --task detection --primary_ratio 0.3 --augmented_ratio 0.3 --secondary_ratio 0.4
python ml/data/create_hybrid_dataset.py --task segmentation --primary_ratio 0.3 --augmented_ratio 0.3 --secondary_ratio 0.4
```

### Characteristics
- **Source**: Combined from all sources
- **Authenticity**: High (local) + Medium (augmented) + Variable (external)
- **Variety**: Very high (local + global)
- **Size**: Largest (sum of all sources)
- **Quality**: Balanced (local authenticity + global diversity)

---

## Application to Herbicide Recommendation

### Training Phase
The AI uses the **Hybrid Dataset** to become highly accurate at identifying exactly which leaf belongs to a groundnut crop and which belongs to a weed.

**Why Hybrid?**
- **Primary**: Teaches the AI your specific local conditions
- **Augmented**: Teaches the AI to handle variations in lighting, angle, and weather
- **Secondary**: Teaches the AI about weed species you might not have in your local area but could encounter

### Inference Phase
Once the AI detects a specific weed type, the system references a lookup table to trigger a herbicide recommendation:
- **Example**: If the AI detects "nutsedge" (Cyperus spp.)
- **Action**: Lookup table recommends "halosulfuron-methyl" (selective herbicide for nutsedge in groundnut)
- **Safety**: Recommendation service checks crop compatibility, regional registration, and application timing

---

## Training with Different Dataset Types

### Classification Models
```bash
# Train with primary dataset only
python ml/classification/train_classifier.py --model efficientnet --epochs 30 --dataset_type primary

# Train with augmented dataset only
python ml/classification/train_classifier.py --model efficientnet --epochs 30 --dataset_type primary_augmented

# Train with secondary dataset only
python ml/classification/train_classifier.py --model efficientnet --epochs 30 --dataset_type secondary

# Train with hybrid dataset (recommended)
python ml/classification/train_classifier.py --model efficientnet --epochs 30 --dataset_type hybrid

# Train hybrid models for higher accuracy
python ml/classification/train_classifier.py --model resnet50_mobilenet --epochs 30 --dataset_type hybrid
python ml/classification/train_classifier.py --model resnet50_efficientnet --epochs 30 --dataset_type hybrid
python ml/classification/train_classifier.py --model resnet50_efficientnet_mobilenet --epochs 30 --dataset_type hybrid
```

### Detection Models
```bash
# Train with different dataset types
python ml/detection/train_yolo.py --epochs 100 --batch 16 --device 0 --dataset_type primary
python ml/detection/train_yolo.py --epochs 100 --batch 16 --device 0 --dataset_type primary_augmented
python ml/detection/train_yolo.py --epochs 100 --batch 16 --device 0 --dataset_type secondary
python ml/detection/train_yolo.py --epochs 100 --batch 16 --device 0 --dataset_type hybrid
```

### Segmentation Models
```bash
# Train with different dataset types
python ml/segmentation/train_yoloseg.py --epochs 100 --batch 16 --device 0 --dataset_type primary
python ml/segmentation/train_yoloseg.py --epochs 100 --batch 16 --device 0 --dataset_type primary_augmented
python ml/segmentation/train_yoloseg.py --epochs 100 --batch 16 --device 0 --dataset_type secondary
python ml/segmentation/train_yoloseg.py --epochs 100 --batch 16 --device 0 --dataset_type hybrid
```

---

## Best Practices

### 1. Always Start with Primary Dataset
- Collect your own field images first
- This gives the AI "local knowledge"
- Essential for your specific environmental conditions

### 2. Augment Primary Dataset
- Apply 3-5 augmentations per image
- Common augmentations: flip, rotate, brightness, contrast
- Increases training data without additional field work

### 3. Add Secondary Dataset for Diversity
- Download relevant public datasets
- Ensure classes match your target classes (crop, weed)
- Helps with weed species not present in your local area

### 4. Use Hybrid Dataset for Final Training
- Combines strengths of all sources
- Default ratio: 30% primary, 30% augmented, 40% secondary
- Adjust ratios based on your specific needs

### 5. Validate on Primary Test Set
- Always test on unseen primary data
- This measures real-world performance
- Avoid overfitting to secondary data

### 6. Track Dataset Lineage
- Record which dataset type was used for each model
- Update model registry with dataset source
- Essential for reproducibility

---

## Metadata

Each dataset directory contains a `metadata.json` file that tracks:
- Dataset type (primary, primary_augmented, secondary, hybrid)
- Task (classification, detection, segmentation)
- Source path
- Creation date
- Augmentation parameters (if applicable)
- Combination ratios (if hybrid)

Example metadata for hybrid dataset:
```json
{
  "dataset_type": "hybrid",
  "task": "classification",
  "sources": {
    "primary": "datasets/primary/classification",
    "primary_augmented": "datasets/primary_augmented/classification",
    "secondary": "datasets/secondary/classification"
  },
  "ratios": {
    "primary": 0.3,
    "primary_augmented": 0.3,
    "secondary": 0.4
  }
}
```

---

## Summary

| Dataset Type | Source | Authenticity | Variety | Size | Use Case |
|--------------|--------|--------------|---------|------|----------|
| Primary | Your field data | High | Low | Medium | Local knowledge |
| Primary Augmented | Augmented primary | Medium | High | Large | Increased training data |
| Secondary | External public | Variable | Very High | Very Large | Global diversity |
| Hybrid | Combined | High | Very High | Largest | **Recommended for training** |

**Recommended Training Workflow:**
1. Collect primary dataset from your fields
2. Augment primary dataset (5x factor)
3. Download relevant secondary datasets
4. Create hybrid dataset combining all sources
5. Train models on hybrid dataset
6. Validate on primary test set
7. Deploy with confidence in local performance
