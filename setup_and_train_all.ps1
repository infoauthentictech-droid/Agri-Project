# ============================================================================
# AgriVision/Groundnut AI - Complete Dataset Setup and Training Script
# Four-Tier Dataset Methodology
# ============================================================================
# This script will:
# 1. Organize all datasets into the four-tier structure (Primary, Augmented Primary, Secondary, Hybrid)
# 2. Create Augmented Primary Dataset from Primary images
# 3. Combine all sources into Hybrid Dataset
# 4. Create train/val/test splits
# 5. Train all models with GPU enforcement
# ============================================================================

cd "D:\Flutter\PHD PROJECT\Agri Project"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "FOUR-TIER DATASET METHODOLOGY" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "1. Primary Dataset - Your original field images" -ForegroundColor Yellow
Write-Host "2. Augmented Primary Dataset - Artificially augmented primary images" -ForegroundColor Yellow
Write-Host "3. Secondary Dataset - External/public datasets" -ForegroundColor Yellow
Write-Host "4. Hybrid Dataset - Combined Primary + Augmented + Secondary" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "STEP 1: Creating New Dataset Directory Structure" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# Create new dataset directories
$directories = @(
    "datasets/primary/classification/crop",
    "datasets/primary/classification/weed",
    "datasets/primary/detection/images",
    "datasets/primary/detection/labels",
    "datasets/primary/segmentation/images",
    "datasets/primary/segmentation/labels",
    "datasets/primary_augmented/classification/crop",
    "datasets/primary_augmented/classification/weed",
    "datasets/primary_augmented/detection/images",
    "datasets/primary_augmented/detection/labels",
    "datasets/primary_augmented/segmentation/images",
    "datasets/primary_augmented/segmentation/labels",
    "datasets/secondary/classification/crop",
    "datasets/secondary/classification/weed",
    "datasets/secondary/detection/images",
    "datasets/secondary/detection/labels",
    "datasets/secondary/segmentation/images",
    "datasets/secondary/segmentation/labels",
    "datasets/hybrid/classification/crop",
    "datasets/hybrid/classification/weed",
    "datasets/hybrid/detection/images",
    "datasets/hybrid/detection/labels",
    "datasets/hybrid/segmentation/images",
    "datasets/hybrid/segmentation/labels"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "Created: $dir" -ForegroundColor Green
    } else {
        Write-Host "Exists: $dir" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "STEP 2: Preparing PRIMARY DATASET (Your Field Images)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# Copy crop images to primary classification
Write-Host "Copying crop images to primary/classification/crop..." -ForegroundColor Yellow
Get-ChildItem -Path "datasets/my_own_dataset/crop" -Recurse -Filter *.jpg -ErrorAction SilentlyContinue | ForEach-Object {
    Copy-Item $_.FullName -Destination "datasets/primary/classification/crop/" -Force
}
Get-ChildItem -Path "datasets/my_own_dataset/crop" -Recurse -Filter *.JPG -ErrorAction SilentlyContinue | ForEach-Object {
    Copy-Item $_.FullName -Destination "datasets/primary/classification/crop/" -Force
}
Get-ChildItem -Path "datasets/my_own_dataset/crop" -Recurse -Filter *.jpeg -ErrorAction SilentlyContinue | ForEach-Object {
    Copy-Item $_.FullName -Destination "datasets/primary/classification/crop/" -Force
}
Get-ChildItem -Path "datasets/my_own_dataset/crop" -Recurse -Filter *.png -ErrorAction SilentlyContinue | ForEach-Object {
    Copy-Item $_.FullName -Destination "datasets/primary/classification/crop/" -Force
}

# Copy weed images to primary classification
Write-Host "Copying weed images to primary/classification/weed..." -ForegroundColor Yellow
Get-ChildItem -Path "datasets/my_own_dataset/weed" -Filter *.jpg -ErrorAction SilentlyContinue | ForEach-Object {
    Copy-Item $_.FullName -Destination "datasets/primary/classification/weed/" -Force
}
Get-ChildItem -Path "datasets/my_own_dataset/weed" -Filter *.JPG -ErrorAction SilentlyContinue | ForEach-Object {
    Copy-Item $_.FullName -Destination "datasets/primary/classification/weed/" -Force
}
Get-ChildItem -Path "datasets/my_own_dataset/weed" -Filter *.jpeg -ErrorAction SilentlyContinue | ForEach-Object {
    Copy-Item $_.FullName -Destination "datasets/primary/classification/weed/" -Force
}
Get-ChildItem -Path "datasets/my_own_dataset/weed" -Filter *.png -ErrorAction SilentlyContinue | ForEach-Object {
    Copy-Item $_.FullName -Destination "datasets/primary/classification/weed/" -Force
}

# Copy mixed dataset for detection
Write-Host "Copying mixed dataset to primary/detection/images..." -ForegroundColor Yellow
Get-ChildItem -Path "datasets/my_own_dataset/Mixed dataset (crop and weed)" -Filter *.jpg -ErrorAction SilentlyContinue | ForEach-Object {
    Copy-Item $_.FullName -Destination "datasets/primary/detection/images/" -Force
}
Get-ChildItem -Path "datasets/my_own_dataset/Mixed dataset (crop and weed)" -Filter *.jpeg -ErrorAction SilentlyContinue | ForEach-Object {
    Copy-Item $_.FullName -Destination "datasets/primary/detection/images/" -Force
}

# Copy labeled/masked dataset for segmentation
Write-Host "Copying labeled dataset to primary/segmentation/images..." -ForegroundColor Yellow
if (Test-Path "datasets/public_dataset/crop_weed_labled_or_masked_images/images") {
    Get-ChildItem -Path "datasets/public_dataset/crop_weed_labled_or_masked_images/images" -Filter *.jpg -ErrorAction SilentlyContinue | ForEach-Object {
        Copy-Item $_.FullName -Destination "datasets/primary/segmentation/images/" -Force
    }
    Get-ChildItem -Path "datasets/public_dataset/crop_weed_labled_or_masked_images/images" -Filter *.png -ErrorAction SilentlyContinue | ForEach-Object {
        Copy-Item $_.FullName -Destination "datasets/primary/segmentation/images/" -Force
    }
}
if (Test-Path "datasets/public_dataset/crop_weed_labled_or_masked_images/labels") {
    Copy-Item -Path "datasets/public_dataset/crop_weed_labled_or_masked_images/labels/*" -Destination "datasets/primary/segmentation/labels/" -Recurse -Force
}

Write-Host ""
Write-Host "Primary Dataset Counts:" -ForegroundColor Green
Write-Host "  Crop: $((Get-ChildItem 'datasets/primary/classification/crop' -Filter *.jpg,*.JPG,*.jpeg,*.png -ErrorAction SilentlyContinue).Count) images"
Write-Host "  Weed: $((Get-ChildItem 'datasets/primary/classification/weed' -Filter *.jpg,*.JPG,*.jpeg,*.png -ErrorAction SilentlyContinue).Count) images"
Write-Host "  Detection: $((Get-ChildItem 'datasets/primary/detection/images' -Filter *.jpg,*.jpeg -ErrorAction SilentlyContinue).Count) images"
Write-Host "  Segmentation: $((Get-ChildItem 'datasets/primary/segmentation/images' -Filter *.jpg,*.png -ErrorAction SilentlyContinue).Count) images"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "STEP 3: Preparing SECONDARY DATASET (Public/External Data)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# Copy public dataset to secondary
Write-Host "Copying public dataset to secondary..." -ForegroundColor Yellow
if (Test-Path "datasets/public_dataset/crop_weed_mixed_public_dataset") {
    Get-ChildItem -Path "datasets/public_dataset/crop_weed_mixed_public_dataset" -Recurse -Filter *.jpg -ErrorAction SilentlyContinue | ForEach-Object {
        if ($_.Directory.Name -eq "crop" -or $_.Directory.Name -eq "weeds" -or $_.Directory.Name -eq "weed") {
            $targetDir = if ($_.Directory.Name -eq "crop") { "secondary/classification/crop" } else { "secondary/classification/weed" }
            Copy-Item $_.FullName -Destination "datasets/$targetDir/" -Force
        } else {
            Copy-Item $_.FullName -Destination "datasets/secondary/detection/images/" -Force
        }
    }
}

# Also copy from labeled public dataset
if (Test-Path "datasets/public_dataset/crop_weed_labled_or_masked_images/images") {
    Get-ChildItem -Path "datasets/public_dataset/crop_weed_labled_or_masked_images/images" -Filter *.jpg -ErrorAction SilentlyContinue | ForEach-Object {
        Copy-Item $_.FullName -Destination "datasets/secondary/detection/images/" -Force
    }
}
if (Test-Path "datasets/public_dataset/crop_weed_labled_or_masked_images/labels") {
    Copy-Item -Path "datasets/public_dataset/crop_weed_labled_or_masked_images/labels/*" -Destination "datasets/secondary/detection/labels/" -Recurse -Force
}

Write-Host ""
Write-Host "Secondary Dataset Counts:" -ForegroundColor Green
Write-Host "  Crop: $((Get-ChildItem 'datasets/secondary/classification/crop' -Filter *.jpg,*.JPG,*.jpeg,*.png -ErrorAction SilentlyContinue).Count) images"
Write-Host "  Weed: $((Get-ChildItem 'datasets/secondary/classification/weed' -Filter *.jpg,*.JPG,*.jpeg,*.png -ErrorAction SilentlyContinue).Count) images"
Write-Host "  Detection: $((Get-ChildItem 'datasets/secondary/detection/images' -Filter *.jpg -ErrorAction SilentlyContinue).Count) images"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "STEP 4: Creating Augmentation Script" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

$augmentationScript = @"
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random

class ImageAugmentor:
    """
    Creates augmented versions of images using various transformations
    """
    def __init__(self, augmentation_factor=5):
        self.augmentation_factor = augmentation_factor
        self.transformations = [
            'horizontal_flip',
            'vertical_flip',
            'rotate_90',
            'rotate_180',
            'rotate_270',
            'brightness_increase',
            'brightness_decrease',
            'contrast_increase',
            'contrast_decrease',
            'blur'
        ]

    def horizontal_flip(self, image):
        return cv2.flip(image, 1)

    def vertical_flip(self, image):
        return cv2.flip(image, 0)

    def rotate_90(self, image):
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    def rotate_180(self, image):
        return cv2.rotate(image, cv2.ROTATE_180)

    def rotate_270(self, image):
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    def brightness_increase(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, 50)
        v[v > 255] = 255
        final_hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    def brightness_decrease(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.subtract(v, 50)
        v[v < 0] = 0
        final_hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    def contrast_increase(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.add(l, 30)
        l[l > 255] = 255
        final_lab = cv2.merge((l, a, b))
        return cv2.cvtColor(final_lab, cv2.COLOR_LAB2BGR)

    def contrast_decrease(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.subtract(l, 30)
        l[l < 0] = 0
        final_lab = cv2.merge((l, a, b))
        return cv2.cvtColor(final_lab, cv2.COLOR_LAB2BGR)

    def blur(self, image):
        return cv2.GaussianBlur(image, (5, 5), 0)

    def augment_image(self, image_path, output_dir, base_name):
        """
        Apply random transformations to create augmented versions
        """
        image = cv2.imread(str(image_path))
        if image is None:
            return

        original_name = Path(image_path).stem
        ext = Path(image_path).suffix

        # Create augmented versions
        for i in range(self.augmentation_factor):
            # Randomly select transformation
            transform_name = random.choice(self.transformations)
            transformed = getattr(self, transform_name)(image)

            # Save augmented image
            aug_name = f"{original_name}_aug_{i}_{transform_name}{ext}"
            output_path = Path(output_dir) / aug_name
            cv2.imwrite(str(output_path), transformed)

    def augment_directory(self, source_dir, output_dir):
        """
        Augment all images in a directory
        """
        source_path = Path(source_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG']:
            image_files.extend(source_path.glob(ext))

        print(f"Augmenting {len(image_files)} images from {source_dir}...")
        for image_file in tqdm(image_files):
            self.augment_image(image_file, output_dir, image_file.stem)

        print(f"Created {len(image_files) * self.augmentation_factor} augmented images in {output_dir}")

if __name__ == '__main__':
    random.seed(42)

    print("=" * 60)
    print("Creating Augmented Primary Dataset")
    print("=" * 60)

    augmentor = ImageAugmentor(augmentation_factor=5)

    # Augment classification datasets
    print("\nAugmenting classification crop images...")
    augmentor.augment_directory(
        'datasets/primary/classification/crop',
        'datasets/primary_augmented/classification/crop'
    )

    print("\nAugmenting classification weed images...")
    augmentor.augment_directory(
        'datasets/primary/classification/weed',
        'datasets/primary_augmented/classification/weed'
    )

    # Augment detection datasets
    print("\nAugmenting detection images...")
    augmentor.augment_directory(
        'datasets/primary/detection/images',
        'datasets/primary_augmented/detection/images'
    )

    # Copy labels for detection (no augmentation needed for labels)
    if Path('datasets/primary/detection/labels').exists():
        import shutil
        shutil.copytree(
            'datasets/primary/detection/labels',
            'datasets/primary_augmented/detection/labels',
            dirs_exist_ok=True
        )
        print("Copied detection labels to augmented dataset")

    # Augment segmentation datasets
    print("\nAugmenting segmentation images...")
    augmentor.augment_directory(
        'datasets/primary/segmentation/images',
        'datasets/primary_augmented/segmentation/images'
    )

    # Copy labels for segmentation
    if Path('datasets/primary/segmentation/labels').exists():
        shutil.copytree(
            'datasets/primary/segmentation/labels',
            'datasets/primary_augmented/segmentation/labels',
            dirs_exist_ok=True
        )
        print("Copied segmentation labels to augmented dataset")

    print("\n" + "=" * 60)
    print("Augmented Primary Dataset Created!")
    print("=" * 60)
"@

$augmentationScript | Out-File -FilePath "ml\data\create_augmented_dataset.py" -Encoding UTF8
Write-Host "Created: ml\data\create_augmented_dataset.py" -ForegroundColor Green

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "STEP 5: Running Augmentation" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

.venv-gpu\Scripts\python.exe ml\data\create_augmented_dataset.py

Write-Host ""
Write-Host "Augmented Primary Dataset Counts:" -ForegroundColor Green
Write-Host "  Crop: $((Get-ChildItem 'datasets/primary_augmented/classification/crop' -Filter *.jpg,*.JPG,*.jpeg,*.png -ErrorAction SilentlyContinue).Count) images"
Write-Host "  Weed: $((Get-ChildItem 'datasets/primary_augmented/classification/weed' -Filter *.jpg,*.JPG,*.jpeg,*.png -ErrorAction SilentlyContinue).Count) images"
Write-Host "  Detection: $((Get-ChildItem 'datasets/primary_augmented/detection/images' -Filter *.jpg,*.jpeg -ErrorAction SilentlyContinue).Count) images"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "STEP 6: Creating Dataset Split Preparation Script" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

$splitScript = @"
import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

def create_train_val_test_split(source_dir, output_base_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Create train/val/test splits from a dataset directory
    """
    source_path = Path(source_dir)
    output_path = Path(output_base_dir)

    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_path / split).mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG']:
        image_files.extend(source_path.glob(ext))

    # Shuffle and split
    random.shuffle(image_files)
    total = len(image_files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_files = image_files[:train_end]
    val_files = image_files[train_end:val_end]
    test_files = image_files[val_end:]

    # Copy files
    for files, split in [(train_files, 'train'), (val_files, 'val'), (test_files, 'test')]:
        for file in files:
            shutil.copy2(file, output_path / split / file.name)

    print(f"Created splits for {source_dir}:")
    print(f"  Train: {len(train_files)}")
    print(f"  Val: {len(val_files)}")
    print(f"  Test: {len(test_files)}")

def create_classification_splits(dataset_type):
    """
    Create train/val/test splits for classification datasets
    """
    base_dir = Path(f'datasets/{dataset_type}/classification')

    for class_name in ['crop', 'weed']:
        source_dir = base_dir / class_name
        if source_dir.exists():
            output_dir = base_dir / class_name
            create_train_val_test_split(source_dir, output_dir)

def create_detection_splits(dataset_type):
    """
    Create train/val/test splits for detection datasets
    """
    base_dir = Path(f'datasets/{dataset_type}/detection')

    # For detection, we need to split images and their corresponding labels
    images_dir = base_dir / 'images'
    labels_dir = base_dir / 'labels'

    if not images_dir.exists():
        print(f"Images directory not found: {images_dir}")
        return

    # Get all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG']:
        image_files.extend(images_dir.glob(ext))

    # Shuffle and split
    random.seed(42)
    random.shuffle(image_files)
    total = len(image_files)
    train_end = int(total * 0.7)
    val_end = train_end + int(total * 0.15)

    train_files = image_files[:train_end]
    val_files = image_files[train_end:val_end]
    test_files = image_files[val_end:]

    # Create output directories
    for split in ['train', 'val', 'test']:
        (base_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        if labels_dir.exists():
            (base_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Copy files
    for files, split in [(train_files, 'train'), (val_files, 'val'), (test_files, 'test')]:
        for file in files:
            shutil.copy2(file, base_dir / 'images' / split / file.name)
            # Copy corresponding label if exists
            label_file = labels_dir / f"{file.stem}.txt"
            if label_file.exists():
                shutil.copy2(label_file, base_dir / 'labels' / split / f"{file.stem}.txt")

    print(f"Created detection splits for {dataset_type}:")
    print(f"  Train: {len(train_files)}")
    print(f"  Val: {len(val_files)}")
    print(f"  Test: {len(test_files)}")

def create_segmentation_splits(dataset_type):
    """
    Create train/val/test splits for segmentation datasets
    """
    # Segmentation uses same structure as detection
    create_detection_splits(dataset_type)

def create_yolo_yaml(dataset_type, task):
    """
    Create YOLO data.yaml file for detection/segmentation
    """
    if task == 'detection':
        yaml_content = f"""path: datasets/{dataset_type}/detection
train: images/train
val: images/val
test: images/test

names:
  0: crop
  1: weed
"""
    elif task == 'segmentation':
        yaml_content = f"""path: datasets/{dataset_type}/segmentation
train: images/train
val: images/val
test: images/test

names:
  0: crop
  1: weed
"""

    yaml_path = Path(f'datasets/{dataset_type}/{task}/data.yaml')
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"Created YOLO YAML: {yaml_path}")

if __name__ == '__main__':
    # Set random seed for reproducibility
    random.seed(42)

    print("=" * 60)
    print("Creating Train/Val/Test Splits for Classification")
    print("=" * 60)

    # Create splits for primary
    create_classification_splits('primary')
    create_classification_splits('primary_augmented')
    create_classification_splits('secondary')

    print("\n" + "=" * 60)
    print("Creating Train/Val/Test Splits for Detection")
    print("=" * 60)

    # Create splits for primary detection
    create_detection_splits('primary')
    create_detection_splits('primary_augmented')
    create_detection_splits('secondary')

    print("\n" + "=" * 60)
    print("Creating Train/Val/Test Splits for Segmentation")
    print("=" * 60)

    # Create splits for primary segmentation
    create_segmentation_splits('primary')
    create_segmentation_splits('primary_augmented')
    create_segmentation_splits('secondary')

    print("\n" + "=" * 60)
    print("Creating YOLO YAML Files")
    print("=" * 60)

    # Create YOLO YAML files
    create_yolo_yaml('primary', 'detection')
    create_yolo_yaml('primary', 'segmentation')
    create_yolo_yaml('primary_augmented', 'detection')
    create_yolo_yaml('primary_augmented', 'segmentation')
    create_yolo_yaml('secondary', 'detection')
    create_yolo_yaml('secondary', 'segmentation')

    print("\n" + "=" * 60)
    print("Dataset Split Preparation Complete!")
    print("=" * 60)
"@

$splitScript | Out-File -FilePath "ml\data\prepare_dataset_splits.py" -Encoding UTF8
Write-Host "Created: ml\data\prepare_dataset_splits.py" -ForegroundColor Green

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "STEP 7: Running Dataset Split Preparation" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

.venv-gpu\Scripts\python.exe ml\data\prepare_dataset_splits.py

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "STEP 8: Creating HYBRID DATASET (Primary + Augmented + Secondary)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

$hybridScript = @"
import os
import shutil
from pathlib import Path

def combine_datasets(sources, target, task, class_name=None):
    """
    Combine multiple source datasets into a target dataset
    """
    target_path = Path(target)
    target_path.mkdir(parents=True, exist_ok=True)

    for source in sources:
        source_path = Path(source)
        if not source_path.exists():
            print(f"Source not found: {source}")
            continue

        if class_name:
            # For classification
            source_dir = source_path / class_name
            target_dir = target_path / class_name
        else:
            # For detection/segmentation
            source_dir = source_path
            target_dir = target_path

        if not source_dir.exists():
            continue

        # Copy all files (including subdirectories like train/val/test)
        for item in source_dir.glob('*'):
            if item.is_file():
                dest_file = target_dir / item.name
                # Handle name conflicts
                counter = 1
                while dest_file.exists():
                    stem = item.stem
                    suffix = item.suffix
                    dest_file = target_dir / f"{stem}_{counter}{suffix}"
                    counter += 1
                shutil.copy2(item, dest_file)
            elif item.is_dir():
                # Copy subdirectories (train/val/test)
                dest_dir = target_dir / item.name
                dest_dir.mkdir(parents=True, exist_ok=True)
                for sub_item in item.glob('*'):
                    if sub_item.is_file():
                        dest_file = dest_dir / sub_item.name
                        counter = 1
                        while dest_file.exists():
                            stem = sub_item.stem
                            suffix = sub_item.suffix
                            dest_file = dest_dir / f"{stem}_{counter}{suffix}"
                            counter += 1
                        shutil.copy2(sub_item, dest_file)

        print(f"Copied from {source} to {target}")

if __name__ == '__main__':
    print("=" * 60)
    print("Creating HYBRID DATASET")
    print("Combining: Primary + Augmented Primary + Secondary")
    print("=" * 60)

    # Combine classification datasets
    print("\nCombining Classification Datasets...")
    for class_name in ['crop', 'weed']:
        sources = [
            'datasets/primary/classification',
            'datasets/primary_augmented/classification',
            'datasets/secondary/classification'
        ]
        target = 'datasets/hybrid/classification'
        combine_datasets(sources, target, 'classification', class_name)

    # Combine detection datasets
    print("\nCombining Detection Datasets...")
    sources = [
        'datasets/primary/detection',
        'datasets/primary_augmented/detection',
        'datasets/secondary/detection'
    ]
    target = 'datasets/hybrid/detection'
    combine_datasets(sources, target, 'detection')

    # Combine segmentation datasets
    print("\nCombining Segmentation Datasets...")
    sources = [
        'datasets/primary/segmentation',
        'datasets/primary_augmented/segmentation',
        'datasets/secondary/segmentation'
    ]
    target = 'datasets/hybrid/segmentation'
    combine_datasets(sources, target, 'segmentation')

    print("\n" + "=" * 60)
    print("HYBRID DATASET Created!")
    print("=" * 60)
"@

$hybridScript | Out-File -FilePath "ml\data\create_hybrid_dataset.py" -Encoding UTF8
Write-Host "Created: ml\data\create_hybrid_dataset.py" -ForegroundColor Green

.venv-gpu\Scripts\python.exe ml\data\create_hybrid_dataset.py

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "STEP 9: Verifying Final Dataset Structure" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

Write-Host ""
Write-Host "PRIMARY DATASET:" -ForegroundColor Green
Write-Host "  Crop: $((Get-ChildItem 'datasets/primary/classification/crop' -Filter *.jpg,*.JPG,*.jpeg,*.png -ErrorAction SilentlyContinue).Count) images"
Write-Host "  Weed: $((Get-ChildItem 'datasets/primary/classification/weed' -Filter *.jpg,*.JPG,*.jpeg,*.png -ErrorAction SilentlyContinue).Count) images"
Write-Host "  Detection: $((Get-ChildItem 'datasets/primary/detection/images' -Filter *.jpg,*.jpeg -ErrorAction SilentlyContinue).Count) images"

Write-Host ""
Write-Host "AUGMENTED PRIMARY DATASET:" -ForegroundColor Green
Write-Host "  Crop: $((Get-ChildItem 'datasets/primary_augmented/classification/crop' -Filter *.jpg,*.JPG,*.jpeg,*.png -ErrorAction SilentlyContinue).Count) images"
Write-Host "  Weed: $((Get-ChildItem 'datasets/primary_augmented/classification/weed' -Filter *.jpg,*.JPG,*.jpeg,*.png -ErrorAction SilentlyContinue).Count) images"
Write-Host "  Detection: $((Get-ChildItem 'datasets/primary_augmented/detection/images' -Filter *.jpg,*.jpeg -ErrorAction SilentlyContinue).Count) images"

Write-Host ""
Write-Host "SECONDARY DATASET:" -ForegroundColor Green
Write-Host "  Crop: $((Get-ChildItem 'datasets/secondary/classification/crop' -Filter *.jpg,*.JPG,*.jpeg,*.png -ErrorAction SilentlyContinue).Count) images"
Write-Host "  Weed: $((Get-ChildItem 'datasets/secondary/classification/weed' -Filter *.jpg,*.JPG,*.jpeg,*.png -ErrorAction SilentlyContinue).Count) images"
Write-Host "  Detection: $((Get-ChildItem 'datasets/secondary/detection/images' -Filter *.jpg -ErrorAction SilentlyContinue).Count) images"

Write-Host ""
Write-Host "HYBRID DATASET (Combined):" -ForegroundColor Green
Write-Host "  Crop: $((Get-ChildItem 'datasets/hybrid/classification/crop' -Filter *.jpg,*.JPG,*.jpeg,*.png -ErrorAction SilentlyContinue).Count) images"
Write-Host "  Weed: $((Get-ChildItem 'datasets/hybrid/classification/weed' -Filter *.jpg,*.JPG,*.jpeg,*.png -ErrorAction SilentlyContinue).Count) images"
Write-Host "  Detection: $((Get-ChildItem 'datasets/hybrid/detection/images' -Filter *.jpg,*.jpeg -ErrorAction SilentlyContinue).Count) images"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "STEP 10: Training Commands (Run These Manually)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

Write-Host ""
Write-Host "ACTIVATE VIRTUAL ENVIRONMENT:" -ForegroundColor Yellow
Write-Host ".venv-gpu\Scripts\Activate.ps1"
Write-Host ""

Write-Host "TRAIN CLASSIFICATION MODELS (GPU Enforced):" -ForegroundColor Yellow
Write-Host ""
Write-Host "# Single Models"
Write-Host ".venv-gpu\Scripts\python.exe ml\classification\train_classifier.py --model efficientnet --epochs 30 --dataset_type hybrid --device 0"
Write-Host ".venv-gpu\Scripts\python.exe ml\classification\train_classifier.py --model mobilenet --epochs 30 --dataset_type hybrid --device 0"
Write-Host ".venv-gpu\Scripts\python.exe ml\classification\train_classifier.py --model resnet50 --epochs 30 --dataset_type hybrid --device 0"
Write-Host ""
Write-Host "# Hybrid Models (Project Objective - Higher Accuracy)"
Write-Host ".venv-gpu\Scripts\python.exe ml\classification\train_classifier.py --model resnet50_mobilenet --epochs 30 --dataset_type hybrid --device 0"
Write-Host ".venv-gpu\Scripts\python.exe ml\classification\train_classifier.py --model resnet50_efficientnet --epochs 30 --dataset_type hybrid --device 0"
Write-Host ".venv-gpu\Scripts\python.exe ml\classification\train_classifier.py --model resnet50_efficientnet_mobilenet --epochs 30 --dataset_type hybrid --device 0"
Write-Host ""

Write-Host "TRAIN DETECTION MODEL (GPU Enforced):" -ForegroundColor Yellow
Write-Host ".venv-gpu\Scripts\python.exe ml\detection\train_yolo.py --epochs 100 --batch 16 --device 0 --dataset_type hybrid"
Write-Host ""

Write-Host "TRAIN SEGMENTATION MODEL (GPU Enforced):" -ForegroundColor Yellow
Write-Host ".venv-gpu\Scripts\python.exe ml\segmentation\train_yoloseg.py --epochs 100 --batch 16 --device 0 --dataset_type hybrid"
Write-Host ""

Write-Host "TRAIN PLANT GATE MODEL (GPU Enforced):" -ForegroundColor Yellow
Write-Host ".venv-gpu\Scripts\python.exe ml\gate\train_plant_gate.py --device 0"
Write-Host ""

Write-Host "COMPARE HYBRID VS SINGLE MODELS:" -ForegroundColor Yellow
Write-Host ".venv-gpu\Scripts\python.exe ml\experiments\compare_hybrid_models.py"
Write-Host ""

Write-Host "RUN TESTS:" -ForegroundColor Yellow
Write-Host ".venv-gpu\Scripts\python.exe -m pytest tests\test_ai_services.py tests\test_ml_regression.py -v -k `"not manifest`""
Write-Host ""

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Dataset Setup Complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Four-Tier Dataset Methodology:" -ForegroundColor Yellow
Write-Host "1. Primary Dataset - Your original field images" -ForegroundColor White
Write-Host "2. Augmented Primary Dataset - 5x augmented versions of primary" -ForegroundColor White
Write-Host "3. Secondary Dataset - External/public datasets" -ForegroundColor White
Write-Host "4. Hybrid Dataset - Combined Primary + Augmented + Secondary" -ForegroundColor White
Write-Host ""
Write-Host "Next: Run the training commands above to train all models with GPU" -ForegroundColor Yellow
