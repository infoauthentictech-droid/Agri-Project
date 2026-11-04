"""
Organize Existing Datasets Script

This script transfers existing datasets to the new four-tier methodology structure.
"""

import shutil
from pathlib import Path
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

def organize_primary_classification():
    """Organize existing crop and weed images into primary dataset."""
    primary_dir = PROJECT_ROOT / "datasets" / "primary" / "classification"
    primary_dir.mkdir(parents=True, exist_ok=True)
    
    # Create train/val/test splits
    for split in ["train", "val", "test"]:
        (primary_dir / split / "crop").mkdir(parents=True, exist_ok=True)
        (primary_dir / split / "weed").mkdir(parents=True, exist_ok=True)
    
    # Collect all crop images
    crop_images = []
    for subdir in ["groundnut_closeup_images", "groundnut_field_images"]:
        crop_path = PROJECT_ROOT / "datasets" / "my_own_dataset" / "crop" / subdir
        if crop_path.exists():
            for subsub in crop_path.iterdir():
                if subsub.is_dir():
                    crop_images.extend(list(subsub.glob("*.jpg")) + list(subsub.glob("*.JPG")) + list(subsub.glob("*.png")) + list(subsub.glob("*.jpeg")))
    
    # Collect all weed images
    weed_images = []
    weed_path = PROJECT_ROOT / "datasets" / "my_own_dataset" / "weed" / "weeds data"
    if weed_path.exists():
        weed_images.extend(list(weed_path.glob("*.jpg")) + list(weed_path.glob("*.JPG")) + list(weed_path.glob("*.png")) + list(weed_path.glob("*.jpeg")))
    
    # Also check mixed dataset (these contain weeds mixed with crops, treat as weed class)
    mixed_path = PROJECT_ROOT / "datasets" / "my_own_dataset" / "Mixed dataset (crop and weed)"
    if mixed_path.exists():
        weed_images.extend(list(mixed_path.glob("*.jpg")) + list(mixed_path.glob("*.JPG")) + list(mixed_path.glob("*.png")) + list(mixed_path.glob("*.jpeg")))
    
    print(f"Found {len(crop_images)} crop images")
    print(f"Found {len(weed_images)} weed images")
    
    # Shuffle and split
    random.shuffle(crop_images)
    random.shuffle(weed_images)
    
    # 70% train, 15% val, 15% test
    crop_train = crop_images[:int(len(crop_images) * 0.7)]
    crop_val = crop_images[int(len(crop_images) * 0.7):int(len(crop_images) * 0.85)]
    crop_test = crop_images[int(len(crop_images) * 0.85):]
    
    weed_train = weed_images[:int(len(weed_images) * 0.7)]
    weed_val = weed_images[int(len(weed_images) * 0.7):int(len(weed_images) * 0.85)]
    weed_test = weed_images[int(len(weed_images) * 0.85):]
    
    # Copy images
    for img in crop_train:
        shutil.copy2(img, primary_dir / "train" / "crop" / img.name)
    for img in crop_val:
        shutil.copy2(img, primary_dir / "val" / "crop" / img.name)
    for img in crop_test:
        shutil.copy2(img, primary_dir / "test" / "crop" / img.name)
    
    for img in weed_train:
        shutil.copy2(img, primary_dir / "train" / "weed" / img.name)
    for img in weed_val:
        shutil.copy2(img, primary_dir / "val" / "weed" / img.name)
    for img in weed_test:
        shutil.copy2(img, primary_dir / "test" / "weed" / img.name)
    
    print(f"Primary classification dataset organized:")
    print(f"  Crop: {len(crop_train)} train, {len(crop_val)} val, {len(crop_test)} test")
    print(f"  Weed: {len(weed_train)} train, {len(weed_val)} val, {len(weed_test)} test")

def organize_secondary_classification():
    """Organize public dataset into secondary dataset."""
    secondary_dir = PROJECT_ROOT / "datasets" / "secondary" / "classification"
    secondary_dir.mkdir(parents=True, exist_ok=True)
    
    # Create train split only for secondary
    (secondary_dir / "train" / "crop").mkdir(parents=True, exist_ok=True)
    (secondary_dir / "train" / "weed").mkdir(parents=True, exist_ok=True)
    
    # Check public dataset
    public_dir = PROJECT_ROOT / "datasets" / "public_dataset"
    if public_dir.exists():
        # Assume it has crop and weed subdirectories
        for class_name in ["crop", "weed"]:
            class_path = public_dir / class_name
            if class_path.exists():
                images = list(class_path.glob("*.jpg")) + list(class_path.glob("*.JPG")) + list(class_path.glob("*.png"))
                for img in images:
                    shutil.copy2(img, secondary_dir / "train" / class_name / img.name)
                print(f"  Secondary {class_name}: {len(images)} images")
    
    print("Secondary classification dataset organized")

def organize_primary_augmented_classification():
    """Create augmented version of primary dataset."""
    from PIL import Image, ImageEnhance, ImageOps
    import numpy as np
    
    source_dir = PROJECT_ROOT / "datasets" / "primary" / "classification"
    output_dir = PROJECT_ROOT / "datasets" / "primary_augmented" / "classification"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create train/val/test splits
    for split in ["train", "val", "test"]:
        (output_dir / split / "crop").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "weed").mkdir(parents=True, exist_ok=True)
    
    augmentation_factor = 5
    
    for split in ["train", "val", "test"]:
        for class_name in ["crop", "weed"]:
            class_dir = source_dir / split / class_name
            if not class_dir.exists():
                continue
            
            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.JPG")) + list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpeg"))
            print(f"  Augmenting {split}/{class_name}: {len(images)} images")
            
            for img_path in images:
                img = Image.open(img_path)
                
                # Original
                img.save(output_dir / split / class_name / img_path.name)
                
                # Augmented versions
                for i in range(augmentation_factor):
                    aug_img = img.copy()
                    
                    # Random transformation
                    choice = random.randint(0, 4)
                    if choice == 0:
                        aug_img = ImageOps.mirror(aug_img)
                    elif choice == 1:
                        aug_img = ImageOps.flip(aug_img)
                    elif choice == 2:
                        aug_img = aug_img.rotate(random.choice([90, 180, 270]))
                    elif choice == 3:
                        enhancer = ImageEnhance.Brightness(aug_img)
                        aug_img = enhancer.enhance(random.uniform(0.7, 1.3))
                    else:
                        enhancer = ImageEnhance.Contrast(aug_img)
                        aug_img = enhancer.enhance(random.uniform(0.7, 1.3))
                    
                    name = img_path.stem + f"_aug{i}" + img_path.suffix
                    aug_img.save(output_dir / split / class_name / name)
    
    print("Primary augmented dataset created")

def create_hybrid_classification():
    """Create hybrid dataset combining all sources."""
    output_dir = PROJECT_ROOT / "datasets" / "hybrid" / "classification"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create train/val/test splits
    for split in ["train", "val", "test"]:
        (output_dir / split / "crop").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "weed").mkdir(parents=True, exist_ok=True)
    
    primary_dir = PROJECT_ROOT / "datasets" / "primary" / "classification"
    augmented_dir = PROJECT_ROOT / "datasets" / "primary_augmented" / "classification"
    secondary_dir = PROJECT_ROOT / "datasets" / "secondary" / "classification"
    
    for split in ["train", "val", "test"]:
        for class_name in ["crop", "weed"]:
            combined_images = []
            
            # Primary
            p_dir = primary_dir / split / class_name
            if p_dir.exists():
                combined_images.extend(list(p_dir.glob("*.jpg")) + list(p_dir.glob("*.JPG")) + list(p_dir.glob("*.png")) + list(p_dir.glob("*.jpeg")))
            
            # Augmented
            a_dir = augmented_dir / split / class_name
            if a_dir.exists():
                combined_images.extend(list(a_dir.glob("*.jpg")) + list(a_dir.glob("*.JPG")) + list(a_dir.glob("*.png")) + list(a_dir.glob("*.jpeg")))
            
            # Secondary (only for train)
            if split == "train":
                s_dir = secondary_dir / "train" / class_name
                if s_dir.exists():
                    combined_images.extend(list(s_dir.glob("*.jpg")) + list(s_dir.glob("*.JPG")) + list(s_dir.glob("*.png")) + list(s_dir.glob("*.jpeg")))
            
            print(f"  Hybrid {split}/{class_name}: {len(combined_images)} images")
            
            for img in combined_images:
                shutil.copy2(img, output_dir / split / class_name / img.name)
    
    print("Hybrid classification dataset created")

def main():
    print("="*60)
    print("Organizing Existing Datasets")
    print("="*60)
    
    print("\n1. Organizing Primary Dataset...")
    organize_primary_classification()
    
    print("\n2. Organizing Secondary Dataset...")
    organize_secondary_classification()
    
    print("\n3. Creating Augmented Dataset...")
    organize_primary_augmented_classification()
    
    print("\n4. Creating Hybrid Dataset...")
    create_hybrid_classification()
    
    print("\n" + "="*60)
    print("Dataset Organization Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Train classification models on hybrid dataset")
    print("2. Train detection models on hybrid dataset")
    print("3. Train segmentation models on hybrid dataset")

if __name__ == "__main__":
    main()
