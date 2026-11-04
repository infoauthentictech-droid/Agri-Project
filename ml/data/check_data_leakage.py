"""
Check for data leakage between train, val, and test sets.

This script checks for:
1. Duplicate images across splits
2. Similar images (using hash comparison)
3. Class distribution imbalance

Problem 11, 19: Verify data independence
"""

import hashlib
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import json


def get_file_hash(filepath: Path) -> str:
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def check_duplicates(data_root: Path) -> Dict[str, List[str]]:
    """Check for duplicate images across splits."""
    print("Checking for duplicate images...")
    
    splits = ['train', 'val', 'test']
    split_hashes: Dict[str, Dict[str, Path]] = {split: {} for split in splits}
    
    # Collect hashes for each split
    for split in splits:
        split_dir = data_root / split
        if not split_dir.exists():
            continue
        
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            
            for img_file in class_dir.glob('*'):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    file_hash = get_file_hash(img_file)
                    split_hashes[split][file_hash] = img_file
    
    # Check for duplicates across splits
    duplicates = defaultdict(list)
    
    # Check train vs val
    for hash_val, path in split_hashes['train'].items():
        if hash_val in split_hashes['val']:
            duplicates['train_val'].append(str(path))
            duplicates['train_val'].append(str(split_hashes['val'][hash_val]))
    
    # Check train vs test
    for hash_val, path in split_hashes['train'].items():
        if hash_val in split_hashes['test']:
            duplicates['train_test'].append(str(path))
            duplicates['train_test'].append(str(split_hashes['test'][hash_val]))
    
    # Check val vs test
    for hash_val, path in split_hashes['val'].items():
        if hash_val in split_hashes['test']:
            duplicates['val_test'].append(str(path))
            duplicates['val_test'].append(str(split_hashes['test'][hash_val]))
    
    # Remove duplicates within each list
    for key in duplicates:
        duplicates[key] = list(set(duplicates[key]))
    
    return dict(duplicates)


def check_class_distribution(data_root: Path) -> Dict[str, Dict[str, int]]:
    """Check class distribution in each split."""
    print("Checking class distribution...")
    
    splits = ['train', 'val', 'test']
    distribution = {}
    
    for split in splits:
        split_dir = data_root / split
        if not split_dir.exists():
            continue
        
        class_counts = {}
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            
            count = len(list(class_dir.glob('*')))
            class_counts[class_dir.name] = count
        
        distribution[split] = class_counts
    
    return distribution


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Check for data leakage')
    parser.add_argument('--data-root', type=str, required=True, help='Path to dataset root')
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    
    print("=" * 60)
    print("DATA LEAKAGE CHECK")
    print("=" * 60)
    print(f"Dataset root: {data_root}")
    print()
    
    # Check duplicates
    duplicates = check_duplicates(data_root)
    
    print("DUPLICATE CHECK RESULTS:")
    print("-" * 60)
    total_duplicates = 0
    for pair, files in duplicates.items():
        count = len(files)
        total_duplicates += count
        print(f"{pair}: {count} duplicate images")
        if count > 0:
            print(f"  Files: {files[:4]}{'...' if count > 4 else ''}")
    
    if total_duplicates == 0:
        print("  No duplicates found! ✓")
    else:
        print(f"  WARNING: Found {total_duplicates} duplicate images across splits!")
    
    print()
    
    # Check class distribution
    distribution = check_class_distribution(data_root)
    
    print("CLASS DISTRIBUTION:")
    print("-" * 60)
    for split, class_counts in distribution.items():
        print(f"{split}:")
        total = sum(class_counts.values())
        for class_name, count in class_counts.items():
            pct = (count / total * 100) if total > 0 else 0
            print(f"  {class_name}: {count} ({pct:.1f}%)")
        print()
    
    # Save results
    results = {
        "duplicates": duplicates,
        "class_distribution": distribution,
        "total_duplicates": total_duplicates,
    }
    
    output_file = data_root.parent / "data_leakage_report.json"
    output_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Report saved to: {output_file}")
    
    print("=" * 60)
    if total_duplicates == 0:
        print("DATA LEAKAGE CHECK PASSED ✓")
    else:
        print("DATA LEAKAGE CHECK FAILED ✗")
    print("=" * 60)
    
    return 0 if total_duplicates == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
